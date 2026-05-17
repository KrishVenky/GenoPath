"""
GenoPath Gradio UI - dark theme, two-column layout.

Left panel:  input (text phenotypes or image upload) + task selector + run button
Right panel: live graph trail, step counter, exclusion signals, candidate variants table,
             final result card

Run:  python src/ui/app.py
"""
from __future__ import annotations

import os
import socket
import threading
from pathlib import Path
from typing import Generator, Optional

import sys

import gradio as gr

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Lazy graph init (warm up once at startup, not per request)
# ---------------------------------------------------------------------------
_graph = None
_graph_lock = threading.Lock()


def _get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from src.graph.graph import get_graph
                _graph = get_graph()
    return _graph


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Episode runner -- yields UI update tuples per step
# ---------------------------------------------------------------------------

def run_episode_stream(
    phenotype_text: str,
    task_type_label: str,
    image_path: Optional[str],
) -> Generator:
    """
    Yields (trail_md, step_md, exclusion_md, variants_md, result_md) tuples
    one per step so the UI updates live.
    """
    from src.episode.environment import GenoPathEnvironment
    from src.episode.models import GenoPathAction
    from src.agent.tools import format_observation, GENOPATH_TOOLS, SYSTEM_PROMPT
    import ollama

    task_map = {
        "Monogenic": "monogenic",
        "Oligogenic": "oligogenic",
        "Phenotype Mismatch": "phenotype_mismatch",
    }
    task_type = task_map.get(task_type_label, "monogenic")

    raw_phenotypes: list[str] = []
    if image_path:
        try:
            from src.agent.intake import extract_phenotypes_from_image
            matched = extract_phenotypes_from_image(image_path)
            raw_phenotypes = [m["name"] or m["raw"] for m in matched]
        except Exception as e:
            yield (f"**Error extracting phenotypes from image:** {e}", "", "", "", "")
            return
    elif phenotype_text.strip():
        raw_phenotypes = [p.strip() for p in phenotype_text.replace(",", "\n").splitlines() if p.strip()]

    graph = _get_graph()
    env = GenoPathEnvironment(graph)
    result = env.reset(task_type)

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(result.observation)},
    ]

    model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    exclusion_log: list[str] = []

    def _render_trail(obs) -> str:
        if not obs.trail:
            return "_No trail yet_"
        parts = []
        for node in obs.trail:
            icon = {
                "gene": "[gene]", "variant": "[var]", "phenotype": "[hpo]",
                "disease": "[dis]", "pathway": "[path]",
            }.get(node.type, "[*]")
            parts.append(f"{icon} **{node.name}** `{node.id}`")
        return " -> ".join(parts)

    def _render_step(obs, reward: float) -> str:
        return (
            f"**Step {obs.step}/{obs.max_steps}** | "
            f"Task: `{obs.task_type}` | "
            f"Step reward: `{reward:+.4f}` | "
            f"Cumulative: `{obs.cumulative_reward:.4f}`"
        )

    def _render_exclusions(obs) -> str:
        lines = []
        for gene, absent in obs.gene_absent_phenotypes.items():
            if absent:
                lines.append(f"**{gene}**: patient LACKS -- {', '.join(absent[:3])}")
        return "\n\n".join(lines) if lines else "_No exclusion signals yet_"

    def _render_variants(obs) -> str:
        rows = [
            "| Variant | Gene | Type | Path | Significance |",
            "|---------|------|------|------|-------------|",
        ]
        for v in obs.candidate_variants:
            rows.append(
                f"| `{v.id}` | **{v.gene}** | {v.variant_type} "
                f"| {v.pathogenicity_score:.2f} | {v.clinical_significance} |"
            )
        return "\n".join(rows)

    obs = result.observation
    yield (
        _render_trail(obs),
        _render_step(obs, 0.0),
        _render_exclusions(obs),
        _render_variants(obs),
        "_Episode running..._",
    )

    while not result.done:
        try:
            response = ollama.chat(
                model=model,
                messages=conversation,
                tools=GENOPATH_TOOLS,
            )
        except Exception as e:
            yield (
                _render_trail(obs),
                _render_step(obs, 0.0),
                _render_exclusions(obs),
                _render_variants(obs),
                f"**Ollama error:** {e}\n\nIs `{model}` pulled? Run: `ollama pull {model}`",
            )
            return

        _VALID = {"hop", "flag_causal", "backtrack", "summarise_trail"}
        real_calls = [c for c in (response.message.tool_calls or []) if c.function.name in _VALID]
        if real_calls:
            call = real_calls[0]
            action_dict = {"action_type": call.function.name, **call.function.arguments}
        else:
            action_dict = {"action_type": "summarise_trail", "reasoning": "no valid tool call"}

        action = GenoPathAction(**action_dict)

        if real_calls:
            conversation.append({
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": [
                    {"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}}
                    for c in real_calls
                ],
            })
        else:
            conversation.append({
                "role": "assistant",
                "content": response.message.content or "No tool call produced.",
            })

        result = env.step(action)
        obs = result.observation

        for gene, absent in obs.gene_absent_phenotypes.items():
            if absent and gene not in exclusion_log:
                exclusion_log.append(gene)

        if not result.done:
            if real_calls:
                conversation.append({"role": "tool", "content": format_observation(obs)})
            else:
                conversation.append({
                    "role": "user",
                    "content": (
                        "You must call one of the four tools: hop, flag_causal, backtrack, "
                        "or summarise_trail. Do not respond with plain text.\n\n"
                        + format_observation(obs)
                    ),
                })

        yield (
            _render_trail(obs),
            _render_step(obs, result.reward),
            _render_exclusions(obs),
            _render_variants(obs),
            "_Episode running..._",
        )

    # Final result card
    flagged_ids = env._flagged_allele_ids
    causal_ids = set(env._case.causal_allele_ids)
    is_success = result.reward > 0.5

    if flagged_ids:
        flagged_variants = [v for v in obs.candidate_variants if v.allele_id in flagged_ids]
        variant_lines = "\n".join(
            f"- `{v.id}` | **{v.gene}** | {v.variant_type} | path={v.pathogenicity_score:.2f}"
            for v in flagged_variants
        )
        status = "SUCCESS" if is_success else "INCORRECT"
        result_md = (
            f"## {status}\n\n"
            f"**Flagged variants:**\n{variant_lines}\n\n"
            f"**Final reward:** `{result.reward:.4f}`  \n"
            f"**Steps taken:** {obs.step}/{obs.max_steps}  \n"
            f"**Disease:** {env._case.disease_name}"
        )
        if env._case.decoy_gene and env._case.decoy_gene in {v.gene for v in flagged_variants}:
            result_md += f"\n\n**DECOY FLAGGED: {env._case.decoy_gene}** -- phenotype evidence was ignored"
    else:
        result_md = (
            f"## TIMEOUT\n\n"
            f"No variant flagged within {obs.max_steps} steps.  \n"
            f"**Final reward:** `{result.reward:.4f}` (exploration credit only)"
        )

    yield (
        _render_trail(obs),
        _render_step(obs, result.reward),
        _render_exclusions(obs),
        _render_variants(obs),
        result_md,
    )


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    local_ip = _local_ip()
    intake_url = f"http://{local_ip}:8000/intake"

    with gr.Blocks(title="GenoPath") as demo:
        gr.Markdown("# GenoPath\n*Offline rare disease genomics reasoning -- Gemma 4 E4B + Ollama*")

        with gr.Row():
            # ---- LEFT PANEL ------------------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### Patient Input")
                input_mode = gr.Radio(
                    ["Type phenotypes", "Photo of clinical report"],
                    value="Type phenotypes",
                    label="Input mode",
                )
                phenotype_text = gr.Textbox(
                    label="Phenotypes (comma or line separated)",
                    placeholder="seizures, hypotonia, global developmental delay",
                    lines=4,
                    visible=True,
                )
                image_input = gr.Image(
                    label="Clinical report photo",
                    type="filepath",
                    visible=False,
                )
                task_type = gr.Dropdown(
                    choices=["Monogenic", "Oligogenic", "Phenotype Mismatch"],
                    value="Monogenic",
                    label="Task type",
                )
                run_btn = gr.Button("Run GenoPath", variant="primary")

                with gr.Accordion("Connect phone", open=False):
                    gr.Markdown(
                        f"Send phenotypes from your phone:\n\n"
                        f"```\nPOST {intake_url}\n```\n\n"
                        f'Body: `{{"raw_phenotypes": ["seizures", ...], "source": "image"}}`\n\n'
                        f"Both devices must be on the same WiFi. No internet needed."
                    )

            # ---- RIGHT PANEL -----------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### Episode")
                step_info = gr.Markdown("_Click Run GenoPath to start_")
                trail_display = gr.Markdown(
                    "_Trail will appear here_",
                    elem_classes=["trail-box"],
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Exclusion Signals**")
                        exclusion_display = gr.Markdown("_No signals yet_")
                    with gr.Column():
                        gr.Markdown("**Candidate Variants**")
                        variants_display = gr.Markdown("_No variants yet_")
                result_display = gr.Markdown("", elem_classes=["result-box"])

        def _toggle_input(mode):
            return (
                gr.update(visible=(mode == "Type phenotypes")),
                gr.update(visible=(mode == "Photo of clinical report")),
            )

        input_mode.change(
            _toggle_input,
            inputs=[input_mode],
            outputs=[phenotype_text, image_input],
        )

        run_btn.click(
            fn=run_episode_stream,
            inputs=[phenotype_text, task_type, image_input],
            outputs=[trail_display, step_info, exclusion_display, variants_display, result_display],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Warming up GenoPathGraph (first load ~8s)...")
    _get_graph()
    print("Graph ready. Launching Gradio...")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
