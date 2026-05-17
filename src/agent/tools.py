"""
Gemma 4 function call tool definitions and observation formatter for GenoPath.

GENOPATH_TOOLS: passed directly to ollama.chat(tools=...)
SYSTEM_PROMPT: injected as the first system message in every episode conversation
format_observation(): converts GenoPathObservation → formatted string for the model
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.episode.models import GenoPathObservation

# ---------------------------------------------------------------------------
# Tool definitions — four graph navigation actions
# ---------------------------------------------------------------------------

GENOPATH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "hop",
            "description": (
                "Move to a connected node in the knowledge graph. "
                "Use to follow phenotype→disease→gene→variant chains."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": (
                            "Target node ID. Format: "
                            "GENE:SCN1A | HP:0001250 | DIS:dravet_syndrome | VAR:12345 | PATH:cardiac"
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence: why this node?",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_causal",
            "description": (
                "Declare the causal variant. "
                "TERMINAL for monogenic/phenotype_mismatch — ends episode immediately. "
                "For oligogenic call once per suspected causal variant; "
                "episode ends at max_steps with partial credit. "
                "Only call when confident."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {
                        "type": "string",
                        "description": "Format: VAR:XXXXX",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why is this variant causal given the patient's phenotypes?",
                    },
                },
                "required": ["variant_id", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtrack",
            "description": "Return to the previous node. Use when the current path is clearly wrong.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_trail",
            "description": "Review the nodes visited so far. Costs a step — use sparingly.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical genomics reasoning agent. Your task: identify the causal \
genetic variant for a rare disease patient by navigating a gene-disease knowledge graph.

STRATEGY (follow this order):
1. Start at the HPO phenotype node. Hop to a disease node that matches the patient's symptoms.
2. From disease, hop to the most likely causal gene.
3. Check EXCLUSION SIGNALS before committing to any gene. If the gene expects phenotypes \
the patient LACKS, it is a decoy -- backtrack and try another gene.
4. From the confirmed gene, hop to its variant node. Read the variant's pathogenicity and type.
5. You MUST hop to the variant node (VAR:XXXXX) before calling flag_causal. \
The environment ENFORCES this: flagging an unvisited variant returns a -0.10 penalty \
and the episode continues. Navigate there first, then flag.

HARD RULES:
- BRCA1, BRCA2, TP53 are ALWAYS decoys when phenotypes are cardiac or neurological. \
Pathogenicity score does NOT override phenotype evidence -- check exclusion signals first.
- You MUST call one of the four tools at every step. Never respond with plain text.
- Backtrack if a node has no useful neighbors. Do not revisit nodes already in the trail.\
"""

# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def format_observation(obs: GenoPathObservation) -> str:
    steps_left = obs.max_steps - obs.step
    lines: List[str] = [
        f"STEP {obs.step}/{obs.max_steps} | Task: {obs.task_type} | Steps remaining: {steps_left}",
        "",
        "PATIENT PHENOTYPES:",
    ]
    for hid, name in zip(obs.patient_phenotypes, obs.phenotype_names):
        lines.append(f"  {hid} -- {name}")

    n = obs.current_node
    lines += [
        "",
        f"CURRENT NODE: [{n.type.upper()}] {n.name} ({n.id})",
        f"  Neighbors ({len(n.connected_node_ids)}): "
        + ", ".join(n.connected_node_ids[:10])
        + (" ..." if len(n.connected_node_ids) > 10 else ""),
    ]

    if obs.trail:
        trail_str = " -> ".join(
            f"{t.name}({t.id})" for t in obs.trail[-5:]
        )
        lines.append(f"  Trail: {trail_str}")

    lines += ["", "CANDIDATE VARIANTS:"]
    for v in obs.candidate_variants:
        lines.append(
            f"  {v.id} | {v.gene} | {v.variant_type} | "
            f"path={v.pathogenicity_score:.2f} | {v.clinical_significance}"
        )

    if obs.gene_absent_phenotypes:
        lines += [
            "",
            "EXCLUSION SIGNALS (gene has these expected phenotypes -- patient LACKS them):",
        ]
        for gene, absent in obs.gene_absent_phenotypes.items():
            if absent:
                lines.append(
                    f"  {gene}: patient LACKS -> {', '.join(absent[:3])}"
                )

    lines.append(
        f"\nStep reward: {obs.step_reward:+.4f} | Cumulative: {obs.cumulative_reward:.4f}"
    )

    # Hint to guide the model's next action
    trail_types = {t.type for t in obs.trail} if obs.trail else set()
    current_type = obs.current_node.type
    if current_type == "variant":
        lines.append(
            "HINT: You are ON a variant node. If its gene matches patient phenotypes and "
            "exclusion signals are clear, call flag_causal(variant_id) now."
        )
    elif current_type == "gene":
        lines.append(
            "HINT: You are on a gene node. Check EXCLUSION SIGNALS above. "
            "If signals are absent/clear, hop to one of its variant neighbors."
        )
    elif current_type in ("phenotype", "disease"):
        lines.append(
            "HINT: Hop toward a gene node that causes this condition. "
            "Avoid genes with strong exclusion signals."
        )

    return "\n".join(lines)
