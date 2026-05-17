# CLAUDE.md - GenoPath
# Read this, then SYSTEM.md, then TASKS.md before writing any code.

## Project
**GenoPath** - offline rare disease genomics reasoning. No cloud. No API keys.
Gemma 4 Good Hackathon (Kaggle). Deadline: May 18, 2026 23:59 UTC.
Built by Krishna Venkatesh.

Gemma 4 E4B (Ollama, desktop) navigates a 57,124-node gene-disease knowledge graph via native
function calling. Gemma 4 E2B (LiteRT, any Android phone) extracts phenotypes from clinical report photos.

**Baseline (gemma4:e4b, 10 eps x 3 tasks, 2026-05-03):**
monogenic 30%/0.333 | oligogenic 10%/0.192 | phenotype_mismatch 40%/0.441

---

## Hardware

**Desktop: 4060 Ti 16GB VRAM**
- gemma4:e4b (9.6GB) - fits cleanly, ~20 tok/s, fully GPU-resident
- gemma4:26b (18GB MoE) - DO NOT USE, benchmarked at 24 min/response (CPU offload)

**Phone: any Android with Gemma support (AI Edge Gallery or custom APK)**
- E2B (2B 4-bit, ~1.5GB) - fits any phone >=4GB RAM; default for reliability
- E4B (4B 4-bit, ~2.5GB) - fits phones >=6GB RAM; close heavy apps first

**Colab training: T4 (16GB) works for E2B GRPO fine-tuning**
- E2B at 4-bit + LoRA fits comfortably; batch_size=1, gradient_accumulation=8
- ~2-3x slower than A100 but fully functional; keep max_seq_length<=1024

---

## Architecture

```
Phone (E2B/LiteRT): photo -> phenotypes -> POST :8000/intake (local WiFi)
Desktop (E4B/Ollama): FastAPI receives -> GenoPathEnvironment.reset() ->
  agent loop: format_observation -> ollama.chat(tools) -> tool call -> env.step -> repeat
  -> Gradio UI: live trail, exclusion signals, result card
```

---

## Key Files

| File | Purpose |
|---|---|
| `src/graph/graph.py` | GenoPathGraph singleton: build + query (HPO + ClinVar) |
| `src/graph/case_generator.py` | Generate PatientCase for each task type |
| `src/episode/environment.py` | reset/step/reward, variant visit guard |
| `src/episode/models.py` | Pydantic v2 schemas |
| `src/agent/tools.py` | GENOPATH_TOOLS, SYSTEM_PROMPT, format_observation() |
| `src/agent/ollama_client.py` | run_episode(): full conversation loop |
| `src/agent/intake.py` | E2B/LiteRT phenotype extraction from image |
| `src/server/intake_api.py` | FastAPI: POST /intake from phone |
| `src/ui/app.py` | Gradio demo UI |
| `scripts/sanity_check.py` | Integration smoke test - run after every change |
| `scripts/run_benchmark.py` | 10 eps x 3 tasks benchmark -> results/baseline_e4b.json |
| `scripts/cognitive_load.py` | Clinician candidates vs agent hops ratio |
| `training/gemma4_grpo.ipynb` | Colab GRPO fine-tuning of E2B (T4-compatible) |

---

## Commands

```bash
pip install -r requirements.txt
ollama pull gemma4:e4b          # one-time, ~9.6GB
python scripts/sanity_check.py  # run after every change
python src/ui/app.py            # Gradio UI at localhost:7860
uvicorn src.server.intake_api:app --host 0.0.0.0 --port 8000  # phone intake
python scripts/run_benchmark.py # full 30-episode benchmark (~30 min)
python scripts/cognitive_load.py
```

---

## Recurring Bugs - Do Not Reintroduce

### Gemma 4 emits thought_output pseudo-tool-calls
Filter to valid actions only before dispatching:
```python
_VALID = {"hop", "flag_causal", "backtrack", "summarise_trail"}
real_calls = [c for c in (response.message.tool_calls or []) if c.function.name in _VALID]
action = {"action_type": real_calls[0].function.name, **real_calls[0].function.arguments} \
    if real_calls else {"action_type": "summarise_trail", "reasoning": "no valid tool call"}
```

### Ollama multi-turn tool use: role="tool", not role="user"
Tool results must use `role: "tool"`. Assistant turn must include the `tool_calls` list.
Violating this loses all context after step 1.
```python
conversation.append({"role": "assistant", "content": resp.message.content or "",
    "tool_calls": [{"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}}
                   for c in real_calls]})
conversation.append({"role": "tool", "content": format_observation(result.observation)})
# On fallback (no tool call), re-prompt with role="user" instead of role="tool"
```

### Variant visit guard (environment-enforced)
`_do_flag()` checks `f"VAR:{allele_id}" in self._trail_set` before accepting a flag.
Unvisited flags return R_HALLUCINATED (-0.10) and the episode continues.
The model must navigate to the VAR: node first. This is the #1 fix for premature flagging.

### PowerShell Set-Content corrupts UTF-8
Always write files from Python (`open(path, "w", encoding="utf-8")`), never via PowerShell
Set-Content. Em dashes, arrows, any non-ASCII becomes Windows-1252 mojibake.

### Graph singleton must warm up before first request
`get_graph()` takes ~8s. Call once at startup, not lazily per request.

### Rewards always clamped to (0.01, 0.99)
Never return raw 0.0 or 1.0. Always pass through `_clamp()`.

### Absent phenotypes for ALL candidate genes, not just causal
Decoy gene's absent phenotypes are the critical signal in phenotype_mismatch cases.

### Training: max_completion_length must be 600 (not 300)
300 fills the budget with thinking tokens, zero learning signal.

### Training: disable Gemma 4 thinking mode before GRPO
```python
_orig = tokenizer.apply_chat_template
def _no_think(*a, **kw): kw["enable_thinking"] = False; return _orig(*a, **kw)
tokenizer.apply_chat_template = _no_think
```

---

## Conventions
- Pydantic v2 everywhere. No raw dicts crossing module boundaries.
- All paths via `Path(__file__).parent`. GenoPathGraph is a singleton; GenoPathEnvironment is per-episode.
- Type hints on all public functions. No explanatory comments — only WHY comments.
- Rewards always clamped. Never raw floats.

---

## Handoff Protocol
Finish a task: mark `[x]` in TASKS.md, run `python scripts/sanity_check.py`, add a check if new code isn't covered.
Resume an `[~]` task: read the in-progress note, read listed source files, continue — do not restart.
