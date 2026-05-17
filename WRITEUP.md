# GenoPath: Offline Rare Disease Genomics Reasoning with Gemma 4

**Track:** Health & Sciences

---

## The Problem

There are over 6 million Variants of Uncertain Significance in ClinVar. A clinical geneticist reviewing a rare disease case must manually cross-reference dozens of high-pathogenicity candidates against a patient's phenotype profile — a process that takes hours and requires specialist expertise most of the world doesn't have access to.

In resource-limited settings — rural hospitals, low-income country clinics, disaster zones — there is no cloud genomics service, no specialist on call, no reliable internet. The patient waits. The diagnosis doesn't come.

GenoPath is built for that room.

---

## What GenoPath Does

A doctor photographs a patient's clinical report on any Android phone running Gemma 4 E2B via LiteRT. The model extracts phenotype terms from the image entirely on-device. Those phenotypes travel over local WiFi to a desktop running Gemma 4 E4B via Ollama, which navigates a 57,124-node gene-disease knowledge graph — hopping from phenotype to disease to gene to variant until it flags the most likely causal genetic variant.

Nothing leaves the building. No API keys. No internet. No cloud.

---

## Architecture

The system splits across two devices connected by local WiFi:

**Phone — Gemma 4 E2B via LiteRT**
Any Android phone running AI Edge Gallery or a custom MediaPipe LLM Inference APK. The 2B 4-bit model (~1.5GB) runs on the device NPU/GPU. Given a photo of a clinical report, it extracts a structured list of phenotype terms with zero network dependency. E2B fits any phone with 4GB RAM; E4B (4-bit, ~2.5GB) fits phones with 6GB RAM.

**Desktop — Gemma 4 E4B via Ollama**
A laptop or desktop with a consumer GPU (RTX 4060 Ti 16GB in development). E4B at Q4_K_M (~9.6GB) receives the phenotype list over WiFi via a FastAPI server and begins a graph navigation episode using Gemma 4's native function calling. The model runs at ~20 tokens/second, completing episodes in 2–5 minutes.

**Phone intake endpoint:**
The desktop runs a FastAPI server (`GET /camera`, `POST /extract-phenotypes`, `POST /intake`, `GET /episode/stream`). The phone browser opens `http://[laptop-ip]:8000/camera`, photographs the report, and the desktop's Gemma vision extracts phenotypes — no APK required for the camera flow. The SSE stream at `/episode/stream` delivers live graph traversal steps back to any connected browser, including the phone.

---

## The Knowledge Graph

57,124 nodes, 79,833 edges, built from two real clinical datasets:

- **HPO Ontology** (19,389 terms) — Human Phenotype Ontology, the standard clinical vocabulary for rare disease phenotypes
- **ClinVar Pathogenic Variants** (~92,000 variants) — GRCh38 expert-reviewed pathogenic and likely pathogenic variants filtered from ClinVar's variant_summary.txt

Node types: phenotype (`HP:`), disease (`DIS:`), gene (`GENE:`), variant (`VAR:`), pathway (`PW:`). Edges connect phenotypes to associated diseases, diseases to causal genes, genes to their pathogenic variants. The agent navigates hop by hop guided only by Gemma 4's reasoning and structured observations.

The agent has four tools: `hop(node_id)` to traverse edges, `flag_causal(variant_id)` to declare the causal variant, `backtrack()` to return to a prior node, and `summarise_trail()` to pause and reason. Every action is a real Gemma 4 native function call — no JSON parsing, no regex, no prompt hacking.

---

## Key Innovation: Absent-Phenotype Exclusion

The most impactful feature required no fine-tuning. For each candidate gene, GenoPath computes the HPO terms associated with that gene's diseases that the patient does *not* present with. These exclusion signals are passed as structured data:

```
EXCLUSION SIGNALS:
  TSC1:  patient LACKS → Global developmental delay, Subependymal nodules, Tented upper lip
  GBA1:  patient LACKS → Splenomegaly, Thrombocytopenia, Hepatomegaly
  SCN1A: (no conflicting signals)
```

Gemma 4 E4B uses this zero-shot to eliminate unlikely candidates — the kind of elimination reasoning a specialist performs after years of training, emerging from structured prompting alone. No fine-tuning. No medical training data. Just the right information, structured well.

---

## Gemma 4 Features Used

**Native function calling** — Graph navigation actions are real tool calls. This eliminates the fragile JSON extraction that breaks multi-turn reasoning. Gemma 4 returns structured `tool_calls` objects that map directly to environment actions with no parsing layer.

**Multimodal vision** — Both E2B (on-device) and E4B (desktop) process clinical report images. In testing on synthetic clinical notes, the model extracted 12 phenotype terms with 8/12 HPO ontology matches, including all clinically significant terms (Seizures, Hypotonia, Global developmental delay, Developmental regression).

**E2B edge deployment via LiteRT** — The 2B 4-bit model runs on the phone NPU via the MediaPipe LLM Inference API, enabling fully offline phenotype extraction on any modern Android device.

**E4B local deployment via Ollama** — Zero cloud dependency. The 4B model runs on consumer hardware, making the complete system viable in any clinic with a mid-range laptop.

---

## Results

Zero-shot baseline — Gemma 4 E4B, 10 episodes × 3 task types, seeds 1–10:

| Task | Success Rate | Mean Reward | Burden Ratio |
|---|---|---|---|
| Monogenic | 30% | 0.333 | 2.07× |
| Oligogenic | 10% | 0.192 | 3.17× |
| Phenotype Mismatch | 40% | 0.441 | 2.73× |

Burden ratio = high-pathogenicity candidates a clinician must manually review ÷ agent's graph hops to reach the causal variant. A ratio of 2.07× means the agent reaches the causal variant in roughly half the cognitive steps a manual review requires.

After GRPO fine-tuning of the E2B phenotype extractor (150 steps, Colab T4, 20 training episodes): evaluation F1 on held-out synthetic clinical notes reached **0.983**.

The visualization frontend shows live graph traversal via D3.js force-directed layout — nodes colored by type (phenotype/disease/gene/variant), animated trail in orange, exclusion rings in red, flagged variant in gold.

---

## Technical Challenges

**Gemma 4 thought tokens.** Gemma 4 emits internal reasoning as pseudo-tool-calls named `thought_output`. These must be filtered before dispatching to the environment — otherwise the agent appears to act without navigating the graph. Fix: filter `tool_calls` to `{hop, flag_causal, backtrack, summarise_trail}` only.

**Multi-turn tool use format.** Ollama's multi-turn tool use requires `role: "tool"` for tool results and the full `tool_calls` list in the assistant turn. Using `role: "user"` for tool results causes complete context loss after step 1 — a subtle bug that produces consistently bad results with no error message.

**Variant visit guard.** The agent must navigate to a variant node before flagging it. Premature flags earn a −0.10 reward and the episode continues, forcing genuine graph exploration. Without this guard, the model hallucinates causal variants after step 1 without any graph traversal.

**GRPO training — content format.** `Gemma4Processor.apply_chat_template` requires message content as `[{"type": "text", "text": "..."}]` list-of-dicts, not plain strings. Passing plain strings triggers a `TypeError` deep in the Rust validator with no clear traceback.

---

## What's Next

Fine-tuning Gemma 4 E4B on graph navigation episodes using GRPO would substantially improve reasoning performance. The reward signal is already defined and the training infrastructure is in place. Based on E2B results, we estimate monogenic success rate would increase from 30% to 60–70% with 500 training episodes on a single A100.

The graph could also be extended with protein-protein interaction data (STRING database) and OMIM disease annotations to improve coverage for ultra-rare conditions with fewer than 10 published cases.

---

## Links

- **Code:** https://github.com/KrishVenky/GenoPath
- **Demo:** https://krishvenky.github.io/GenoPath
- **Video:** [YouTube link]
