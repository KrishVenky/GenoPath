# GenoPath — Gemma 4 Good Hackathon
# Complete briefing for a fresh Claude Code session. Read this entire file before writing a single line of code.

---

## Core Philosophy — Read This First

**This is NOT an RL project.** Do not default to training as the solution.

Gemma 4 12B is already a capable reasoning model. The architecture wins by giving it:
1. A structured knowledge graph that constrains the action space to valid moves only
2. Native function calling so tool use is clean and reliable
3. Absent-phenotype exclusion signals so it can reason by elimination, not just pattern matching
4. A system prompt with clinical rules

That combination — capable model + constrained environment + structured signals — produces
a working clinical reasoning agent with zero training. Build that first. Demo that.

Fine-tuning E4B with Unsloth is an **optional parallel track** for the Unsloth special prize.
It is not the core of the project. If time is short, skip it. The 12B story stands alone.

---

## What We're Building

**GenoPath** — an offline-first clinical genomics reasoning agent that runs entirely on local hardware.
No cloud. No API keys. No internet required after setup.

A doctor or community health worker photographs a patient's clinical report on their **Pixel 9a phone**.
Gemma 4 E4B (running on the phone via LiteRT) reads the image and extracts the patient's phenotypes.
Those phenotypes are sent over local WiFi to a **laptop** running Gemma 4 12B via Ollama.
The laptop model uses Gemma 4's native function calling to navigate a 55,000-node gene-disease
knowledge graph — hopping from phenotype to disease to gene to variant — until it flags the
most likely causal genetic variant.

The family gets an answer. Nothing left the building.

**Hackathon:** Gemma 4 Good (Kaggle) — Deadline May 18, 2026 at 23:59 UTC
**Prize targets:** Main Track + Health & Sciences ($10K) + Ollama ($10K) + LiteRT ($10K) + Cactus ($10K)
**Optional prize:** Unsloth ($10K) — only if fine-tuning track is completed

---

## The Problem (State This Exactly This Way)

There are **6 million+ Variants of Uncertain Significance** in ClinVar — genetic variants where we don't
know if they cause disease. A clinical geneticist reviewing a rare disease case must cross-reference
dozens of high-pathogenicity candidates against a patient's phenotype profile. The bottleneck is not
data — the data exists. The bottleneck is **causal reasoning at the point of care**.

In resource-limited settings — rural hospitals in sub-Saharan Africa, community clinics in South Asia,
disaster-affected regions — there is no cloud genomics service, no specialist on call, no reliable
internet. A patient with unexplained symptoms that match a known rare disease profile goes undiagnosed
because the reasoning infrastructure doesn't reach them.

GenoPath brings that reasoning to any clinic with a phone and a laptop on a local network.
Both running open models. Both running offline.

---

## Hardware

- **Pixel 9a (Tensor G4, 8GB RAM)** — intake device. Gemma 4 E4B via LiteRT / AI Edge Gallery.
  Handles: clinical report image → phenotype extraction. Simple task, edge model, on-device.
  Phone will warm up and battery drains during sustained inference. Fine for demo, not for dev loop.

- **Desktop/laptop with 4060 Ti 16GB VRAM** — reasoning device. Gemma 4 12B via Ollama.
  Handles: phenotype → knowledge graph navigation → causal variant flag. Complex task, larger model.
  12B at Q4_K_M uses ~9GB VRAM — fits comfortably. Runs at ~20 tokens/sec.

**VRAM note:** Do NOT attempt Gemma 4 27B on the 4060 Ti. At 4-bit it needs ~17GB — will OOM.
Use 12B for inference. Use E4B only for fine-tuning track (on Colab, not locally).

This phone→laptop split is the **Cactus routing architecture**: intelligently route tasks between
models based on complexity. Simple extraction on the edge, complex reasoning on local compute.

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Phone inference | Gemma 4 E4B via LiteRT (AI Edge Gallery or MediaPipe LLM Inference API) | Phenotype extraction from clinical report image |
| Laptop inference | Gemma 4 12B via Ollama | Graph navigation with native function calling |
| Knowledge graph | Built in-process from ClinVar + HPO (pure Python, no external DB) | The domain knowledge the agent navigates |
| Phone→laptop comms | FastAPI local HTTP on laptop | Receive phenotypes from phone over WiFi |
| UI | Gradio on laptop | Demo interface, live graph trail visualisation |
| Fine-tuning (optional) | Unsloth + TRL GRPO on Colab A100/L4 | E4B adapter for Unsloth prize track only |

---

## Data Sources (Build From Scratch — No Prior Code Exists)

### 1. HPO Ontology
File: `data/hp.obo`
Source: https://hpo.jax.org/data/ontology (download hp.obo directly)
Contains: 19,389 human phenotype terms with definitions and parent-child relationships.
Parse: plain Python OBO parser. Read [Term] stanzas, extract id/name/is_a/def. Skip is_obsolete.
Result: `{HP:XXXXXXX: {name, parents, def}}` dict.

### 2. ClinVar Pathogenic Variants
File: `data/clinvar_pathogenic.tsv`
Source: Filtered from ClinVar variant_summary.txt (GRCh38, expert-reviewed, pathogenic/likely pathogenic only).
Script: `scripts/filter_clinvar.py` — reads raw ClinVar download, filters, writes TSV.
Columns: AlleleID, GeneSymbol, Name, Type, ClinicalSignificance, PhenotypeList, Chromosome, Start.
Result: ~92,000 high-confidence pathogenic variants across ~3,268 genes.

If `data/clinvar_pathogenic.tsv` already exists in the repo, use it directly.

---

## Knowledge Graph Design

Build in-memory from the two data files. Pure Python. No external DB. No Neo4j. No SQLite.

### Node types
- `phenotype` — HP:XXXXXXX terms from HPO
- `disease` — disease names from ClinVar PhenotypeList
- `gene` — gene symbols (MYH7, SCN1A, BRCA1…)
- `variant` — individual ClinVar variants (node ID: `VAR:{allele_id}`)
- `pathway` — coarse classification (cardiac, neurological, metabolic, cancer…)

### Edge rules (all bidirectional)
- phenotype ↔ phenotype (HPO parent-child, up to 5 ancestor levels)
- phenotype ↔ disease (from disease catalog HPO IDs)
- gene ↔ disease (from variant PhenotypeList)
- gene ↔ variant (all variants for that gene)
- gene ↔ pathway (hardcoded pathway map)

### Expected graph stats
~55,000 nodes, ~70,000 edge pairs. Loads in ~8 seconds, then singleton cached.

### Pathway classification (hardcode this)
```python
PATHWAY_MAP = {
    "MYH7": "cardiac", "MYBPC3": "cardiac", "SCN5A": "cardiac",
    "KCNQ1": "cardiac", "KCNH2": "cardiac", "TTN": "cardiac", "LMNA": "cardiac",
    "RYR2": "cardiac", "DSP": "cardiac", "PKP2": "cardiac",
    "SCN1A": "neurological", "MECP2": "neurological",
    "TSC1": "neurological", "TSC2": "neurological",
    "HTT": "neurological", "SNCA": "neurological", "LRRK2": "neurological",
    "PAH": "metabolic", "GBA1": "metabolic", "HEXA": "metabolic",
    "ATP7B": "metabolic", "LDLR": "metabolic", "APOB": "metabolic", "PCSK9": "metabolic",
    "BRCA1": "cancer", "BRCA2": "cancer", "TP53": "cancer",
    "MLH1": "cancer", "MSH2": "cancer",
    "CFTR": "pulmonary",
    "PKD1": "renal", "PKD2": "renal",
    "FBN1": "connective_tissue",
    "DMD": "musculoskeletal",
    "ABCA4": "ophthalmology", "USH2A": "ophthalmology",
    "HBB": "haematology", "F8": "haematology",
}
```

---

## Disease Catalog (Curated — Embed in Code)

Maps diseases to genes and HPO phenotype IDs. Used by case generator.

Minimum disease set:

**Cardiac:** Hypertrophic cardiomyopathy (MYH7/MYBPC3), Long QT syndrome (KCNQ1/KCNH2/SCN5A), Dilated cardiomyopathy (TTN/LMNA)
**Neurological:** Dravet syndrome (SCN1A), Rett syndrome (MECP2), Tuberous sclerosis (TSC1/TSC2)
**Metabolic:** Phenylketonuria (PAH), Gaucher disease (GBA1), Tay-Sachs (HEXA), Wilson disease (ATP7B)
**Pulmonary:** Cystic fibrosis (CFTR)
**Connective tissue:** Marfan syndrome (FBN1)
**Musculoskeletal:** Duchenne muscular dystrophy (DMD)
**Ophthalmology:** Stargardt disease (ABCA4)
**Lipid:** Familial hypercholesterolaemia (LDLR/APOB/PCSK9)
**Cancer (DECOYS only — task_types=[]):** BRCA1/BRCA2 (breast/ovarian), TP53 (Li-Fraumeni)

Each entry: `{disease, genes, hpo_ids, pathway, task_types}`.
task_types options: `["monogenic"]`, `["oligogenic"]`, `["monogenic", "phenotype_mismatch"]`, `[]` for decoy-only.

---

## Episode Design

Three task tiers. All run locally, in-process — no server, no WebSocket.

### monogenic
Single causal gene, 3–4 patient phenotypes, 5–8 candidate variants. Max 15 steps.

### oligogenic
2 causal genes, 5–7 phenotypes, 10–15 candidates. Max 25 steps. Agent must flag ALL causal variants.

### phenotype_mismatch
1 causal gene (cardiac/neuro) + a BRCA1/BRCA2/TP53 decoy in the candidate pool.
Patient phenotypes are all cardiac or neurological — zero cancer phenotypes.
The decoy has high pathogenicity score and looks maximally dangerous. Phenotype evidence rules it out.
This is the hardest task. Tests whether the model uses exclusion reasoning, not just pathogenicity rank.

---

## Action Space + Rewards

```python
ACTIONS = {
    "hop":            "+0.15 if relevant node, -0.05 if irrelevant. -0.01 per-step efficiency penalty.",
    "flag_causal":    "Terminal. +1.0 correct, -0.5 wrong. +0.2 timing bonus if before step 10.",
    "backtrack":      "+0.05 if last hop was irrelevant. -0.05 otherwise.",
    "summarise_trail": "0.0 neutral. No reward. Use sparingly — repeated use wastes steps."
}
# All rewards clamped to (0.01, 0.99)
# Terminal reward += overseer_score (0.0–0.3 for reasoning quality, computed locally — no LLM call)
```

Overseer (local, rule-based — NOT a second LLM call):
- -0.05 per hallucinated hop (node exists but not connected to current node)
- +0.05 if causal gene node was visited during episode
- -0.10 if fewer than 3 unique nodes visited (no real exploration)

---

## Absent Phenotypes — Key Innovation

For each candidate gene, compute HPO terms associated with that gene's diseases that the patient LACKS.
Show this in the observation. It enables exclusion reasoning.

Example: Patient has seizures + hypotonia. BRCA1 is in candidate pool.
BRCA1's diseases associate with HP:0003002 (Breast carcinoma), HP:0100615 (Ovarian neoplasm).
Patient has neither. Observation shows:
```
EXCLUSION SIGNALS:
  BRCA1: patient LACKS → Breast carcinoma, Ovarian neoplasm
  KCNQ1: patient LACKS → Prolonged QT interval
```

Gemma 4 12B is smart enough to use this signal without training. It's just good prompting + structured data.
Include absent phenotypes in every observation. Add small reward bonus (+0.05 per absent phenotype
explicitly mentioned in reasoning, capped at 2) — this rewards the model for using the signal.

---

## Gemma 4 Function Calling (Core Technical Differentiator)

Do NOT parse text JSON. Use Gemma 4 native function calling via Ollama.

```python
NARADA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "hop",
            "description": "Move to a connected node in the gene-disease knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Target node ID. Examples: GENE:SCN1A, HP:0001250, DIS:dravet_syndrome, VAR:12345, PATH:cardiac"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence: why move here?"
                    }
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_causal",
            "description": "Declare this variant as the causal diagnosis. TERMINAL — ends the episode. Only call when confident.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "Format: VAR:XXXXX"},
                    "reasoning": {"type": "string", "description": "Why is this variant causal given the patient's phenotypes?"}
                },
                "required": ["variant_id", "reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "backtrack",
            "description": "Return to the previous node. Use when current path is clearly wrong.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_trail",
            "description": "Get summary of visited nodes. Use sparingly — it costs a step.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]
```

Inference call:
```python
import ollama

def get_action(messages: list[dict]) -> dict:
    response = ollama.chat(
        model="gemma4:12b",
        messages=messages,
        tools=NARADA_TOOLS,
    )
    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        return {"action_type": call.function.name, **call.function.arguments}
    # Fallback — model didn't call a tool
    return {"action_type": "summarise_trail", "reasoning": "no tool call generated"}
```

---

## System Prompt for Gemma 4 Agent

```
You are a clinical genomics reasoning agent navigating a gene-disease knowledge graph to identify
the causal genetic variant for a rare disease patient.

CLINICAL RULES:
1. Follow the evidence chain: phenotype → disease → gene → variant.
2. EXCLUSION FIRST: If a candidate gene has many expected phenotypes that the patient LACKS,
   eliminate it before chasing its variants.
3. A high-pathogenicity cancer gene (BRCA1, BRCA2, TP53) is a DECOY when patient phenotypes
   are cardiac or neurological. Pathogenicity score does NOT override phenotype mismatch.
4. Flag early — correct flags before step 10 earn a bonus. Don't over-explore.
5. Backtrack when a path produces only irrelevant nodes.

You must call one of the provided tools at each step. No plain text responses.
```

---

## Observation Format (What the Agent Sees Each Step)

```
STEP 3/15 | Task: phenotype_mismatch

PATIENT PHENOTYPES:
  HP:0001250 — Seizures
  HP:0001263 — Global developmental delay
  HP:0001252 — Hypotonia

CURRENT NODE: [DISEASE] Dravet syndrome (DIS:dravet_syndrome)
  Neighbors (8): GENE:SCN1A, HP:0001250, HP:0002069, HP:0002373, DIS:lennox_gastaut, ...

TRAIL: HP:0001250 → DIS:dravet_syndrome

CANDIDATE VARIANTS:
  VAR:12345 | SCN1A  | frameshift | path=0.95 | Pathogenic       | Dravet syndrome
  VAR:67890 | BRCA1  | deletion   | path=0.95 | Pathogenic       | Breast carcinoma
  VAR:11111 | KCNQ1  | missense   | path=0.75 | Likely pathogenic | Long QT syndrome

EXCLUSION SIGNALS (expected-but-absent phenotypes per candidate gene):
  BRCA1: patient LACKS → Breast carcinoma, Ovarian neoplasm
  KCNQ1: patient LACKS → Prolonged QT interval, Atrial fibrillation

Step reward: +0.1500 | Cumulative: 0.2100
```

---

## Phone → Laptop Communication

Laptop runs a FastAPI server. Phone sends POST request over local WiFi. No internet needed.

```python
# src/server/intake_api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PhenotypePayload(BaseModel):
    raw_phenotypes: list[str]   # text extracted by E4B from image
    source: str = "image"       # "image" | "manual"

@app.post("/intake")
async def receive_phenotypes(payload: PhenotypePayload):
    # Match raw text to HPO IDs (fuzzy match against graph's HPO term names)
    # Start episode
    # Return episode_id + first observation
    ...
```

Phone sends: `POST http://{laptop_local_ip}:8000/intake`
Both devices on same WiFi. No internet.

---

## Phone Integration (LiteRT)

Pixel 9a, Tensor G4, 8GB RAM. Gemma 4 E4B at 4-bit = ~2.5GB model. Fits comfortably.

**Step 1:** Confirm AI Edge Gallery runs E4B on device (download model, test with a clinical report photo).
**Step 2:** For LiteRT prize eligibility, reference MediaPipe LLM Inference API in code:

```python
# This documents the on-device inference approach (runs on Android)
# from mediapipe.tasks.python.genai import inference as genai_inference
# options = genai_inference.LlmInferenceOptions(model_path='gemma4_e4b.bin', max_tokens=512)
# llm = genai_inference.LlmInference.create_from_options(options)
# phenotypes = llm.generate_response(prompt_with_image)
```

**Acceptable for 16-day timeline:** Demo uses AI Edge Gallery visually in the video.
Code contains the MediaPipe integration as a documented module with comments.
Judges need to see the architecture is real — a full Android APK is not required.

The E4B's job is simple: look at the image, extract phenotype terms as a list. That's it.
The 12B does all the reasoning. Keep this division clear.

---

## File Structure

```
genopath/
├── CLAUDE.md                    # This file — paste as first message in new session
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── hp.obo                   # Download from hpo.jax.org
│   └── clinvar_pathogenic.tsv   # Generated by scripts/filter_clinvar.py
│
├── scripts/
│   ├── filter_clinvar.py        # Filter raw ClinVar → clinvar_pathogenic.tsv
│   └── cognitive_load.py        # Compute: clinician manual review burden vs agent hops
│
├── src/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph.py             # NaradaGraph: build + query the knowledge graph
│   │   └── case_generator.py    # PatientCase: monogenic / oligogenic / phenotype_mismatch
│   │
│   ├── episode/
│   │   ├── __init__.py
│   │   ├── environment.py       # reset(), step(), reward, absent phenotype computation
│   │   └── models.py            # Pydantic: Observation, Action, StepResult, Variant, GraphNode
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py             # NARADA_TOOLS: Gemma 4 function definitions
│   │   ├── ollama_client.py     # Laptop inference: Ollama + function calling loop
│   │   └── intake.py            # Phone intake: image → phenotype extraction (LiteRT/MediaPipe)
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   └── intake_api.py        # FastAPI: phone → laptop phenotype handoff endpoint
│   │
│   └── ui/
│       ├── __init__.py
│       └── app.py               # Gradio: demo UI, live graph trail, exclusion signal display
│
└── training/                    # OPTIONAL — only if pursuing Unsloth prize
    └── gemma4_grpo.ipynb        # Colab: Unsloth GRPO fine-tuning of E4B
```

---

## Implementation Order (Do Exactly This Sequence)

**1. Graph first.** Everything depends on it.
- Parse hp.obo → HPO dict
- Parse clinvar_pathogenic.tsv → gene_variants dict
- Build NaradaGraph: add nodes, add edges, compute pathway nodes
- Test: `graph.get_neighbors("HP:0001250")` returns real disease + phenotype nodes

**2. Case generator second.**
- Implement generate_monogenic_case() — single gene, pick variants, pick phenotypes, compute absent phenotypes
- Implement generate_mismatch_case() — add BRCA1/BRCA2 decoy to candidate pool
- Test: generate 10 cases, print candidate variants, verify decoy is present in mismatch cases

**3. Episode logic third.**
- reset() → returns first observation with absent phenotypes populated
- step(action) → dispatches to hop/flag/backtrack/summarise, computes reward, returns next observation
- Test: run 5 episodes with hardcoded actions, verify rewards are in (0.01, 0.99)

**4. Agent fourth.**
- tools.py → NARADA_TOOLS definitions
- ollama_client.py → call Ollama with tools, extract tool_calls, fallback if no tool call
- Run one full episode with Gemma 4 12B zero-shot. Print every step. Does it navigate at all?
- Measure: what's the zero-shot score on monogenic? phenotype_mismatch?

**5. UI fifth.**
- Gradio app: two input modes (image upload / text), task selector, run button
- Live trail display: updates per step
- Exclusion signals panel
- Final result card

**6. Phone + server sixth.**
- FastAPI intake_api.py
- intake.py: given image path, call Ollama vision to extract phenotypes
- Test: upload clinical report image → get phenotype list → start episode

**7. Video + writeup last.**

---

## Cognitive Load Analysis

Write `scripts/cognitive_load.py`. Run it before making the video. Get this number.

For 10 randomly sampled cases per task tier:
- Count: candidate variants with pathogenicity_score >= 0.75 (what a clinician manually reviews)
- Count: hops in the optimal path from starting phenotype to causal variant
- Print: mean candidates reviewed vs mean agent hops

Target output:
```
Task                | Avg candidates (path>=0.75) | Avg optimal hops
monogenic           |          18.3               |       6.2
oligogenic          |          31.7               |      11.4
phenotype_mismatch  |          23.1               |       8.8
```

Put the monogenic number in the video: "A clinician reviews 18 high-risk candidates. GenoPath: 6 hops."

---

## Gradio UI Spec

Single page. Clean. Dark theme.

Left panel:
- Image upload (clinical report photo) OR text area (type phenotypes)
- Task type dropdown (monogenic / oligogenic / phenotype_mismatch)
- "Run GenoPath" button

Right panel — live during episode:
- Graph trail: `HP:0001250 → DIS:dravet_syndrome → GENE:SCN1A → VAR:12345`
- Step counter + current reward
- Exclusion signals box (updates as genes are encountered)
- Candidate variants table (highlight when flagged)

Bottom: result card — variant flagged, gene, disease, confidence, reasoning string.

The trail is the proof. Show it prominently. It demonstrates real graph navigation, not a lookup.

---

## Optional: Fine-Tuning Track (Unsloth Prize — $10K)

Only pursue this if core project is complete and time remains.

- Base model: `google/gemma-2-4b-it` or Unsloth's Gemma 4 E4B variant (check HF on session start)
- Method: GRPO via Unsloth + TRL on Colab A100 or L4 (NOT the local 4060 Ti)
- Same reward function as the episode logic above
- max_completion_length: **600** (critical — 300 causes zero learning, thinking tokens fill the budget)
- Curriculum: monogenic → oligogenic → phenotype_mismatch
- Push adapter to HF Hub, publish weights
- Compare: E4B fine-tuned vs E4B zero-shot vs 12B zero-shot — three-way benchmark

**Training bugs to avoid:**
1. TRL 0.24+ passes completions as `List[List[Dict]]`. Extract: `text = completion[-1]["content"] if isinstance(completion, list) else str(completion)`
2. Clear Unsloth compiled cache if max_completion_length changes: `shutil.rmtree("/content/unsloth_compiled_cache")`
3. `import nest_asyncio; nest_asyncio.apply()` before any asyncio in Colab
4. `os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"` before FastLanguageModel import
5. Check if Gemma 4 has thinking mode. If yes, disable it — same issue as Qwen3 eating all tokens

---

## Video Shot List (3 min max)

**0:00–0:25 — The Problem**
"6 million unclassified genetic variants. Most rare disease patients wait years for a diagnosis.
In rural clinics, they wait forever." Map. Rural hospital. Doctor with paper notes.

**0:25–1:15 — The Demo**
Pixel 9a. Doctor photographs a clinical report. Gemma 4 E4B on the phone extracts phenotypes.
Phone sends to laptop over local WiFi (show no-internet indicator). Gradio UI opens.
Graph trail builds: hop → hop → hop → flag_causal. Show exclusion signals panel: BRCA1 ruled out.
Correct variant flagged. Green result card.

**1:15–1:50 — The Architecture**
Diagram: Phone (E4B / LiteRT) → local WiFi → Laptop (12B / Ollama).
"Simple tasks on the edge. Complex reasoning on local compute. Zero cloud dependency."
Show function call in terminal: `{"name": "hop", "arguments": {"node_id": "GENE:SCN1A"}}`.

**1:50–2:20 — The Numbers**
Table: clinician reviews 18 high-risk candidates | GenoPath: 6 hops.
Phenotype mismatch task: BRCA1 decoy rejected. SCN1A causal variant flagged.
"Gemma 4 12B doesn't just find the right answer — it rules out the wrong one."

**2:20–2:50 — Optional: Training (if Unsloth track completed)**
"We also fine-tuned E4B for edge deployment. Same task. Smaller model. Phone-ready."
Loss curve. Before/after scores. HF link.

**2:50–3:00 — Vision**
"Any phone. Any laptop. Any clinic. Gemma 4 makes frontier reasoning local."
GitHub + HF links. Fade.

---

## Kaggle Writeup Outline (≤1,500 words)

1. **The Problem** (150w) — VUS burden, rural access gap, cloud dependency
2. **Solution Overview** (100w) — offline, phone→laptop routing, Gemma 4
3. **Knowledge Graph** (200w) — why graph navigation vs retrieval, 55K nodes, ClinVar + HPO
4. **Routing Architecture** (200w) — E4B phone intake, 12B laptop reasoning, why this split
5. **Gemma 4 Specific Features** (200w) — native function calling vs text parsing, LiteRT on Tensor G4, multimodal vision
6. **Absent Phenotype Exclusion** (150w) — what it is, why pure graph search can't do this
7. **Results** (150w) — zero-shot 12B scores, cognitive load comparison, decoy resistance
8. **Limitations** (100w) — honest: not clinical-grade, HPO matching is approximate, E4B quality ceiling
9. **How to Run** (100w) — ollama pull, pip install, python src/ui/app.py

---

## Prize Track Checklist

| Prize | Evidence Required | How to Show It |
|---|---|---|
| Main Track | Working demo + compelling video | Full Gradio demo + YouTube video |
| Health & Sciences | Medical domain, rural access story | Clinical cases, real ClinVar data, rural framing in video |
| Ollama ($10K) | Gemma 4 12B via Ollama | `ollama pull gemma4:12b` in README, Ollama terminal visible in demo |
| LiteRT ($10K) | E4B on Pixel 9a | MediaPipe code in intake.py, AI Edge Gallery on phone in video |
| Cactus ($10K) | Intelligent routing phone→laptop | Architecture diagram explicit in README, routing logic in code |
| Unsloth ($10K) — optional | Fine-tuned E4B adapter | HF weights published, training notebook with Unsloth |

---

## Day-by-Day Plan

### Days 1–2: Graph
- pip install, verify Ollama pulls gemma4:12b
- Build graph.py: parse hp.obo + clinvar_pathogenic.tsv, NaradaGraph class
- Verify: ~55K nodes, loads without error

### Days 3–4: Episode + Cases
- models.py: Pydantic schemas
- case_generator.py: monogenic + mismatch generators, absent phenotype computation
- environment.py: reset(), step(), reward logic

### Days 5–6: Agent
- tools.py + ollama_client.py
- Run first full zero-shot episode with 12B
- Get baseline scores on all three task types (10 episodes each)

### Days 7–8: Phone + Server
- intake_api.py (FastAPI)
- intake.py (image → phenotypes via Ollama vision)
- Test phone→laptop flow over local WiFi

### Days 9–10: Gradio UI
- Both input modes, live trail, exclusion signals, result card
- Test all three task types in UI

### Days 11–12: Polish + cognitive_load.py
- Run cognitive load analysis, get the numbers
- Fix any UI bugs, clean up code, write README

### Days 13–14: Video + Writeup
- Record 3-minute video (shot list above)
- Write Kaggle writeup

### Day 15: Fine-tuning (optional, if ahead of schedule)
- Colab: run E4B GRPO if time allows, for Unsloth prize

### Day 16: Submit
- All attachments attached. Writeup submitted. Verify everything is public.

---

## .env.example

```bash
# Laptop inference — Ollama runs locally, no key needed
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma4:12b

# Phone intake server
INTAKE_HOST=0.0.0.0
INTAKE_PORT=8000

# Optional: HF token for pushing fine-tuned weights (training track only)
HF_TOKEN=
```

---

## requirements.txt

```
# Core
ollama>=0.2.0
fastapi>=0.110.0
uvicorn>=0.29.0
gradio>=4.30.0
pydantic>=2.7.0
python-dotenv>=1.0.0
requests>=2.31.0
rich>=13.7.0

# Training only (Colab) — do not install locally
# unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
# trl>=0.9.0
# peft>=0.10.0
# accelerate>=0.30.0
# bitsandbytes>=0.43.0
# datasets>=2.19.0
```

---

## Definition of Done (Minimum for Submission)

- [ ] `python src/ui/app.py` runs and completes a monogenic episode with Gemma 4 12B
- [ ] Graph loads from data/ files in <15 seconds
- [ ] Exclusion signals appear correctly in phenotype_mismatch episodes
- [ ] Phone→laptop flow works (at minimum: documented + shown in video)
- [ ] Cognitive load numbers computed and in README
- [ ] GitHub repo public, code clearly shows function calling
- [ ] 3-minute YouTube video, no login required
- [ ] Kaggle writeup submitted with all 4 attachments before May 18 23:59 UTC
