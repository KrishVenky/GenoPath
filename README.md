# GenoPath — Offline Rare Disease Genomics Reasoning with Gemma 4

> **Gemma 4 Good Hackathon submission — Health & Sciences track**
> Gemma 4 running entirely on local hardware. No cloud. No API keys. No internet required.

---

## The Problem

There are **6 million+ Variants of Uncertain Significance** in ClinVar. A clinical geneticist reviewing a rare disease case must manually cross-reference dozens of high-pathogenicity candidates against a patient's phenotype profile — a process that takes hours and requires specialist expertise that most of the world doesn't have access to.

In resource-limited settings — rural hospitals, low-income country clinics, disaster zones — there is no cloud genomics service, no specialist on call, no reliable internet. The patient waits. The diagnosis doesn't come.

---

## What GenoPath Does

A fully local clinical reasoning agent powered by Gemma 4. The doctor photographs a patient's clinical report on **any Android phone running Gemma**. Gemma 4 E2B (running on the phone via LiteRT) extracts phenotype terms from the image. Those phenotypes are sent over local WiFi to a **desktop/laptop** running Gemma 4 E4B via Ollama.

The E4B model uses Gemma 4's **native function calling** to navigate a **55,000-node gene-disease knowledge graph** built from real ClinVar and HPO data — hopping from phenotype → disease → gene → variant nodes until it flags the most likely causal genetic variant.

Everything runs offline. Nothing leaves the building.

---

## Demo

**Baseline benchmark — gemma4:e4b, 10 episodes x 3 task types (seeds 1-10)**

| Task | Success Rate | Mean Reward | Avg Steps | Clinician Candidates | Agent BFS Hops | Burden Ratio |
|---|---|---|---|---|---|---|
| Monogenic | 30% | 0.333 | ~7 | 6.2 | 3.0 | 2.07x |
| Oligogenic | 10% | 0.192 | 25 | 9.5 | 3.0 | 3.17x |
| Phenotype Mismatch | 40% | 0.441 | ~5 | 8.2 | 3.0 | 2.73x |

Burden ratio = high-pathogenicity candidates a clinician must manually review divided by the agent\'s graph hops to reach the causal variant. Higher = greater reduction in cognitive load.

*Zero-shot, no fine-tuning. Full benchmark log: 
esults/baseline_e4b.json*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENOPATH SYSTEM                          │
├──────────────────────────┬──────────────────────────────────────┤
│   PHONE (any Android     │   DESKTOP (4060 Ti 16GB VRAM)       │
│   with Gemma support)    │   48GB RAM                          │
│                          │                                      │
│  ┌─────────────────┐     │  ┌─────────────────────────────────┐ │
│  │ Gemma 4 E2B     │     │  │  Gemma 4 E4B via Ollama         │ │
│  │ via LiteRT      │     │  │  Native function calling agent  │ │
│  │                 │     │  └────────────┬────────────────────┘ │
│  │ Input: photo of │     │               │ tool calls           │
│  │ clinical report │     │  ┌────────────▼────────────────────┐ │
│  │                 │     │  │  GenoPathEnvironment            │ │
│  │ Output: list of │     │  │  (episode logic, rewards)       │ │
│  │ phenotype terms │     │  └────────────┬────────────────────┘ │
│  └────────┬────────┘     │               │ observations         │
│           │ POST /intake │  ┌────────────▼────────────────────┐ │
│           │ local WiFi   │  │  GenoPathGraph                  │ │
│           └──────────────┼─▶│  55K nodes: HPO + ClinVar       │ │
│                          │  └─────────────────────────────────┘ │
│                          │                                      │
│                          │  ┌─────────────────────────────────┐ │
│                          │  │  Gradio UI                      │ │
│                          │  │  Live trail + exclusion signals │ │
│                          │  └─────────────────────────────────┘ │
└──────────────────────────┴──────────────────────────────────────┘
```

This phone→desktop split keeps extraction on the edge (E2B / LiteRT) and complex reasoning on local compute (E4B / Ollama) — works with any Android phone that can run Gemma.

---

## Gemma 4 Features Used

| Feature | How Used |
|---|---|
| **Native function calling** | Graph navigation actions are real tool calls — no JSON parsing, no regex |
| **Multimodal (vision)** | Clinical report image → phenotype extraction on phone before episode starts |
| **E2B edge model (LiteRT)** | On-device inference on any Android phone via LiteRT — fully offline intake |
| **E4B local deployment (Ollama)** | Zero cloud dependency, works fully offline |

---

## Key Innovation: Absent-Phenotype Exclusion

For each candidate gene, GenoPath computes HPO terms associated with that gene's diseases that the patient **lacks**. This enables elimination reasoning without any fine-tuning:

```
EXCLUSION SIGNALS:
  BRCA1: patient LACKS → Breast carcinoma, Ovarian neoplasm
  KCNQ1: patient LACKS → Prolonged QT interval, Atrial fibrillation
```

Gemma 4 E4B uses this signal zero-shot — structured data + a clinical system prompt is enough.

---

## Quickstart

```bash
# 1. Pull Gemma 4 E4B via Ollama (one-time, ~9.6GB download)
ollama pull gemma4:e4b

# 2. Install Python deps
pip install -r requirements.txt

# 3. Set up env
cp .env.example .env

# 4. Build graph data — only needed if clinvar_pathogenic.tsv is missing
#    (it is included in the repo, so skip this unless regenerating)
python scripts/filter_clinvar.py

# 5. Launch the app
python src/ui/app.py
# → Open http://localhost:7860

# 6. (Optional) Phone intake server — separate terminal
uvicorn src.server.intake_api:app --host 0.0.0.0 --port 8000
```

---

## Hardware Requirements

**Desktop / Laptop (reasoning device):**
- GPU: 4060 Ti 16GB VRAM or equivalent
- Gemma 4 E4B at Q4_K_M ≈ 9GB VRAM — fits comfortably
- **Do NOT attempt 27B** — needs ~17GB VRAM, will OOM on 16GB
- Runs at ~20 tok/s

**Phone (intake device) — any Android with Gemma support:**
- Any Android phone running Gemma via AI Edge Gallery or a custom LiteRT APK
- **E2B (2B 4-bit):** ~1.5GB — fits any phone with >=4GB RAM, nothing to close
- **E4B (4B 4-bit, "performance mode"):** ~2.5GB — needs >=6GB RAM, close heavy apps first
- Phone will warm up during sustained inference — normal, not damaging

---

## Cognitive Load Analysis

```
Task                | Avg candidates (path≥0.75) | Avg optimal hops | Ratio
monogenic           |         18.3               |       6.2        |  2.95x
oligogenic          |         31.7               |      11.4        |  2.78x
phenotype_mismatch  |         23.1               |       8.8        |  2.63x
```

A clinician manually reviews ~18 high-risk candidates for a monogenic case. GenoPath: 6 hops.

---

## Results

**Zero-shot Gemma 4 E4B baseline:**

| Task | Mean Score |
|---|---|
| monogenic | ~0.45 |
| oligogenic | ~0.24 |
| phenotype_mismatch | ~0.06 |

---

## Data Sources

- **HPO Ontology** (`data/hp.obo`) — Human Phenotype Ontology, 19,389 terms. [hpo.jax.org](https://hpo.jax.org/data/ontology)
- **ClinVar Pathogenic Variants** (`data/clinvar_pathogenic.tsv`) — ~92,000 GRCh38 expert-reviewed pathogenic variants, filtered from ClinVar variant_summary.txt by `scripts/filter_clinvar.py`.

---

## Project Structure

```
genopath/
├── src/
│   ├── graph/          # GenoPathGraph: build + query the knowledge graph
│   ├── episode/        # Environment, Pydantic models, reward logic
│   ├── agent/          # Ollama client, GENOPATH_TOOLS, phone intake
│   ├── server/         # FastAPI phone→desktop intake endpoint
│   └── ui/             # Gradio demo UI
├── scripts/
│   ├── filter_clinvar.py    # Filter raw ClinVar → clinvar_pathogenic.tsv
│   ├── sanity_check.py      # Integration smoke test (run after every change)
│   └── cognitive_load.py    # Clinician burden vs agent hops analysis
├── training/                # Optional: Unsloth GRPO notebook (Colab A100/L4)
├── data/                    # hp.obo + clinvar_pathogenic.tsv
└── results/                 # Benchmark outputs
```

---

## Prize Tracks

| Prize | Evidence |
|---|---|
| Main Track | Working Gradio demo + 3-min YouTube video |
| Health & Sciences | Real ClinVar/HPO data, rural clinic framing |
| Ollama ($10K) | `ollama pull gemma4:e4b` above, Ollama visible in demo video |
| LiteRT ($10K) | MediaPipe LLM Inference code in `src/agent/intake.py`, AI Edge Gallery on Android in video |
| Cactus ($10K) | Phone→desktop routing diagram above, routing logic in code |
| Unsloth ($10K, optional) | HF adapter published, training notebook, before/after benchmark |

---

## Links

- [Kaggle Writeup](https://kaggle.com/TODO)
- [Demo Video](https://youtube.com/watch?v=TODO)
- [HuggingFace Weights](https://huggingface.co/TODO)

---

## Author

Krishna Venkatesh — [GitHub](https://github.com/KrishVenky)
