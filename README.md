# Narada Edge — Offline Rare Disease Diagnosis with Gemma 4

> **Gemma 4 Good Hackathon submission — Health & Sciences track**
> Gemma 4 running locally on any laptop. No cloud. No API keys. No internet required.

---

## The Problem

There are **6 million+ Variants of Uncertain Significance** in ClinVar. A clinical geneticist reviewing a rare disease case must manually cross-reference dozens of high-pathogenicity candidates against a patient's phenotype profile — a process that takes hours and requires specialist expertise that most of the world doesn't have access to.

In resource-limited settings: rural hospitals, low-income country clinics, disaster zones — there is no cloud, no specialist network, no genomics service. The patient waits. The diagnosis doesn't come.

---

## What Narada Edge Does

A fully local clinical reasoning agent powered by Gemma 4. The doctor provides a patient's phenotype list (typed, or photographed from a clinical report). Gemma 4 navigates a **55,000-node gene-disease knowledge graph** built from real ClinVar and HPO data, using **native function calling** to hop between phenotype → disease → gene → variant nodes until it identifies the most likely causal variant.

Everything runs on the laptop. Gemma 4 via Ollama. Graph in memory. Zero cloud dependency.

---

## Demo

[![Watch the demo](docs/cover.png)](https://youtube.com/watch?v=TODO)

| Task | Agent Steps | Clinician Manual Review | Result |
|---|---|---|---|
| Monogenic (Dravet syndrome) | 7 hops | 43 high-path candidates | ✓ SCN1A causal variant flagged |
| Oligogenic (HCM) | 12 hops | 71 candidates | ✓ MYH7 + MYBPC3 both flagged |
| Phenotype Mismatch (cardiac + BRCA1 decoy) | 9 hops | 38 candidates | ✓ KCNQ1 flagged, BRCA1 decoy resisted |

---

## How It Works

```
Patient Input (text or image)
        │
        ▼
Gemma 4 Vision (optional)
  └── Extracts phenotypes from clinical report photo
        │
        ▼
Gemma 4 Agent (local, via Ollama)
  ├── Tool: hop(node_id)         → traverse knowledge graph
  ├── Tool: flag_causal(var_id)  → declare diagnosis
  ├── Tool: backtrack()          → return to previous node
  └── Tool: summarise_trail()    → review visited path
        │
        ▼
NaradaGraph (in-memory, 55K nodes)
  ├── 19,389 HPO phenotype terms
  ├── 92,000 ClinVar pathogenic variants
  └── 3,268 gene nodes + pathway edges
        │
        ▼
Local Gradio UI
  └── Shows graph trail, step rewards, final diagnosis
```

---

## Gemma 4 Features Used

| Feature | How Used |
|---|---|
| **Native function calling** | Graph navigation actions are real tool calls — no JSON parsing, no regex |
| **Multimodal (vision)** | Clinical report image → phenotype extraction before episode starts |
| **E4B edge model** | Fine-tuned for on-device deployment, runs on 8GB RAM laptops |
| **Local deployment (Ollama)** | Zero cloud dependency, works offline |

---

## Fine-Tuning

Base model: `google/gemma-4-4b-it`
Method: GRPO (Group Relative Policy Optimisation) via Unsloth + TRL
Training data: 60 graph navigation episodes across 3 task tiers
Reward signal: Correct causal variant flag (+1.0) + path quality (Overseer score 0–0.3)

Published weights: [HuggingFace — TODO](https://huggingface.co)
Training notebook: `training/gemma4_grpo.ipynb`

---

## Quickstart

```bash
# 1. Pull Gemma 4 via Ollama
ollama pull gemma4:4b

# 2. Install Python deps
pip install -r requirements.txt

# 3. Set up env
cp .env.example .env

# 4. Build graph data (one-time, ~3 min)
python scripts/filter_clinvar.py

# 5. Launch the app
python src/ui/app.py
# → Open http://localhost:7860
```

---

## Results

**Baseline (zero-shot Gemma 4):**
| Task | Score |
|---|---|
| monogenic | ~0.45 |
| oligogenic | ~0.24 |
| phenotype_mismatch | ~0.06 |

**After GRPO fine-tuning:**
| Task | Score | Improvement |
|---|---|---|
| monogenic | TODO | +TODO |
| oligogenic | TODO | +TODO |
| phenotype_mismatch | TODO | +TODO |

---

## Why This Matters

The bottleneck in rare disease diagnosis isn't data — ClinVar has 2M+ catalogued variants. The bottleneck is **reasoning under uncertainty** at the point of care. Narada Edge brings that reasoning capability to any clinical setting, on any hardware, with no ongoing infrastructure cost.

---

## Links

- [Kaggle Writeup](https://kaggle.com/TODO)
- [Demo Video](https://youtube.com/watch?v=TODO)
- [HuggingFace Weights](https://huggingface.co/TODO)
- [Original Narada-Env (OpenEnv)](https://github.com/KrishVenky/Narada-Env)

---

## Team

Krishna Venkataraman — [GitHub](https://github.com/KrishVenky)
