"""
Phenotype extraction from clinical report images.

Desktop path (default): calls Gemma 4 E4B via Ollama with vision.
On-device path: uses MediaPipe LLM Inference API via LiteRT on any Android phone
running Gemma (AI Edge Gallery, or a custom APK with the MediaPipe SDK).

The on-device path is documented below for LiteRT prize eligibility.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# HPO matcher (exact > substring > synonym, then fall back to raw string)
# ---------------------------------------------------------------------------

_hpo_index: Optional[Dict[str, Dict]] = None


def _get_hpo_index() -> Dict[str, Dict]:
    global _hpo_index
    if _hpo_index is None:
        from src.graph.graph import get_graph
        graph = get_graph()
        _hpo_index = graph.hpo_terms
    return _hpo_index


def match_to_hpo(raw_term: str) -> Dict[str, str]:
    """
    Match a raw phenotype string to an HPO term.
    Returns {"raw": ..., "hpo_id": ..., "name": ...}.
    hpo_id is empty string if no match found.
    """
    hpo_terms = _get_hpo_index()
    raw_lower = raw_term.lower().strip()

    # 1. Exact match on name
    for hpo_id, term in hpo_terms.items():
        if term["name"].lower() == raw_lower:
            return {"raw": raw_term, "hpo_id": hpo_id, "name": term["name"]}

    # 2. Exact match on synonyms
    for hpo_id, term in hpo_terms.items():
        for syn in term.get("synonyms", []):
            if syn.lower() == raw_lower:
                return {"raw": raw_term, "hpo_id": hpo_id, "name": term["name"]}

    # 3. Substring match on name (raw term length > 6 to avoid spurious matches)
    if len(raw_lower) > 6:
        for hpo_id, term in hpo_terms.items():
            if raw_lower in term["name"].lower():
                return {"raw": raw_term, "hpo_id": hpo_id, "name": term["name"]}

    # 4. Substring match on synonyms
    if len(raw_lower) > 6:
        for hpo_id, term in hpo_terms.items():
            for syn in term.get("synonyms", []):
                if raw_lower in syn.lower():
                    return {"raw": raw_term, "hpo_id": hpo_id, "name": term["name"]}

    # No match — include as raw string so 12B downstream can still reason with it
    return {"raw": raw_term, "hpo_id": "", "name": raw_term}


# ---------------------------------------------------------------------------
# Desktop path: Ollama vision (Gemma 4 26B)
# ---------------------------------------------------------------------------

def extract_phenotypes_from_image(image_path: str) -> List[Dict[str, str]]:
    """
    Extract phenotype terms from a clinical report image.

    Uses Gemma 4 E4B vision (via Ollama) when called on the desktop.
    Returns list of {"raw": ..., "hpo_id": ..., "name": ...} dicts.
    """
    import ollama

    model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    prompt = (
        "You are a clinical phenotype extractor. "
        "Look at this clinical report image and extract ALL symptoms, signs, and phenotypes mentioned. "
        "Return them as a plain list, one per line. "
        "Use standard medical terms where possible (e.g. 'Seizures', 'Hypotonia', 'Global developmental delay'). "
        "Do not include diagnoses or gene names — only phenotypes the patient presents with."
    )

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_path],
        }],
    )

    raw_text = response.message.content or ""
    raw_terms = [
        line.strip().lstrip("-•*").strip()
        for line in raw_text.splitlines()
        if line.strip() and len(line.strip()) > 2
    ]

    return [match_to_hpo(term) for term in raw_terms if term]


# ---------------------------------------------------------------------------
# On-device path documentation (LiteRT / MediaPipe)
#
# This block documents the on-device inference architecture for LiteRT prize
# eligibility. Works on any Android phone that can run Gemma via AI Edge Gallery
# or a custom APK with the MediaPipe LLM Inference SDK.
#
# E2B (2B, 4-bit, ~1.5GB) fits all phones with >=4GB RAM.
# E4B (4B, 4-bit, ~2.5GB) fits phones with >=6GB RAM; close heavy apps first.
# ---------------------------------------------------------------------------

# from mediapipe.tasks.python.genai import inference as genai_inference
#
# def extract_phenotypes_on_device(image_path: str) -> List[Dict[str, str]]:
#     """
#     On-device phenotype extraction using Gemma 4 E2B via LiteRT / MediaPipe.
#     Runs entirely on the phone's NPU/GPU. No network required.
#
#     Model: Gemma 4 E2B (2B parameters, 4-bit quantised, ~1.5GB)
#     API:   MediaPipe LLM Inference API (ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)
#     """
#     model_path = "/data/local/tmp/gemma4_e2b.bin"   # deployed via adb or AI Edge Gallery
#     options = genai_inference.LlmInferenceOptions(
#         model_path=model_path,
#         max_tokens=512,
#         top_k=40,
#         top_p=0.95,
#         temperature=0.0,
#     )
#     llm = genai_inference.LlmInference.create_from_options(options)
#
#     # Build multimodal prompt (text + image encoded as base64)
#     import base64
#     with open(image_path, "rb") as f:
#         img_b64 = base64.b64encode(f.read()).decode()
#     prompt = (
#         "<image>" + img_b64 + "</image>\n"
#         "Extract all clinical phenotypes from this report. One per line. "
#         "Standard medical terms only."
#     )
#
#     raw_text = llm.generate_response(prompt)
#     raw_terms = [line.strip() for line in raw_text.splitlines() if line.strip()]
#
#     # HPO matching happens on the laptop after POST /intake
#     return [{"raw": t, "hpo_id": "", "name": t} for t in raw_terms]
