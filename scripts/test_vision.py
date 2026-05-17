"""
Quick vision extraction test.
Usage:
  python scripts/test_vision.py path/to/image.png
If no path given, tests with a dummy prompt to confirm vision is working.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ollama

MODEL = "gemma4:e4b"
PROMPT = (
    "You are a clinical phenotype extractor. "
    "Look at this clinical report image and extract ALL symptoms, signs, and phenotypes mentioned. "
    "Return them as a plain list, one per line. "
    "Use standard medical terms (e.g. 'Seizures', 'Hypotonia', 'Global developmental delay'). "
    "Do not include diagnoses or gene names — only phenotypes the patient presents with."
)

def test_vision(image_path: str):
    print(f"Sending {image_path} to {MODEL} vision...")
    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_path],
        }],
    )
    raw = response.message.content or ""
    print("\n--- RAW OUTPUT ---")
    print(raw)

    terms = [
        line.strip().lstrip("-•*123456789.").strip()
        for line in raw.splitlines()
        if line.strip() and len(line.strip()) > 2
    ]
    print("\n--- EXTRACTED TERMS ---")
    for t in terms:
        print(f"  {t}")

    print("\n--- HPO MATCHING ---")
    from src.agent.intake import match_to_hpo
    for t in terms:
        m = match_to_hpo(t)
        status = m['hpo_id'] if m['hpo_id'] else "NO MATCH"
        print(f"  [{status}] {t} -> {m['name']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_vision.py <image_path>")
        print("\nSteps to test:")
        print("  1. Open data/test_report.txt in any text editor")
        print("  2. Screenshot it (Win+Shift+S or Snipping Tool)")
        print("  3. Save as data/test_report.png")
        print("  4. Run: python scripts/test_vision.py data/test_report.png")
        sys.exit(0)

    test_vision(sys.argv[1])
