"""
Creates genopath_upload.zip for Colab training.
Run from the project root: python scripts/make_colab_zip.py

Includes: src/, data/, requirements.txt
Excludes: __pycache__, results/, .env, .git, models
Output: genopath_upload.zip (~44MB)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "genopath_upload.zip"

INCLUDE = [
    "src/",
    "data/hp.obo",
    "data/clinvar_pathogenic.tsv",
    "requirements.txt",
    "training/gemma4_grpo.ipynb",
]
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_DIRS     = {"__pycache__", ".git", "results", "scripts"}

def should_skip(path: Path) -> bool:
    if path.suffix in SKIP_SUFFIXES:
        return True
    return any(part in SKIP_DIRS for part in path.parts)

files_added = []
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for pattern in INCLUDE:
        if pattern.endswith("/"):
            folder = ROOT / pattern.rstrip("/")
            for f in sorted(folder.rglob("*")):
                if f.is_file() and not should_skip(f.relative_to(ROOT)):
                    arcname = f.relative_to(ROOT)
                    zf.write(f, arcname)
                    files_added.append(str(arcname))
        else:
            f = ROOT / pattern
            if f.exists():
                zf.write(f, pattern)
                files_added.append(pattern)
            else:
                print(f"  WARNING: {pattern} not found, skipping")

size_mb = OUT.stat().st_size / 1024 / 1024
print(f"Created: {OUT.name}  ({size_mb:.1f} MB, {len(files_added)} files)")
print()
print("Files included:")
for f in files_added:
    print(f"  {f}")
print()
print("Next steps:")
print("  1. Upload genopath_upload.zip to Google Drive (root or any folder)")
print("  2. Open training/gemma4_grpo.ipynb in Colab")
print("  3. In Cell 3, update ZIP_PATH to match your Drive location")
print("  4. Set HF_TOKEN in Cell 2")
print("  5. Runtime -> T4 GPU -> Run all")
