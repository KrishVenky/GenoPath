"""
Filter raw ClinVar variant_summary.txt → data/clinvar_pathogenic.tsv

Only needed if clinvar_pathogenic.tsv is missing. The filtered file is committed to the repo.
Raw file download: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz

Filter criteria:
  - GRCh38 assembly only
  - ClinicalSignificance contains "Pathogenic" or "Likely pathogenic"
  - ReviewStatus contains "criteria provided" or "reviewed by expert panel" or "practice guideline"
  - Deduplicated by AlleleID
  - Gene symbol must be present and not "-"
  - PhenotypeList must be present and not just "not provided"
"""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "variant_summary.txt"
OUTPUT = ROOT / "data" / "clinvar_pathogenic.tsv"

KEEP_SIG_SUBSTRINGS = {"pathogenic", "likely pathogenic"}
KEEP_REVIEW_SUBSTRINGS = {"criteria provided", "reviewed by expert panel", "practice guideline"}


def _is_pathogenic(sig: str) -> bool:
    s = sig.lower()
    return any(sub in s for sub in KEEP_SIG_SUBSTRINGS)


def _is_reviewed(review: str) -> bool:
    r = review.lower()
    return any(sub in r for sub in KEEP_REVIEW_SUBSTRINGS)


def main() -> None:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found.")
        print("Download from: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz")
        print("Then: gunzip variant_summary.txt.gz && mv variant_summary.txt data/")
        return

    seen: set[str] = set()
    rows_written = 0

    with (
        open(INPUT, encoding="utf-8", errors="replace") as fin,
        open(OUTPUT, "w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin, delimiter="\t")

        writer = csv.writer(fout, delimiter="\t")
        writer.writerow([
            "#AlleleID", "GeneSymbol", "ClinicalSignificance",
            "PhenotypeList", "PhenotypeIDS", "Type", "Name",
            "Chromosome", "Start",
        ])

        for row in reader:
            # Handle BOM-prefixed first column key
            allele_id_key = next((k for k in row if "AlleleID" in k), None)
            if allele_id_key is None:
                continue

            if row.get("Assembly", "") != "GRCh38":
                continue

            sig = row.get("ClinicalSignificance", "").strip()
            if not _is_pathogenic(sig):
                continue

            review = row.get("ReviewStatus", "").strip()
            if not _is_reviewed(review):
                continue

            allele_id = row[allele_id_key].strip()
            if not allele_id or allele_id in seen:
                continue

            gene = row.get("GeneSymbol", "").strip()
            if not gene or gene == "-" or ";" in gene:
                continue

            pheno_list = row.get("PhenotypeList", "").strip()
            if not pheno_list or pheno_list.lower() == "not provided":
                continue

            seen.add(allele_id)
            writer.writerow([
                allele_id,
                gene,
                sig,
                pheno_list,
                row.get("PhenotypeIDS", "").strip(),
                row.get("Type", "").strip(),
                row.get("Name", "").strip(),
                row.get("Chromosome", "").strip(),
                row.get("Start", "").strip(),
            ])
            rows_written += 1

    print(f"Written {rows_written:,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
