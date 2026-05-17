"""
GenoPathGraph — in-memory gene-disease knowledge graph built from ClinVar + HPO.

Build order:
  1. Pathway nodes (hardcoded PATHWAY_MAP)
  2. Gene nodes + gene→pathway edges
  3. Variant nodes + variant→gene edges, collect disease names
  4. Disease nodes + disease→gene edges
  5. HPO phenotype nodes (catalog IDs + 5-level ancestors)
  6. HPO hierarchy edges (parent→child within added nodes)
  7. Catalog phenotype→disease edges (explicit, high-quality)
  8. Fuzzy phenotype→disease edges (inverted word-overlap index)
  9. Direct variant→phenotype edges (from PhenotypeIDS HP: values)
"""
from __future__ import annotations

import csv
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Pathogenicity score table
# ---------------------------------------------------------------------------

PATHOGENICITY_SCORES: Dict[str, float] = {
    "pathogenic": 0.95,
    "pathogenic/likely pathogenic": 0.85,
    "likely pathogenic": 0.75,
}


# ---------------------------------------------------------------------------
# Pathway map (hardcoded)
# ---------------------------------------------------------------------------

PATHWAY_MAP: Dict[str, str] = {
    # Cardiac
    "MYH7": "cardiac", "MYBPC3": "cardiac", "MYH6": "cardiac",
    "TNNT2": "cardiac", "TNNI3": "cardiac", "TPM1": "cardiac",
    "TTN": "cardiac", "LMNA": "cardiac", "SCN5A": "cardiac",
    "KCNQ1": "cardiac", "KCNH2": "cardiac", "PLN": "cardiac",
    "RYR2": "cardiac", "DSP": "cardiac", "PKP2": "cardiac",
    # Neurological
    "SCN1A": "neurological", "MECP2": "neurological",
    "TSC1": "neurological", "TSC2": "neurological",
    "HTT": "neurological", "SNCA": "neurological", "LRRK2": "neurological",
    "PARK2": "neurological", "GBA1": "neurological",
    # Metabolic
    "PAH": "metabolic", "HEXA": "metabolic", "HEXB": "metabolic",
    "ATP7B": "metabolic", "LDLR": "metabolic", "APOB": "metabolic", "PCSK9": "metabolic",
    "GALC": "metabolic", "ARSA": "metabolic",
    # Cancer — decoys only
    "BRCA1": "cancer", "BRCA2": "cancer", "TP53": "cancer",
    "MLH1": "cancer", "MSH2": "cancer", "MSH6": "cancer",
    "APC": "cancer", "RB1": "cancer", "VHL": "cancer",
    # Connective tissue
    "FBN1": "connective_tissue", "COL1A1": "connective_tissue", "COL3A1": "connective_tissue",
    # Pulmonary
    "CFTR": "pulmonary",
    # Renal
    "PKD1": "renal", "PKD2": "renal",
    # Musculoskeletal
    "DMD": "musculoskeletal", "DYSF": "musculoskeletal",
    # Ophthalmology
    "ABCA4": "ophthalmology", "USH2A": "ophthalmology", "MYO7A": "ophthalmology",
    # Haematology
    "HBB": "haematology", "HBA1": "haematology", "F8": "haematology", "VWF": "haematology",
}


# ---------------------------------------------------------------------------
# Disease catalog (curated — embedded here, used by graph + case generator)
# ---------------------------------------------------------------------------

DISEASE_CATALOG: List[Dict[str, Any]] = [
    # CARDIAC
    {"disease": "Hypertrophic cardiomyopathy", "genes": ["MYH7", "MYBPC3"],
     "hpo_ids": ["HP:0001639", "HP:0001640", "HP:0004308", "HP:0001685", "HP:0004749"],
     "pathway": "cardiac", "task_types": ["monogenic", "oligogenic"]},

    {"disease": "Long QT syndrome", "genes": ["KCNQ1", "KCNH2", "SCN5A"],
     "hpo_ids": ["HP:0001657", "HP:0004749", "HP:0004308", "HP:0001663", "HP:0001297"],
     "pathway": "cardiac", "task_types": ["monogenic", "phenotype_mismatch"]},

    {"disease": "Dilated cardiomyopathy", "genes": ["TTN", "LMNA"],
     "hpo_ids": ["HP:0001644", "HP:0001640", "HP:0001638", "HP:0004308", "HP:0001671"],
     "pathway": "cardiac", "task_types": ["monogenic", "oligogenic"]},

    # NEUROLOGICAL
    {"disease": "Dravet syndrome", "genes": ["SCN1A"],
     "hpo_ids": ["HP:0001250", "HP:0001263", "HP:0000729", "HP:0002194", "HP:0001252"],
     "pathway": "neurological", "task_types": ["monogenic", "phenotype_mismatch"]},

    {"disease": "Rett syndrome", "genes": ["MECP2"],
     "hpo_ids": ["HP:0001250", "HP:0002376", "HP:0001263", "HP:0000729", "HP:0002878"],
     "pathway": "neurological", "task_types": ["monogenic"]},

    {"disease": "Tuberous sclerosis complex", "genes": ["TSC1", "TSC2"],
     "hpo_ids": ["HP:0001250", "HP:0009716", "HP:0001263", "HP:0010804", "HP:0001646"],
     "pathway": "neurological", "task_types": ["oligogenic"]},

    # METABOLIC
    {"disease": "Phenylketonuria", "genes": ["PAH"],
     "hpo_ids": ["HP:0001249", "HP:0001263", "HP:0001250", "HP:0000729", "HP:0001256"],
     "pathway": "metabolic", "task_types": ["monogenic"]},

    {"disease": "Gaucher disease type 1", "genes": ["GBA1"],
     "hpo_ids": ["HP:0001744", "HP:0001903", "HP:0001873", "HP:0002240", "HP:0010885"],
     "pathway": "metabolic", "task_types": ["monogenic"]},

    {"disease": "Tay-Sachs disease", "genes": ["HEXA"],
     "hpo_ids": ["HP:0001250", "HP:0001249", "HP:0001263", "HP:0000365", "HP:0000486"],
     "pathway": "metabolic", "task_types": ["monogenic", "phenotype_mismatch"]},

    {"disease": "Wilson disease", "genes": ["ATP7B"],
     "hpo_ids": ["HP:0001638", "HP:0002480", "HP:0001410", "HP:0001871", "HP:0003128"],
     "pathway": "metabolic", "task_types": ["monogenic"]},

    # PULMONARY
    {"disease": "Cystic fibrosis", "genes": ["CFTR"],
     "hpo_ids": ["HP:0002099", "HP:0002110", "HP:0001738", "HP:0003763", "HP:0001891"],
     "pathway": "pulmonary", "task_types": ["monogenic"]},

    # CONNECTIVE TISSUE
    {"disease": "Marfan syndrome", "genes": ["FBN1"],
     "hpo_ids": ["HP:0003179", "HP:0000518", "HP:0001166", "HP:0002616", "HP:0001083"],
     "pathway": "connective_tissue", "task_types": ["monogenic", "phenotype_mismatch"]},

    # MUSCULOSKELETAL
    {"disease": "Duchenne muscular dystrophy", "genes": ["DMD"],
     "hpo_ids": ["HP:0003560", "HP:0001639", "HP:0001252", "HP:0001263", "HP:0003236"],
     "pathway": "musculoskeletal", "task_types": ["monogenic"]},

    # OPHTHALMOLOGY
    {"disease": "Stargardt disease", "genes": ["ABCA4"],
     "hpo_ids": ["HP:0007663", "HP:0000505", "HP:0007737", "HP:0000529", "HP:0001131"],
     "pathway": "ophthalmology", "task_types": ["monogenic"]},

    # LIPID
    {"disease": "Familial hypercholesterolaemia", "genes": ["LDLR", "APOB", "PCSK9"],
     "hpo_ids": ["HP:0003124", "HP:0000956", "HP:0001297", "HP:0001677", "HP:0000822"],
     "pathway": "metabolic", "task_types": ["monogenic", "oligogenic"]},

    # CANCER — DECOYS ONLY (task_types=[])
    {"disease": "Hereditary breast and ovarian cancer", "genes": ["BRCA1", "BRCA2"],
     "hpo_ids": ["HP:0003002", "HP:0100615", "HP:0002894", "HP:0006740"],
     "pathway": "cancer", "task_types": []},

    {"disease": "Li-Fraumeni syndrome", "genes": ["TP53"],
     "hpo_ids": ["HP:0002671", "HP:0012125", "HP:0001909", "HP:0003003"],
     "pathway": "cancer", "task_types": []},
]

# Gene → list of catalog entries that include that gene
GENE_TO_DISEASES: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
for _entry in DISEASE_CATALOG:
    for _gene in _entry["genes"]:
        GENE_TO_DISEASES[_gene].append(_entry)


# ---------------------------------------------------------------------------
# HPO OBO parser
# ---------------------------------------------------------------------------

def parse_hpo_obo(path: Path) -> Dict[str, Dict]:
    """Parse hp.obo → {HP:XXXXXXX: {id, name, definition, parents, synonyms}}."""
    terms: Dict[str, Dict] = {}
    current: Optional[Dict] = None
    in_term = False

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if line == "[Term]":
                if current and not current.get("is_obsolete") and current.get("id"):
                    terms[current["id"]] = current
                current = {"id": "", "name": "", "definition": "", "parents": [], "synonyms": []}
                in_term = True
                continue

            if line.startswith("[") and line != "[Term]":
                if current and not current.get("is_obsolete") and current.get("id"):
                    terms[current["id"]] = current
                current = None
                in_term = False
                continue

            if not in_term or current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("def: "):
                m = re.match(r'def: "(.+?)" \[', line)
                if m:
                    current["definition"] = m.group(1)
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True
            elif line.startswith("is_a: "):
                m = re.match(r"is_a: (HP:\d+)", line)
                if m:
                    current["parents"].append(m.group(1))
            elif line.startswith("synonym: "):
                m = re.match(r'synonym: "(.+?)"', line)
                if m:
                    current["synonyms"].append(m.group(1))

    if current and not current.get("is_obsolete") and current.get("id"):
        terms[current["id"]] = current

    return terms


# ---------------------------------------------------------------------------
# ClinVar TSV loader
# ---------------------------------------------------------------------------

def load_clinvar_variants(
    path: Path, max_per_gene: int = 50
) -> Dict[str, List[Dict]]:
    """
    Load clinvar_pathogenic.tsv → {gene: [variant_dict, ...]}.

    Columns used: #AlleleID, GeneSymbol, ClinicalSignificance, PhenotypeList,
                  PhenotypeIDS, Type, Name.
    Caps at max_per_gene (highest pathogenicity first after dedup by AlleleID).
    """
    by_gene: Dict[str, Dict[str, Dict]] = defaultdict(dict)  # gene → allele_id → dict

    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            # Handle #AlleleID key (DictReader preserves the literal header string)
            allele_id = (row.get("#AlleleID") or row.get("AlleleID") or "").strip()
            if not allele_id:
                continue

            gene = row.get("GeneSymbol", "").strip()
            if not gene or gene == "-":
                continue

            pheno_list = row.get("PhenotypeList", "").strip()
            if not pheno_list or pheno_list.lower() == "not provided":
                continue

            if allele_id in by_gene[gene]:
                continue

            sig = row.get("ClinicalSignificance", "").strip()
            sig_lower = sig.lower()
            # Normalize compound values like "Pathogenic/Likely pathogenic"
            if "/" in sig_lower:
                parts = {p.strip() for p in sig_lower.split("/")}
                if "pathogenic" in parts and "likely pathogenic" in parts:
                    sig_lower = "pathogenic/likely pathogenic"
            score = PATHOGENICITY_SCORES.get(sig_lower, 0.70)

            # Extract HP: IDs from PhenotypeIDS for direct graph edges (Step 9)
            phenotype_hpo_ids = [
                p.strip()
                for p in re.split(r"[|;,]", row.get("PhenotypeIDS", ""))
                if p.strip().startswith("HP:")
            ]

            disease_associations = [
                d.strip()
                for d in pheno_list.split("|")
                if d.strip() and d.strip().lower() not in {"not provided", ""}
            ][:3]

            by_gene[gene][allele_id] = {
                "allele_id": allele_id,
                "gene": gene,
                "name": row.get("Name", "").strip(),
                "variant_type": row.get("Type", "").strip(),
                "clinical_significance": sig,
                "pathogenicity_score": score,
                "disease_associations": disease_associations,
                "phenotype_hpo_ids": phenotype_hpo_ids,
            }

    result: Dict[str, List[Dict]] = {}
    for gene, var_dict in by_gene.items():
        variants = list(var_dict.values())
        variants.sort(key=lambda v: v["pathogenicity_score"], reverse=True)
        result[gene] = variants[:max_per_gene]

    return result


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    return re.sub(r"\W+", "_", name.lower()).strip("_")


def _words_4plus(text: str) -> Set[str]:
    return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 4}


# ---------------------------------------------------------------------------
# GenoPathGraph
# ---------------------------------------------------------------------------

class GenoPathGraph:
    """
    In-memory knowledge graph. Singleton — use get_graph(), do not instantiate directly.
    Node IDs: HP:XXXXXXX | DIS:{slug} | GENE:{symbol} | VAR:{allele_id} | PATH:{name}
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.hpo_terms: Dict[str, Dict] = {}
        self.gene_variants: Dict[str, List[Dict]] = {}
        self._build_graph()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "name": name,
                "description": description,
                "metadata": metadata or {},
            }

    def _add_edge(self, a: str, b: str) -> None:
        if a in self.nodes and b in self.nodes and a != b:
            self.edges[a].add(b)
            self.edges[b].add(a)

    def _get_ancestors(self, hpo_id: str, max_levels: int) -> Set[str]:
        ancestors: Set[str] = set()
        frontier = [(hpo_id, 0)]
        while frontier:
            current, level = frontier.pop()
            if level >= max_levels:
                continue
            for parent in self.hpo_terms.get(current, {}).get("parents", []):
                if parent not in ancestors:
                    ancestors.add(parent)
                    frontier.append((parent, level + 1))
        return ancestors

    def _build_graph(self) -> None:
        t0 = time.time()
        data_dir = ROOT / "data"

        self.hpo_terms = parse_hpo_obo(data_dir / "hp.obo")
        self.gene_variants = load_clinvar_variants(data_dir / "clinvar_pathogenic.tsv")

        # 1. Pathway nodes
        for pathway in set(PATHWAY_MAP.values()):
            path_id = f"PATH:{pathway}"
            self._add_node(path_id, "pathway", pathway.replace("_", " ").title(),
                           f"{pathway} biological pathway")

        # 2. Gene nodes + gene→pathway edges
        all_genes = set(self.gene_variants.keys()) | set(PATHWAY_MAP.keys())
        for entry in DISEASE_CATALOG:
            all_genes.update(entry["genes"])

        for gene in all_genes:
            gene_id = f"GENE:{gene}"
            pathway = PATHWAY_MAP.get(gene, "")
            self._add_node(gene_id, "gene", gene,
                           f"Gene {gene}" + (f" ({pathway} pathway)" if pathway else ""),
                           {"pathway": pathway})
            if pathway:
                self._add_edge(gene_id, f"PATH:{pathway}")

        # 3. Variant nodes + variant→gene edges; collect disease names per gene
        disease_to_genes: Dict[str, Set[str]] = defaultdict(set)
        for gene, variants in self.gene_variants.items():
            gene_id = f"GENE:{gene}"
            for v in variants:
                var_id = f"VAR:{v['allele_id']}"
                self._add_node(
                    var_id, "variant",
                    v["name"] or var_id,
                    f"{v['variant_type']} in {gene}",
                    {
                        "allele_id": v["allele_id"],
                        "gene": gene,
                        "variant_type": v["variant_type"],
                        "clinical_significance": v["clinical_significance"],
                        "pathogenicity_score": v["pathogenicity_score"],
                        "disease_associations": v["disease_associations"],
                    },
                )
                self._add_edge(var_id, gene_id)
                for dis_name in v["disease_associations"]:
                    disease_to_genes[dis_name].add(gene)

        # 4. Disease nodes (from ClinVar PhenotypeList) + disease→gene edges
        for dis_name, genes in disease_to_genes.items():
            dis_id = f"DIS:{_slugify(dis_name)}"
            self._add_node(dis_id, "disease", dis_name, f"Disease: {dis_name}")
            for gene in genes:
                self._add_edge(dis_id, f"GENE:{gene}")

        # 5. HPO phenotype nodes — catalog IDs + 5-level ancestors
        catalog_hpo_ids: Set[str] = set()
        for entry in DISEASE_CATALOG:
            catalog_hpo_ids.update(entry["hpo_ids"])

        hpo_to_include: Set[str] = set(catalog_hpo_ids)
        for hpo_id in catalog_hpo_ids:
            hpo_to_include.update(self._get_ancestors(hpo_id, 5))

        for hpo_id in hpo_to_include:
            if hpo_id not in self.hpo_terms:
                continue
            term = self.hpo_terms[hpo_id]
            self._add_node(
                hpo_id, "phenotype",
                term["name"],
                term.get("definition", ""),
                {"synonyms": term.get("synonyms", [])},
            )

        # 6. HPO hierarchy edges (parent-child within added nodes)
        for hpo_id in hpo_to_include:
            if hpo_id not in self.nodes:
                continue
            for parent_id in self.hpo_terms.get(hpo_id, {}).get("parents", []):
                if parent_id in self.nodes:
                    self._add_edge(hpo_id, parent_id)

        # 7. Catalog phenotype→disease edges (explicit, high-quality)
        for entry in DISEASE_CATALOG:
            dis_id = f"DIS:{_slugify(entry['disease'])}"
            if dis_id not in self.nodes:
                self._add_node(dis_id, "disease", entry["disease"],
                               f"Disease: {entry['disease']}")
            for hpo_id in entry["hpo_ids"]:
                if hpo_id in self.nodes:
                    self._add_edge(hpo_id, dis_id)
            for gene in entry["genes"]:
                self._add_edge(dis_id, f"GENE:{gene}")

        # 8. Fuzzy phenotype→disease edges via inverted word-overlap index
        disease_word_index: Dict[str, Set[str]] = defaultdict(set)
        for node_id, node in self.nodes.items():
            if node["type"] == "disease":
                for word in _words_4plus(node["name"]):
                    disease_word_index[word].add(node_id)

        for hpo_id in hpo_to_include:
            if hpo_id not in self.nodes:
                continue
            for word in _words_4plus(self.nodes[hpo_id]["name"]):
                for dis_id in disease_word_index.get(word, set()):
                    self._add_edge(hpo_id, dis_id)

        # 9. Direct variant→phenotype edges (PhenotypeIDS HP: values)
        for gene, variants in self.gene_variants.items():
            for v in variants:
                var_id = f"VAR:{v['allele_id']}"
                if var_id not in self.nodes:
                    continue
                for hpo_id in v.get("phenotype_hpo_ids", []):
                    if hpo_id in self.nodes:
                        self._add_edge(var_id, hpo_id)

        edge_pairs = sum(len(neighbors) for neighbors in self.edges.values()) // 2
        print(
            f"GenoPathGraph: {len(self.nodes):,} nodes, {edge_pairs:,} edge pairs"
            f" — built in {time.time() - t0:.1f}s"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.nodes:
            return None
        node = dict(self.nodes[node_id])
        node["connected_node_ids"] = sorted(self.edges.get(node_id, set()))
        return node

    def get_neighbors(self, node_id: str) -> List[str]:
        return sorted(self.edges.get(node_id, set()))

    def get_variants_for_gene(self, gene: str) -> List[Dict]:
        return list(self.gene_variants.get(gene, []))

    def get_hpo_name(self, hpo_id: str) -> str:
        if hpo_id in self.hpo_terms:
            return self.hpo_terms[hpo_id]["name"]
        if hpo_id in self.nodes:
            return self.nodes[hpo_id]["name"]
        return hpo_id

    def relevant_nodes_for_case(
        self, causal_genes: List[str], patient_hpo_ids: List[str]
    ) -> Set[str]:
        """Return the set of node IDs that earn positive step reward when visited."""
        relevant: Set[str] = set(patient_hpo_ids)
        for gene in causal_genes:
            gene_id = f"GENE:{gene}"
            relevant.add(gene_id)
            for v in self.gene_variants.get(gene, []):
                relevant.add(f"VAR:{v['allele_id']}")
            for neighbor in self.edges.get(gene_id, set()):
                if neighbor.startswith("DIS:"):
                    relevant.add(neighbor)
        return relevant

    def get_absent_phenotypes(
        self, gene: str, patient_hpo_ids: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Return (hpo_id, name) pairs: HPO terms in gene's catalog diseases
        that the patient does NOT present with. Capped at 4.
        """
        gene_hpo_set: Set[str] = set()
        for entry in GENE_TO_DISEASES.get(gene, []):
            gene_hpo_set.update(entry["hpo_ids"])
        patient_set = set(patient_hpo_ids)
        absent = [
            (hid, self.get_hpo_name(hid))
            for hid in sorted(gene_hpo_set - patient_set)
        ]
        return absent[:4]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_graph_singleton: Optional[GenoPathGraph] = None


def get_graph() -> GenoPathGraph:
    """Return the shared GenoPathGraph instance. Builds on first call (~8s)."""
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = GenoPathGraph()
    return _graph_singleton
