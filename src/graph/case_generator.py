"""
PatientCase dataclass and generators for the three task tiers:
  monogenic, oligogenic, phenotype_mismatch.

All generators accept a `seed` for reproducibility. Pass the same seed to get
the same case; pass None for a random case.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.episode.models import Variant
from src.graph.graph import DISEASE_CATALOG, GENE_TO_DISEASES, PATHWAY_MAP, GenoPathGraph

MAX_STEPS: Dict[str, int] = {
    "monogenic": 15,
    "oligogenic": 25,
    "phenotype_mismatch": 20,
}


# ---------------------------------------------------------------------------
# PatientCase
# ---------------------------------------------------------------------------

@dataclass
class PatientCase:
    case_id: str
    task_type: str
    disease_name: str
    causal_genes: List[str]
    causal_allele_ids: List[str]
    patient_hpo_ids: List[str]
    patient_phenotype_names: List[str]
    candidate_variants: List[Variant]
    relevant_node_ids: Set[str]
    starting_node_id: str
    decoy_gene: Optional[str]
    gene_absent_phenotypes: Dict[str, List[Tuple[str, str]]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_variant(raw: Dict) -> Variant:
    return Variant(
        id=f"VAR:{raw['allele_id']}",
        allele_id=raw["allele_id"],
        gene=raw["gene"],
        name=raw["name"],
        variant_type=raw["variant_type"],
        clinical_significance=raw["clinical_significance"],
        pathogenicity_score=raw["pathogenicity_score"],
        disease_associations=raw.get("disease_associations", []),
    )


def _sample_variants(
    graph: GenoPathGraph,
    gene: str,
    count: int,
    rng: random.Random,
    min_score: float = 0.0,
    exclude_allele_ids: Optional[Set[str]] = None,
) -> List[Variant]:
    """Sample up to `count` variants for `gene` with score >= min_score."""
    exclude = exclude_allele_ids or set()
    pool = [
        v for v in graph.get_variants_for_gene(gene)
        if v["pathogenicity_score"] >= min_score and v["allele_id"] not in exclude
    ]
    if not pool:
        return []
    chosen = rng.sample(pool, min(count, len(pool)))
    return [_make_variant(v) for v in chosen]


def _same_pathway_genes(
    causal_genes: List[str], exclude: Set[str], graph: GenoPathGraph
) -> List[str]:
    """Return genes in the same pathway(s) as causal_genes, excluding `exclude`."""
    pathways = {PATHWAY_MAP.get(g) for g in causal_genes if PATHWAY_MAP.get(g)}
    return [
        g for g, p in PATHWAY_MAP.items()
        if p in pathways and g not in exclude and graph.get_variants_for_gene(g)
    ]


def _compute_absent_phenotypes(
    graph: GenoPathGraph,
    candidate_variants: List[Variant],
    patient_hpo_ids: List[str],
) -> Dict[str, List[Tuple[str, str]]]:
    """Compute absent phenotypes for every unique gene in the candidate pool."""
    genes = {v.gene for v in candidate_variants}
    return {
        gene: graph.get_absent_phenotypes(gene, patient_hpo_ids)
        for gene in genes
    }


def _pad_distractors(
    graph: GenoPathGraph,
    existing: List[Variant],
    target: int,
    exclude_genes: Set[str],
    rng: random.Random,
) -> List[Variant]:
    """Fill up to `target` distractors by sampling from any remaining genes."""
    used_allele_ids = {v.allele_id for v in existing}
    result = list(existing)
    all_genes = [
        g for g in graph.gene_variants
        if g not in exclude_genes and g not in {v.gene for v in existing}
    ]
    rng.shuffle(all_genes)
    for g in all_genes:
        if len(result) >= target:
            break
        vs = _sample_variants(graph, g, 1, rng, exclude_allele_ids=used_allele_ids)
        if vs:
            result.extend(vs)
            used_allele_ids.add(vs[0].allele_id)
    return result


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_monogenic_case(
    graph: GenoPathGraph, seed: Optional[int] = None
) -> PatientCase:
    rng = random.Random(seed)

    eligible = [e for e in DISEASE_CATALOG if "monogenic" in e["task_types"]]
    entry = rng.choice(eligible)

    causal_gene = rng.choice(entry["genes"])

    n_hpo = rng.randint(3, min(4, len(entry["hpo_ids"])))
    patient_hpo_ids = rng.sample(entry["hpo_ids"], n_hpo)
    patient_phenotype_names = [graph.get_hpo_name(h) for h in patient_hpo_ids]

    causal_variants = _sample_variants(graph, causal_gene, rng.randint(2, 3), rng)
    if not causal_variants:
        raw = graph.get_variants_for_gene(causal_gene)[:2]
        causal_variants = [_make_variant(v) for v in raw]

    causal_allele_ids = [v.allele_id for v in causal_variants]
    used = {v.allele_id for v in causal_variants}

    same_path = _same_pathway_genes([causal_gene], {causal_gene}, graph)
    distractor_variants: List[Variant] = []
    n_want = rng.randint(3, 5)
    for dg in rng.sample(same_path, min(n_want, len(same_path))):
        vs = _sample_variants(graph, dg, 1, rng, exclude_allele_ids=used)
        distractor_variants.extend(vs)
        used.update(v.allele_id for v in vs)

    # Pad to at least 3 distractors using any gene
    if len(distractor_variants) < 3:
        distractor_variants = _pad_distractors(
            graph, distractor_variants, 3, {causal_gene}, rng
        )

    all_variants = causal_variants + distractor_variants
    rng.shuffle(all_variants)

    return PatientCase(
        case_id=uuid.uuid4().hex[:8],
        task_type="monogenic",
        disease_name=entry["disease"],
        causal_genes=[causal_gene],
        causal_allele_ids=causal_allele_ids,
        patient_hpo_ids=patient_hpo_ids,
        patient_phenotype_names=patient_phenotype_names,
        candidate_variants=all_variants,
        relevant_node_ids=graph.relevant_nodes_for_case([causal_gene], patient_hpo_ids),
        starting_node_id=patient_hpo_ids[0],
        decoy_gene=None,
        gene_absent_phenotypes=_compute_absent_phenotypes(graph, all_variants, patient_hpo_ids),
    )


def generate_oligogenic_case(
    graph: GenoPathGraph, seed: Optional[int] = None
) -> PatientCase:
    rng = random.Random(seed)

    eligible = [
        e for e in DISEASE_CATALOG
        if "oligogenic" in e["task_types"] and len(e["genes"]) >= 2
    ]
    entry = rng.choice(eligible)
    causal_genes = list(entry["genes"])

    n_hpo = rng.randint(5, min(7, len(entry["hpo_ids"])))
    patient_hpo_ids = rng.sample(entry["hpo_ids"], n_hpo)
    patient_phenotype_names = [graph.get_hpo_name(h) for h in patient_hpo_ids]

    causal_variants: List[Variant] = []
    used: Set[str] = set()
    for gene in causal_genes:
        vs = _sample_variants(graph, gene, rng.randint(1, 2), rng, exclude_allele_ids=used)
        if not vs:
            raw = graph.get_variants_for_gene(gene)[:1]
            vs = [_make_variant(v) for v in raw]
        causal_variants.extend(vs)
        used.update(v.allele_id for v in vs)

    causal_allele_ids = [v.allele_id for v in causal_variants]

    same_path = _same_pathway_genes(causal_genes, set(causal_genes), graph)
    distractor_variants: List[Variant] = []
    n_want = rng.randint(5, 8)
    for dg in rng.sample(same_path, min(n_want, len(same_path))):
        vs = _sample_variants(graph, dg, 1, rng, exclude_allele_ids=used)
        distractor_variants.extend(vs)
        used.update(v.allele_id for v in vs)

    if len(distractor_variants) < 5:
        distractor_variants = _pad_distractors(
            graph, distractor_variants, 5, set(causal_genes), rng
        )

    all_variants = causal_variants + distractor_variants
    rng.shuffle(all_variants)

    return PatientCase(
        case_id=uuid.uuid4().hex[:8],
        task_type="oligogenic",
        disease_name=entry["disease"],
        causal_genes=causal_genes,
        causal_allele_ids=causal_allele_ids,
        patient_hpo_ids=patient_hpo_ids,
        patient_phenotype_names=patient_phenotype_names,
        candidate_variants=all_variants,
        relevant_node_ids=graph.relevant_nodes_for_case(causal_genes, patient_hpo_ids),
        starting_node_id=patient_hpo_ids[0],
        decoy_gene=None,
        gene_absent_phenotypes=_compute_absent_phenotypes(graph, all_variants, patient_hpo_ids),
    )


def generate_mismatch_case(
    graph: GenoPathGraph, seed: Optional[int] = None
) -> PatientCase:
    rng = random.Random(seed)

    eligible = [e for e in DISEASE_CATALOG if "phenotype_mismatch" in e["task_types"]]
    entry = rng.choice(eligible)
    causal_gene = rng.choice(entry["genes"])

    # Decoy: pick cancer gene that has high-pathogenicity variants
    decoy_candidates = [
        g for g in ["BRCA1", "BRCA2", "TP53"]
        if any(v["pathogenicity_score"] >= 0.90 for v in graph.get_variants_for_gene(g))
    ]
    if not decoy_candidates:
        decoy_candidates = [
            g for g in ["BRCA1", "BRCA2", "TP53"] if graph.get_variants_for_gene(g)
        ]
    decoy_gene = rng.choice(decoy_candidates)

    # Patient phenotypes: all from primary disease, none from cancer
    patient_hpo_ids = list(entry["hpo_ids"])
    patient_phenotype_names = [graph.get_hpo_name(h) for h in patient_hpo_ids]

    causal_variants = _sample_variants(graph, causal_gene, rng.randint(2, 3), rng)
    if not causal_variants:
        raw = graph.get_variants_for_gene(causal_gene)[:2]
        causal_variants = [_make_variant(v) for v in raw]

    causal_allele_ids = [v.allele_id for v in causal_variants]
    used = {v.allele_id for v in causal_variants}

    # 2-3 high-pathogenicity decoy variants
    decoy_pool = [
        v for v in graph.get_variants_for_gene(decoy_gene)
        if v["pathogenicity_score"] >= 0.90
    ]
    n_decoy = rng.randint(2, min(3, len(decoy_pool))) if decoy_pool else 0
    if n_decoy:
        decoy_raw = rng.sample(decoy_pool, n_decoy)
        decoy_variants = [_make_variant(v) for v in decoy_raw]
        used.update(v.allele_id for v in decoy_variants)
    else:
        decoy_variants = []

    # Additional distractors from same pathway as causal gene
    same_path = _same_pathway_genes([causal_gene], {causal_gene, decoy_gene}, graph)
    distractor_variants: List[Variant] = []
    for dg in rng.sample(same_path, min(rng.randint(3, 5), len(same_path))):
        vs = _sample_variants(graph, dg, 1, rng, exclude_allele_ids=used)
        distractor_variants.extend(vs)
        used.update(v.allele_id for v in vs)

    all_variants = causal_variants + decoy_variants + distractor_variants
    rng.shuffle(all_variants)

    return PatientCase(
        case_id=uuid.uuid4().hex[:8],
        task_type="phenotype_mismatch",
        disease_name=entry["disease"],
        causal_genes=[causal_gene],
        causal_allele_ids=causal_allele_ids,
        patient_hpo_ids=patient_hpo_ids,
        patient_phenotype_names=patient_phenotype_names,
        candidate_variants=all_variants,
        relevant_node_ids=graph.relevant_nodes_for_case([causal_gene], patient_hpo_ids),
        starting_node_id=patient_hpo_ids[0],
        decoy_gene=decoy_gene,
        gene_absent_phenotypes=_compute_absent_phenotypes(graph, all_variants, patient_hpo_ids),
    )


def generate_case(
    graph: GenoPathGraph, task_type: str, seed: Optional[int] = None
) -> PatientCase:
    if task_type == "monogenic":
        return generate_monogenic_case(graph, seed)
    if task_type == "oligogenic":
        return generate_oligogenic_case(graph, seed)
    if task_type == "phenotype_mismatch":
        return generate_mismatch_case(graph, seed)
    raise ValueError(
        f"Unknown task_type: {task_type!r}. "
        "Expected: monogenic | oligogenic | phenotype_mismatch"
    )
