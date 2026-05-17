"""
Pydantic v2 schemas for all data flowing between GenoPath components.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    type: str  # gene | variant | phenotype | disease | pathway
    name: str
    description: str
    connected_node_ids: List[str] = []
    metadata: Dict[str, Any] = {}


class Variant(BaseModel):
    id: str  # VAR:{allele_id}
    allele_id: str
    gene: str
    name: str
    variant_type: str
    clinical_significance: str
    pathogenicity_score: float
    disease_associations: List[str] = []


class GenoPathAction(BaseModel):
    action_type: str  # hop | flag_causal | backtrack | summarise_trail
    node_id: Optional[str] = None
    variant_id: Optional[str] = None
    reasoning: str = ""


class GenoPathObservation(BaseModel):
    step: int
    max_steps: int
    task_type: str
    current_node: GraphNode
    trail: List[GraphNode] = []
    patient_phenotypes: List[str]  # HPO IDs
    phenotype_names: List[str]  # human-readable parallel list
    candidate_variants: List[Variant]
    gene_absent_phenotypes: Dict[str, List[str]] = {}  # gene → absent phenotype names
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


class StepResult(BaseModel):
    observation: GenoPathObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class GenoPathState(BaseModel):
    case_id: str
    task_type: str
    step: int
    current_node_id: str
    trail_ids: List[str]
    cumulative_reward: float
    done: bool


class EpisodeResult(BaseModel):
    task_type: str
    steps: List[Dict[str, Any]]
    final_reward: float
    success: bool
