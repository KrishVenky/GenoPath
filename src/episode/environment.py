"""
GenoPathEnvironment — one instance per episode, never shared across sessions.

Action dispatch:
  hop(node_id)           — move along a graph edge; +reward if relevant, -reward if not
  flag_causal(variant_id) — terminal for monogenic/mismatch; accumulates for oligogenic
  backtrack()            — return to previous node in trail
  summarise_trail()      — neutral; costs a step
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.episode.models import (
    EpisodeResult,
    GenoPathAction,
    GenoPathObservation,
    GenoPathState,
    GraphNode,
    StepResult,
    Variant,
)
from src.graph.case_generator import MAX_STEPS, PatientCase, generate_case
from src.graph.graph import GenoPathGraph, get_graph

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_RELEVANT_HOP: float = 0.15
R_IRRELEVANT_HOP: float = -0.05
R_HALLUCINATED: float = -0.10
R_PER_STEP: float = -0.01
R_BACKTRACK_GOOD: float = 0.05
R_BACKTRACK_BAD: float = -0.05

R_TERMINAL_CORRECT: float = 1.0
R_TERMINAL_PARTIAL: float = 0.5   # per-variant weight for oligogenic
R_TERMINAL_WRONG: float = -0.5
R_TIMING_BONUS: float = 0.2

OVERSEER_BASE: float = 0.3
OVERSEER_MIN: float = 0.0


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, value))


# ---------------------------------------------------------------------------
# GenoPathEnvironment
# ---------------------------------------------------------------------------

class GenoPathEnvironment:
    """
    Episode environment. Create a new instance per episode via reset().
    Graph is a shared singleton injected at construction time.
    """

    def __init__(self, graph: Optional[GenoPathGraph] = None) -> None:
        self._graph = graph or get_graph()
        self._case: Optional[PatientCase] = None
        self._step: int = 0
        self._max_steps: int = 15
        self._current_node_id: str = ""
        self._trail: List[str] = []
        self._trail_set: Set[str] = set()
        self._flagged_allele_ids: List[str] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._hallucinated_hops: int = 0
        self._last_hop_was_relevant: bool = False
        self._reasoning_history: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_type: str, seed: Optional[int] = None) -> StepResult:
        self._case = generate_case(self._graph, task_type, seed)
        self._step = 0
        self._max_steps = MAX_STEPS[task_type]
        self._current_node_id = self._case.starting_node_id
        self._trail = [self._current_node_id]
        self._trail_set = {self._current_node_id}
        self._flagged_allele_ids = []
        self._cumulative_reward = 0.0
        self._done = False
        self._hallucinated_hops = 0
        self._last_hop_was_relevant = False
        self._reasoning_history = ""

        obs = self._build_observation(0.0)
        return StepResult(observation=obs, reward=0.0, done=False)

    def step(self, action: GenoPathAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_observation(0.0), reward=0.0, done=True
            )

        self._step += 1

        if action.reasoning:
            self._reasoning_history += " " + action.reasoning

        if action.action_type == "hop":
            reward = self._do_hop(action.node_id or "")
        elif action.action_type == "flag_causal":
            reward = self._do_flag(action.variant_id or "", action.reasoning)
        elif action.action_type == "backtrack":
            reward = self._do_backtrack()
        else:
            # summarise_trail or unknown
            reward = _clamp(R_PER_STEP)

        # Max-step timeout: overwrite step reward with terminal
        if not self._done and self._step >= self._max_steps:
            terminal = self._compute_terminal_reward()
            overseer = self._overseer_score()
            reward = _clamp(terminal + overseer)
            self._done = True

        reward = _clamp(reward)
        self._cumulative_reward += reward
        return StepResult(
            observation=self._build_observation(reward),
            reward=reward,
            done=self._done,
        )

    def state(self) -> GenoPathState:
        assert self._case is not None, "Call reset() before state()"
        return GenoPathState(
            case_id=self._case.case_id,
            task_type=self._case.task_type,
            step=self._step,
            current_node_id=self._current_node_id,
            trail_ids=list(self._trail),
            cumulative_reward=self._cumulative_reward,
            done=self._done,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_hop(self, node_id: str) -> float:
        reward = R_PER_STEP

        if node_id not in self._graph.nodes:
            self._hallucinated_hops += 1
            return _clamp(reward + R_HALLUCINATED)

        if node_id not in self._graph.get_neighbors(self._current_node_id):
            # Node exists but not connected — hallucinated hop, agent stays put
            self._hallucinated_hops += 1
            return _clamp(reward + R_HALLUCINATED)

        is_relevant = node_id in self._case.relevant_node_ids
        self._last_hop_was_relevant = is_relevant
        reward += R_RELEVANT_HOP if is_relevant else R_IRRELEVANT_HOP

        self._current_node_id = node_id
        self._trail.append(node_id)
        self._trail_set.add(node_id)
        return _clamp(reward)

    def _do_flag(self, variant_id: str, reasoning: str = "") -> float:
        allele_id = variant_id.removeprefix("VAR:") if variant_id.startswith("VAR:") else variant_id
        node_id = f"VAR:{allele_id}"

        # Guard: model must have visited the variant node before flagging it.
        # Premature flags (before navigating to the node) earn a penalty and the
        # episode continues, giving the model a chance to actually navigate there.
        if node_id not in self._trail_set:
            self._hallucinated_hops += 1
            return _clamp(R_HALLUCINATED)

        if allele_id and allele_id not in self._flagged_allele_ids:
            self._flagged_allele_ids.append(allele_id)

        # Oligogenic: accumulate flags — episode continues until max_steps
        if self._case.task_type == "oligogenic":
            return _clamp(R_PER_STEP)

        # All other task types: terminal immediately
        terminal = self._compute_terminal_reward()
        overseer = self._overseer_score()
        self._done = True
        return _clamp(terminal + overseer)

    def _do_backtrack(self) -> float:
        reward = R_PER_STEP
        if len(self._trail) <= 1:
            return _clamp(reward + R_BACKTRACK_BAD)
        # Penalise backtracking away from a relevant node
        reward += R_BACKTRACK_GOOD if not self._last_hop_was_relevant else R_BACKTRACK_BAD
        self._trail.pop()
        self._current_node_id = self._trail[-1]
        return _clamp(reward)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_terminal_reward(self) -> float:
        assert self._case is not None
        flagged = set(self._flagged_allele_ids)
        ground_truth = set(self._case.causal_allele_ids)

        if not flagged:
            # Timeout with no flag — partial exploration credit only
            return min(0.3, len(self._trail_set) / self._max_steps * 0.5)

        # phenotype_mismatch: check decoy first — any decoy flag is maximum penalty
        if self._case.decoy_gene:
            decoy_ids = {
                v["allele_id"]
                for v in self._graph.get_variants_for_gene(self._case.decoy_gene)
            }
            if flagged & decoy_ids:
                return R_TERMINAL_WRONG

        correct = flagged & ground_truth
        wrong = flagged - ground_truth

        if self._case.task_type == "monogenic":
            if correct:
                base = R_TERMINAL_CORRECT - 0.3 * len(wrong)
                timing = R_TIMING_BONUS if self._step < 10 else 0.0
                return base + timing
            return R_TERMINAL_WRONG

        if self._case.task_type == "oligogenic":
            n = len(ground_truth)
            partial = (len(correct) / n) * R_TERMINAL_PARTIAL if n > 0 else 0.0
            timing = R_TIMING_BONUS if (self._step < 15 and len(correct) == n) else 0.0
            return partial - 0.2 * len(wrong) + timing

        if self._case.task_type == "phenotype_mismatch":
            if correct:
                timing = R_TIMING_BONUS if self._step < 12 else 0.0
                return R_TERMINAL_CORRECT + timing
            return R_TERMINAL_WRONG

        return 0.0

    def _overseer_score(self) -> float:
        """
        Local rule-based quality bonus added to terminal reward.
        Does NOT call an LLM. Clamped to [OVERSEER_MIN, OVERSEER_BASE].
        """
        assert self._case is not None
        score = OVERSEER_BASE

        # Penalise hallucinated hops (max deduction 0.15)
        score -= min(0.15, self._hallucinated_hops * 0.05)

        # Penalise shallow exploration
        if len(self._trail_set) < 3:
            score -= 0.10

        # Reward visiting the causal gene node
        for gene in self._case.causal_genes:
            if f"GENE:{gene}" in self._trail_set:
                score += 0.05
                break

        # Reward explicit use of absent-phenotype reasoning (cap +0.10)
        absent_bonus = 0.0
        if self._reasoning_history:
            reasoning_lower = self._reasoning_history.lower()
            for pairs in self._case.gene_absent_phenotypes.values():
                for _, name in pairs:
                    if name.lower() in reasoning_lower:
                        absent_bonus += 0.05
                        if absent_bonus >= 0.10:
                            break
                if absent_bonus >= 0.10:
                    break
        score += absent_bonus

        return max(OVERSEER_MIN, min(OVERSEER_BASE, score))

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, step_reward: float) -> GenoPathObservation:
        assert self._case is not None

        raw = self._graph.get_node(self._current_node_id) or {
            "id": self._current_node_id, "type": "unknown",
            "name": self._current_node_id, "description": "",
            "connected_node_ids": [], "metadata": {},
        }
        current_node = GraphNode(
            id=raw["id"], type=raw["type"], name=raw["name"],
            description=raw["description"],
            connected_node_ids=raw["connected_node_ids"],
            metadata=raw["metadata"],
        )

        trail_nodes: List[GraphNode] = []
        for nid in self._trail:
            r = self._graph.get_node(nid)
            if r:
                trail_nodes.append(GraphNode(
                    id=r["id"], type=r["type"], name=r["name"],
                    description=r["description"],
                    connected_node_ids=r["connected_node_ids"],
                    metadata=r["metadata"],
                ))

        gene_absent_names: Dict[str, List[str]] = {
            gene: [name for _, name in pairs]
            for gene, pairs in self._case.gene_absent_phenotypes.items()
        }

        return GenoPathObservation(
            step=self._step,
            max_steps=self._max_steps,
            task_type=self._case.task_type,
            current_node=current_node,
            trail=trail_nodes,
            patient_phenotypes=self._case.patient_hpo_ids,
            phenotype_names=self._case.patient_phenotype_names,
            candidate_variants=self._case.candidate_variants,
            gene_absent_phenotypes=gene_absent_names,
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
        )
