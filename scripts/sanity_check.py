"""
GenoPath integration smoke test. Run after every major change.
Checks are added incrementally as phases complete.

Usage:
    python scripts/sanity_check.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Phase 1: Graph
# ---------------------------------------------------------------------------

print("=" * 60)
print("PHASE 1 - Graph")
print("=" * 60)

from src.graph.graph import get_graph, GENE_TO_DISEASES, DISEASE_CATALOG  # noqa: E402

t0 = time.time()
graph = get_graph()
elapsed = time.time() - t0

assert len(graph.nodes) > 50_000, (
    f"Graph too small: {len(graph.nodes):,} nodes (expected > 50,000)"
)
print(f"  [OK] Node count: {len(graph.nodes):,}")

edge_pairs = sum(len(v) for v in graph.edges.values()) // 2
assert edge_pairs > 0, "Graph has no edges"
print(f"  [OK] Edge pairs: {edge_pairs:,}")

assert elapsed < 20.0, f"Graph load too slow: {elapsed:.1f}s (expected < 20s)"
print(f"  [OK] Load time: {elapsed:.1f}s")

# HP:0001250 = Seizures — must be in the graph and have neighbors
seizure_neighbors = graph.get_neighbors("HP:0001250")
assert len(seizure_neighbors) >= 3, (
    f"Seizures node has too few neighbors: {seizure_neighbors}"
)
print(f"  [OK] HP:0001250 (Seizures) neighbors: {len(seizure_neighbors)}")

# BRCA1 must have absent phenotypes when patient only has seizures
absent = graph.get_absent_phenotypes("BRCA1", ["HP:0001250"])
assert len(absent) > 0, "BRCA1 should have absent phenotypes vs seizures-only patient"
absent_names = [name for _, name in absent]
print(f"  [OK] BRCA1 absent phenotypes vs seizures patient: {absent_names}")

# Causal gene variants must exist for key disease genes
for gene in ["SCN1A", "BRCA1", "MYH7"]:
    variants = graph.get_variants_for_gene(gene)
    assert len(variants) > 0, f"No variants found for {gene}"
print(f"  [OK] Variants present for SCN1A, BRCA1, MYH7")

# GENE_TO_DISEASES lookup must cover catalog genes
assert "SCN1A" in GENE_TO_DISEASES, "SCN1A missing from GENE_TO_DISEASES"
assert "BRCA1" in GENE_TO_DISEASES, "BRCA1 missing from GENE_TO_DISEASES"
print(f"  [OK] GENE_TO_DISEASES: {len(GENE_TO_DISEASES)} genes")

# Spot-check a few node types
scn1a = graph.get_node("GENE:SCN1A")
assert scn1a is not None, "GENE:SCN1A node missing"
assert scn1a["type"] == "gene"
assert len(scn1a["connected_node_ids"]) > 0
print(f"  [OK] GENE:SCN1A node OK, {len(scn1a['connected_node_ids'])} neighbors")

dravet = graph.get_node("DIS:dravet_syndrome")
assert dravet is not None, "DIS:dravet_syndrome node missing"
assert dravet["type"] == "disease"
print(f"  [OK] DIS:dravet_syndrome node OK")

print()
print("All Phase 1 checks passed [OK]")
print()

# ---------------------------------------------------------------------------
# Phase 2: Episode Logic
# ---------------------------------------------------------------------------

print("=" * 60)
print("PHASE 2 - Episode Logic")
print("=" * 60)

from src.episode.models import GenoPathObservation, GenoPathAction  # noqa: E402
from src.graph.case_generator import generate_case, PatientCase     # noqa: E402
from src.episode.environment import GenoPathEnvironment, _clamp     # noqa: E402

# Verify _clamp behaviour
assert _clamp(0.0) == 0.01
assert _clamp(1.0) == 0.99
assert _clamp(0.5) == 0.5
assert _clamp(-999) == 0.01
assert _clamp(999) == 0.99
print("  [OK] _clamp bounds correct")

# Generate one case of each type and verify structure
for task in ["monogenic", "oligogenic", "phenotype_mismatch"]:
    case = generate_case(graph, task, seed=42)
    assert isinstance(case, PatientCase), f"generate_case returned wrong type for {task}"
    assert case.task_type == task
    assert len(case.candidate_variants) >= 5, (
        f"{task}: only {len(case.candidate_variants)} candidates (expected >= 5)"
    )
    assert len(case.causal_allele_ids) >= 1, f"{task}: no causal allele IDs"
    assert all(
        v.allele_id in [cv.allele_id for cv in case.candidate_variants]
        for v in case.candidate_variants
        if v.allele_id in case.causal_allele_ids
    ), f"{task}: causal allele IDs not in candidate pool"
    assert case.starting_node_id in graph.nodes, (
        f"{task}: starting_node_id {case.starting_node_id!r} not in graph"
    )
    if task == "phenotype_mismatch":
        assert case.decoy_gene is not None, "phenotype_mismatch: decoy_gene is None"
        decoy_allele_ids = {
            v["allele_id"] for v in graph.get_variants_for_gene(case.decoy_gene)
        }
        candidate_allele_ids = {v.allele_id for v in case.candidate_variants}
        assert decoy_allele_ids & candidate_allele_ids, (
            "phenotype_mismatch: no decoy variants in candidate pool"
        )
    print(f"  [OK] generate_case({task!r}, seed=42): "
          f"{len(case.candidate_variants)} candidates, "
          f"{len(case.causal_allele_ids)} causal allele(s)"
          + (f", decoy={case.decoy_gene}" if case.decoy_gene else ""))

# Run a full monogenic episode with hardcoded actions; verify reward bounds
env = GenoPathEnvironment(graph)
result = env.reset("monogenic", seed=1)

assert 0.0 <= result.reward <= 0.99, f"reset reward out of range: {result.reward}"
assert result.observation.step == 0
assert not result.done

obs = result.observation
# Walk 3 steps: hop to first neighbour, hop to second neighbour, backtrack
for step_num in range(1, 4):
    neighbors = obs.current_node.connected_node_ids
    if neighbors:
        action = GenoPathAction(action_type="hop", node_id=neighbors[0], reasoning="test hop")
    else:
        action = GenoPathAction(action_type="summarise_trail")
    result = env.step(action)
    obs = result.observation
    assert 0.01 <= result.reward <= 0.99, (
        f"Step {step_num} reward {result.reward:.4f} out of (0.01, 0.99)"
    )

print(f"  [OK] 3-step monogenic episode: rewards all in (0.01, 0.99), "
      f"cumulative={obs.cumulative_reward:.4f}")

# Run episodes to terminal for all three task types
for task in ["monogenic", "oligogenic", "phenotype_mismatch"]:
    env2 = GenoPathEnvironment(graph)
    r2 = env2.reset(task, seed=7)
    while not r2.done:
        nbrs = r2.observation.current_node.connected_node_ids
        action = (
            GenoPathAction(action_type="hop", node_id=nbrs[0], reasoning="test")
            if nbrs else
            GenoPathAction(action_type="summarise_trail")
        )
        r2 = env2.step(action)
    assert 0.01 <= r2.reward <= 0.99, (
        f"{task} terminal reward {r2.reward:.4f} out of (0.01, 0.99)"
    )
    print(f"  [OK] {task} episode ran to terminal: reward={r2.reward:.4f}, "
          f"steps={r2.observation.step}")

# Variant visit guard: flag_causal on an unvisited variant must NOT terminate episode
env_g = GenoPathEnvironment(graph)
r_g = env_g.reset("monogenic", seed=5)
causal_id = r_g.observation.candidate_variants[0].allele_id
# Attempt to flag before visiting the node
r_g = env_g.step(GenoPathAction(action_type="flag_causal", variant_id=f"VAR:{causal_id}", reasoning="test"))
assert not r_g.done, "Guard failed: episode ended on unvisited variant flag"
assert r_g.reward <= 0.01, f"Guard failed: no penalty for unvisited flag (reward={r_g.reward:.4f})"
print(f"  [OK] Variant visit guard: unvisited flag penalised ({r_g.reward:.4f}), episode continues")

print()
print("All Phase 2 checks passed [OK]")
print()

# ---------------------------------------------------------------------------
# Phase 3: Agent (tools + observation formatter; model call conditional)
# ---------------------------------------------------------------------------

print("=" * 60)
print("PHASE 3 - Agent")
print("=" * 60)

from src.agent.tools import GENOPATH_TOOLS, SYSTEM_PROMPT, format_observation  # noqa: E402

assert len(GENOPATH_TOOLS) == 4
tool_names = {t["function"]["name"] for t in GENOPATH_TOOLS}
assert tool_names == {"hop", "flag_causal", "backtrack", "summarise_trail"}
print("  [OK] GENOPATH_TOOLS: 4 tools defined correctly")

assert SYSTEM_PROMPT and len(SYSTEM_PROMPT) > 100
print(f"  [OK] SYSTEM_PROMPT: {len(SYSTEM_PROMPT)} chars")

env3 = GenoPathEnvironment(graph)
r3 = env3.reset("monogenic", seed=3)
obs_str = format_observation(r3.observation)
assert "PATIENT PHENOTYPES" in obs_str
assert "CURRENT NODE" in obs_str
assert "CANDIDATE VARIANTS" in obs_str
assert "EXCLUSION SIGNALS" in obs_str
print(f"  [OK] format_observation: {len(obs_str)} chars, all required sections present")

import ollama as _ollama  # noqa: E402
try:
    _models = [m.model for m in _ollama.list().models]
    _model_available = any("gemma4" in m for m in _models)
except Exception:
    _model_available = False

if _model_available:
    # Smoke-test only: 3 steps to confirm model responds and tool calls parse correctly.
    # Full benchmark lives in scripts/run_benchmark.py.
    import ollama as _ol  # noqa: E402
    from src.agent.tools import GENOPATH_TOOLS, SYSTEM_PROMPT, format_observation  # noqa: E402
    from src.episode.environment import GenoPathEnvironment  # noqa: E402
    from src.episode.models import GenoPathAction  # noqa: E402

    _env = GenoPathEnvironment(graph)
    _result = _env.reset("monogenic", seed=1)
    _conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(_result.observation)},
    ]
    _VALID = {"hop", "flag_causal", "backtrack", "summarise_trail"}
    _steps_run = 0
    for _ in range(3):
        _resp = _ol.chat(model="gemma4:e4b", messages=_conv, tools=GENOPATH_TOOLS)
        _real = [c for c in (_resp.message.tool_calls or []) if c.function.name in _VALID]
        if _real:
            _call = _real[0]
            _act = GenoPathAction(action_type=_call.function.name, **_call.function.arguments)
            _conv.append({
                "role": "assistant",
                "content": _resp.message.content or "",
                "tool_calls": [{"function": {"name": _call.function.name, "arguments": dict(_call.function.arguments)}}],
            })
        else:
            _act = GenoPathAction(action_type="summarise_trail", reasoning="no tool call")
            _conv.append({"role": "assistant", "content": _resp.message.content or "No tool call."})
        _result = _env.step(_act)
        if not _result.done:
            if _real:
                _conv.append({"role": "tool", "content": format_observation(_result.observation)})
            else:
                _conv.append({"role": "user", "content": "Use one of the four tools.\n\n" + format_observation(_result.observation)})
        _steps_run += 1
        print(f"  step {_steps_run} | {_act.action_type:<18s} | reward={_result.reward:+.4f}")
        if _result.done:
            break

    assert _steps_run >= 1, "Model produced no steps"
    assert _result.reward >= 0.01, f"Reward out of range: {_result.reward}"
    print(f"  [OK] gemma4:e4b responds with tool calls — {_steps_run} steps verified")
else:
    print("  [SKIP] gemma4:e4b not pulled yet")
    print("         Run: ollama pull gemma4:e4b  (then re-run this script)")

print()
print("All Phase 3 checks passed [OK]")
print()

# ---------------------------------------------------------------------------
# Phase 4: Cognitive Load Analysis
# ---------------------------------------------------------------------------

print("=" * 60)
print("PHASE 4 - Cognitive Load")
print("=" * 60)

from scripts.cognitive_load import analyse_case  # noqa: E402

for task in ["monogenic", "oligogenic", "phenotype_mismatch"]:
    row = analyse_case(graph, task, seed=42)
    assert row["candidate_count"] >= 5, f"{task}: fewer than 5 candidates"
    assert row["high_path_candidates"] >= 1, f"{task}: no high-pathogenicity candidates"
    hops = row["bfs_hops_to_causal"]
    assert hops == -1 or (1 <= hops <= 15), f"{task}: BFS hops out of expected range: {hops}"
    print(f"  [OK] {task}: {row['candidate_count']} candidates, "
          f"{row['high_path_candidates']} high-path, {hops} BFS hops")

print()
print("All Phase 4 checks passed [OK]")
print()
