"""
Cognitive load analysis: clinician manual review burden vs. agent hop path.

For 10 seeds x 3 task types:
  - Count candidate variants with pathogenicity_score >= 0.75 (manual review burden)
  - Count BFS hops from starting node to causal variant node (agent path length)

Prints a formatted table and saves results/cognitive_load.json.

Usage:
    python scripts/cognitive_load.py
"""
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.graph.case_generator import generate_case
from src.graph.graph import get_graph

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PATHOGENICITY_THRESHOLD = 0.75
N_SEEDS = 10
TASK_TYPES = ["monogenic", "oligogenic", "phenotype_mismatch"]


def bfs_hop_count(graph, start_id: str, target_ids: set) -> int:
    """
    BFS from start_id; return number of hops to reach any node in target_ids.
    Returns -1 if unreachable within 20 hops.
    """
    if start_id in target_ids:
        return 0
    visited = {start_id}
    queue = deque([(start_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth >= 20:
            continue
        for neighbor_id in graph.get_neighbors(node_id):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            if neighbor_id in target_ids:
                return depth + 1
            queue.append((neighbor_id, depth + 1))
    return -1


def analyse_case(graph, task_type: str, seed: int) -> dict:
    case = generate_case(graph, task_type, seed=seed)

    high_path_count = sum(
        1 for v in case.candidate_variants
        if v.pathogenicity_score >= PATHOGENICITY_THRESHOLD
    )

    causal_node_ids = {f"VAR:{v.allele_id}" for v in case.candidate_variants
                       if v.allele_id in case.causal_allele_ids}
    hops = bfs_hop_count(graph, case.starting_node_id, causal_node_ids)

    return {
        "task_type": task_type,
        "seed": seed,
        "candidate_count": len(case.candidate_variants),
        "high_path_candidates": high_path_count,
        "bfs_hops_to_causal": hops,
    }


def main():
    print("Loading graph...")
    graph = get_graph()
    print(f"Graph ready: {len(graph.nodes):,} nodes\n")

    all_results = []
    summary = {}

    for task in TASK_TYPES:
        task_results = []
        for seed in range(1, N_SEEDS + 1):
            row = analyse_case(graph, task, seed)
            task_results.append(row)
            all_results.append(row)

        valid_hops = [r["bfs_hops_to_causal"] for r in task_results if r["bfs_hops_to_causal"] >= 0]
        avg_candidates = sum(r["candidate_count"] for r in task_results) / len(task_results)
        avg_high_path = sum(r["high_path_candidates"] for r in task_results) / len(task_results)
        avg_hops = sum(valid_hops) / len(valid_hops) if valid_hops else float("nan")
        ratio = avg_high_path / avg_hops if avg_hops > 0 else float("nan")

        summary[task] = {
            "n": N_SEEDS,
            "avg_candidates": round(avg_candidates, 1),
            "avg_high_path_candidates": round(avg_high_path, 1),
            "avg_bfs_hops": round(avg_hops, 1),
            "burden_ratio": round(ratio, 2),
        }

    # Print table
    header = f"{'Task':<22} {'Candidates':>12} {'High-path (>=0.75)':>20} {'BFS hops':>10} {'Burden ratio':>14}"
    print(header)
    print("-" * len(header))
    for task, s in summary.items():
        print(
            f"{task:<22} {s['avg_candidates']:>12.1f} {s['avg_high_path_candidates']:>20.1f} "
            f"{s['avg_bfs_hops']:>10.1f} {s['burden_ratio']:>14.2f}"
        )
    print()
    print("Burden ratio = high-pathogenicity candidates a clinician must manually review")
    print("              divided by BFS hops the agent takes to reach the causal variant.")
    print("Higher ratio => greater reduction in cognitive load from the agent.\n")

    out = {
        "summary": summary,
        "episodes": all_results,
        "threshold": PATHOGENICITY_THRESHOLD,
        "n_seeds": N_SEEDS,
    }
    out_path = RESULTS_DIR / "cognitive_load.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
