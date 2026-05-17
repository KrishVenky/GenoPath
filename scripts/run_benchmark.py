"""
Baseline benchmark: 10 episodes x 3 task types with gemma4:e4b.
Saves results to results/baseline_e4b.json.

Progress is printed per episode so you can watch it run.
Safe to Ctrl+C — partial results are NOT saved (re-run from scratch).

Usage:
    python scripts/run_benchmark.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_FILE = RESULTS_DIR / "baseline_e4b.json"

N_EPISODES = 10
TASK_TYPES = ["monogenic", "oligogenic", "phenotype_mismatch"]
SEP = "=" * 60

print(SEP)
print("GenoPath Baseline Benchmark — gemma4:e4b")
print(f"  {N_EPISODES} episodes x {len(TASK_TYPES)} task types = {N_EPISODES * len(TASK_TYPES)} total")
print(f"  Output: {OUT_FILE}")
print(SEP)
print()

from src.graph.graph import get_graph
from src.agent.ollama_client import run_episode

print("Loading graph...")
t0 = time.time()
graph = get_graph()
print(f"Graph ready in {time.time()-t0:.1f}s ({len(graph.nodes):,} nodes)")
print()

all_results: dict = {}
grand_start = time.time()

for task in TASK_TYPES:
    print(SEP)
    print(f"TASK: {task.upper()}")
    print(SEP)

    task_results = []
    task_start = time.time()

    for seed in range(1, N_EPISODES + 1):
        ep_start = time.time()
        print(f"  [{task}] seed={seed:2d} ...", end="", flush=True)

        ep = run_episode(task, seed=seed, verbose=False)
        ep_elapsed = time.time() - ep_start

        status = "SUCCESS" if ep.success else "miss   "
        steps = len(ep.steps)
        print(f"  {status}  reward={ep.final_reward:.4f}  steps={steps:2d}  ({ep_elapsed:.0f}s)")

        task_results.append({
            "seed": seed,
            "task_type": task,
            "final_reward": ep.final_reward,
            "success": ep.success,
            "steps": steps,
            "step_log": ep.steps,
            "elapsed_s": round(ep_elapsed, 1),
        })

    rewards = [r["final_reward"] for r in task_results]
    successes = [r["success"] for r in task_results]
    mean_r = sum(rewards) / len(rewards)
    success_rate = sum(successes) / len(successes)
    task_elapsed = time.time() - task_start

    all_results[task] = {
        "n": N_EPISODES,
        "mean_reward": round(mean_r, 4),
        "success_rate": round(success_rate, 4),
        "episodes": task_results,
        "elapsed_s": round(task_elapsed, 1),
    }

    print()
    print(f"  -- {task}: mean_reward={mean_r:.4f}  success_rate={success_rate:.0%}  ({task_elapsed/60:.1f} min)")
    print()

# Summary table
total_elapsed = time.time() - grand_start
print(SEP)
print("BENCHMARK COMPLETE")
print(SEP)
print(f"  {'Task':<24} {'Mean reward':>12} {'Success rate':>13} {'Time':>8}")
print(f"  {'-'*24} {'-'*12} {'-'*13} {'-'*8}")
for task, r in all_results.items():
    print(f"  {task:<24} {r['mean_reward']:>12.4f} {r['success_rate']:>12.0%}  {r['elapsed_s']/60:>6.1f}m")
print()
print(f"  Total time: {total_elapsed/60:.1f} min")
print()

out = {
    "model": "gemma4:e4b",
    "n_episodes": N_EPISODES,
    "total_elapsed_s": round(total_elapsed, 1),
    "results": all_results,
}
OUT_FILE.write_text(json.dumps(out, indent=2))
print(f"Saved: {OUT_FILE}")
