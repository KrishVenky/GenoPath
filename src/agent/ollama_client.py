"""
Desktop inference: Ollama + Gemma 4 E4B function calling loop.

run_episode() runs one complete GenoPath episode:
  - Calls GenoPathEnvironment.reset() to generate a case
  - Sends observations to gemma4:e4b via ollama.chat(tools=GENOPATH_TOOLS)
  - Extracts tool_calls (with fallback to summarise_trail if none returned)
  - Steps the environment until done
  - Returns EpisodeResult
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import ollama

from src.agent.tools import GENOPATH_TOOLS, SYSTEM_PROMPT, format_observation
from src.episode.environment import GenoPathEnvironment
from src.episode.models import EpisodeResult, GenoPathAction
from src.graph.graph import get_graph

_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")


def run_episode(
    task_type: str,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> EpisodeResult:
    """
    Run one full GenoPath episode with the E4B model.

    Args:
        task_type: "monogenic" | "oligogenic" | "phenotype_mismatch"
        seed: optional RNG seed for reproducible case generation
        verbose: if True, print each step's action and reward

    Returns:
        EpisodeResult with task_type, step log, final_reward, and success flag
    """
    graph = get_graph()
    env = GenoPathEnvironment(graph)
    result = env.reset(task_type, seed=seed)

    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(result.observation)},
    ]
    step_log: List[Dict[str, Any]] = []

    while not result.done:
        response = ollama.chat(
            model=_MODEL,
            messages=conversation,
            tools=GENOPATH_TOOLS,
        )

        # Extract tool call — guard against None and Gemma 4 thinking pseudo-calls.
        # Gemma 4 can emit thought_output/thinking as tool_calls; filter to real actions only.
        _VALID_ACTIONS = {"hop", "flag_causal", "backtrack", "summarise_trail"}
        real_calls = [
            c for c in (response.message.tool_calls or [])
            if c.function.name in _VALID_ACTIONS
        ]
        if real_calls:
            call = real_calls[0]
            action_dict: Dict[str, Any] = {
                "action_type": call.function.name,
                **call.function.arguments,
            }
        else:
            action_dict = {
                "action_type": "summarise_trail",
                "reasoning": "no valid tool call returned",
            }

        action = GenoPathAction(**action_dict)

        # Build assistant turn with proper Ollama tool-call format.
        if real_calls:
            conversation.append({
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": [
                    {"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}}
                    for c in real_calls
                ],
            })
        else:
            conversation.append({
                "role": "assistant",
                "content": response.message.content or "No tool call produced.",
            })

        result = env.step(action)
        log_entry = {
            "step": result.observation.step,
            "action": action.action_type,
            "node_id": action.node_id,
            "variant_id": action.variant_id,
            "reward": round(result.reward, 4),
        }
        step_log.append(log_entry)

        if verbose:
            print(
                f"  step {result.observation.step:2d} | {action.action_type:<18s} "
                f"| reward={result.reward:+.4f} | cum={result.observation.cumulative_reward:.4f}"
            )

        if not result.done:
            if real_calls:
                # Proper tool-result turn so the model knows its action was executed.
                conversation.append({
                    "role": "tool",
                    "content": format_observation(result.observation),
                })
            else:
                # Re-prompt: model missed the tool call, nudge it explicitly.
                conversation.append({
                    "role": "user",
                    "content": (
                        "You must call one of the four tools: hop, flag_causal, backtrack, "
                        "or summarise_trail. Do not respond with plain text.\n\n"
                        + format_observation(result.observation)
                    ),
                })

    final_reward = result.reward
    return EpisodeResult(
        task_type=task_type,
        steps=step_log,
        final_reward=round(final_reward, 4),
        success=final_reward > 0.5,
    )


def run_benchmark(
    n_episodes: int = 10,
    task_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run n_episodes per task type. Returns a dict ready to save as JSON.
    Seeds are 1..n_episodes for reproducibility.
    """
    if task_types is None:
        task_types = ["monogenic", "oligogenic", "phenotype_mismatch"]

    results: Dict[str, Any] = {}
    for task in task_types:
        episode_results = []
        for seed in range(1, n_episodes + 1):
            ep = run_episode(task, seed=seed, verbose=verbose)
            episode_results.append(ep.model_dump())
            if verbose:
                print(
                    f"[{task}] seed={seed} final_reward={ep.final_reward:.4f} "
                    f"success={ep.success}"
                )

        rewards = [e["final_reward"] for e in episode_results]
        successes = [e["success"] for e in episode_results]
        results[task] = {
            "episodes": episode_results,
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "success_rate": round(sum(successes) / len(successes), 4),
            "n": n_episodes,
        }
        print(
            f"  {task}: mean_reward={results[task]['mean_reward']:.4f} "
            f"success_rate={results[task]['success_rate']:.2%}"
        )

    return results
