"""
Model smoke test — run this and paste the output back.

Tests (in order):
  1. Ollama connectivity + model list
  2. Simple chat response (speed check)
  3. Tool calling (does the model return tool_calls?)
  4. 3-step episode (does the agent navigate the graph?)

Usage:
    python scripts/test_model.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import ollama

MODEL = "gemma4:e4b"
SEP = "=" * 60

# ---------------------------------------------------------------------------
# 1. Connectivity
# ---------------------------------------------------------------------------
print(SEP)
print("TEST 1 — Ollama connectivity + model list")
print(SEP)

try:
    models = [m.model for m in ollama.list().models]
    print(f"  Models pulled: {models}")
    if MODEL in models:
        print(f"  [OK] {MODEL} is present")
    else:
        print(f"  [FAIL] {MODEL} not found in pulled models")
        print(f"         Run: ollama pull {MODEL}")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] Cannot reach Ollama: {e}")
    print("         Is Ollama running? Start it with: ollama serve")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Simple chat (speed)
# ---------------------------------------------------------------------------
print()
print(SEP)
print("TEST 2 — Simple chat response (speed check)")
print(SEP)

t0 = time.time()
resp = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Reply with exactly one word: READY"}],
)
elapsed = time.time() - t0
content = resp.message.content or ""
print(f"  Response ({elapsed:.1f}s): {content.strip()[:120]}")
print(f"  [OK] Model responds in {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# 3. Tool calling
# ---------------------------------------------------------------------------
print()
print(SEP)
print("TEST 3 — Tool calling (does model return tool_calls?)")
print(SEP)

from src.agent.tools import GENOPATH_TOOLS, SYSTEM_PROMPT, format_observation
from src.graph.graph import get_graph
from src.episode.environment import GenoPathEnvironment

print("  Loading graph...")
t0 = time.time()
graph = get_graph()
print(f"  Graph loaded in {time.time()-t0:.1f}s ({len(graph.nodes):,} nodes)")

env = GenoPathEnvironment(graph)
result = env.reset("monogenic", seed=1)
obs_str = format_observation(result.observation)

print(f"  Sending observation ({len(obs_str)} chars) to {MODEL}...")
t0 = time.time()
resp = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_str},
    ],
    tools=GENOPATH_TOOLS,
)
elapsed = time.time() - t0

if resp.message.tool_calls:
    call = resp.message.tool_calls[0]
    print(f"  Response ({elapsed:.1f}s): tool_call={call.function.name}  args={dict(call.function.arguments)}")
    print(f"  [OK] Model returned a tool call")
else:
    text = (resp.message.content or "")[:200]
    print(f"  Response ({elapsed:.1f}s): NO tool call — plain text: {text}")
    print(f"  [WARN] Model did not use a tool — falling back to summarise_trail")

# ---------------------------------------------------------------------------
# 4. 3-step episode
# ---------------------------------------------------------------------------
print()
print(SEP)
print("TEST 4 — 3-step episode (graph navigation)")
print(SEP)

from src.episode.models import GenoPathAction

env2 = GenoPathEnvironment(graph)
result2 = env2.reset("monogenic", seed=1)
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": format_observation(result2.observation)},
]

print(f"  Starting node: {result2.observation.current_node.id} ({result2.observation.current_node.name})")
print(f"  Task: monogenic | Max steps: {result2.observation.max_steps}")
print(f"  Patient phenotypes: {result2.observation.phenotype_names}")
print()

for step_num in range(1, 4):
    if result2.done:
        print(f"  Episode ended early at step {step_num - 1}")
        break

    print(f"  Step {step_num}: calling {MODEL}...")
    t0 = time.time()
    resp2 = ollama.chat(model=MODEL, messages=conversation, tools=GENOPATH_TOOLS)
    elapsed = time.time() - t0

    _VALID = {"hop", "flag_causal", "backtrack", "summarise_trail"}
    real2 = [c for c in (resp2.message.tool_calls or []) if c.function.name in _VALID]
    if real2:
        call2 = real2[0]
        action = GenoPathAction(action_type=call2.function.name, **call2.function.arguments)
        tool_label = f"{call2.function.name}({dict(call2.function.arguments)})"
        conversation.append({
            "role": "assistant",
            "content": resp2.message.content or "",
            "tool_calls": [{"function": {"name": call2.function.name, "arguments": dict(call2.function.arguments)}}],
        })
    else:
        action = GenoPathAction(action_type="summarise_trail", reasoning="no tool call")
        tool_label = "summarise_trail [fallback — no tool call]"
        conversation.append({
            "role": "assistant",
            "content": resp2.message.content or "No tool call.",
        })

    result2 = env2.step(action)

    if real2:
        conversation.append({"role": "tool", "content": format_observation(result2.observation)})
    else:
        conversation.append({
            "role": "user",
            "content": "You must call one of: hop, flag_causal, backtrack, summarise_trail.\n\n" + format_observation(result2.observation),
        })

    print(f"    ({elapsed:.1f}s) {tool_label}")
    print(f"    -> now at: {result2.observation.current_node.id} | reward={result2.reward:+.4f}")

print()
print(SEP)
print("SUMMARY")
print(SEP)
print(f"  Model: {MODEL}")
print(f"  Tool calls fired: {resp2.message.tool_calls is not None}")
print(f"  Steps completed: {result2.observation.step}")
print(f"  Cumulative reward: {result2.observation.cumulative_reward:.4f}")
print(f"  Episode done: {result2.done}")
print()
print("Paste this output back to Claude.")
