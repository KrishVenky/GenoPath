"""
FastAPI intake server — receives phenotypes from any phone over local WiFi.

Endpoints:
  POST /intake              — phenotype payload -> HPO match -> episode start
  GET  /episode/stream      — SSE stream of agent steps for a running episode
  GET  /health              — liveness check

Run:
  uvicorn src.server.intake_api:app --host 0.0.0.0 --port 8000

Phone sends:  POST http://{laptop_ip}:8000/intake
Both devices must be on the same local WiFi. No internet required.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.agent.intake import match_to_hpo
from src.agent.tools import GENOPATH_TOOLS, SYSTEM_PROMPT, format_observation
from src.episode.environment import GenoPathEnvironment
from src.episode.models import GenoPathAction
from src.graph.graph import get_graph

app = FastAPI(title="GenoPath API", version="1.0.0")

# Open CORS so the single-file HTML frontend can call from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None
_graph_lock = threading.Lock()
_active_episodes: Dict[str, Dict[str, Any]] = {}


def _get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = get_graph()
    return _graph


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PhenotypePayload(BaseModel):
    raw_phenotypes: List[str]
    source: str = "manual"
    task_type: str = "monogenic"
    seed: Optional[int] = None


class IntakeResponse(BaseModel):
    episode_id: str
    matched_phenotypes: List[Dict[str, str]]
    first_observation: Dict[str, Any]
    task_type: str
    stream_url: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_HTML = Path(__file__).parent.parent.parent / "GenoPath.html"


@app.get("/")
async def frontend():
    return FileResponse(_HTML, media_type="text/html")


@app.get("/health")
async def health() -> Dict[str, Any]:
    graph = _get_graph()
    return {
        "status": "ok",
        "graph_loaded": graph is not None,
        "graph_nodes": len(graph.nodes) if graph else 0,
        "model": "gemma4:e4b",
        "local_ip": _local_ip(),
    }


@app.post("/intake", response_model=IntakeResponse)
async def intake(payload: PhenotypePayload) -> IntakeResponse:
    if not payload.raw_phenotypes:
        raise HTTPException(status_code=422, detail="raw_phenotypes must not be empty")

    graph = _get_graph()
    matched = [match_to_hpo(term) for term in payload.raw_phenotypes]

    env = GenoPathEnvironment(graph)
    result = env.reset(payload.task_type, seed=payload.seed)

    episode_id = env._case.case_id or str(uuid.uuid4())[:8]
    _active_episodes[episode_id] = {
        "env": env,
        "result": result,
        "conversation": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(result.observation)},
        ],
        "matched": matched,
    }

    obs_dict = result.observation.model_dump()
    obs_dict["matched_from_image"] = matched
    obs_dict["action_taken"] = None
    obs_dict["action_reasoning"] = None
    obs_dict["reward"] = result.reward
    obs_dict["done"] = result.done

    return IntakeResponse(
        episode_id=episode_id,
        matched_phenotypes=matched,
        first_observation=obs_dict,
        task_type=payload.task_type,
        stream_url=f"http://{_local_ip()}:8000/episode/stream?episode_id={episode_id}",
    )


@app.get("/episode/stream")
async def episode_stream(episode_id: str):
    """
    SSE endpoint — streams one JSON event per agent step until episode ends.
    Connect with: EventSource('/episode/stream?episode_id=xxx')
    """
    if episode_id not in _active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    async def generate():
        ep = _active_episodes[episode_id]
        env: GenoPathEnvironment = ep["env"]
        conversation: list = ep["conversation"]
        result = ep["result"]
        model = "gemma4:e4b"
        _VALID = {"hop", "flag_causal", "backtrack", "summarise_trail"}

        while not result.done:
            await asyncio.sleep(0)  # yield to event loop

            try:
                response = ollama.chat(
                    model=model,
                    messages=conversation,
                    tools=GENOPATH_TOOLS,
                )
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

            real_calls = [
                c for c in (response.message.tool_calls or [])
                if c.function.name in _VALID
            ]
            if real_calls:
                call = real_calls[0]
                action = GenoPathAction(
                    action_type=call.function.name,
                    **call.function.arguments,
                )
                conversation.append({
                    "role": "assistant",
                    "content": response.message.content or "",
                    "tool_calls": [
                        {"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}}
                        for c in real_calls
                    ],
                })
            else:
                action = GenoPathAction(action_type="summarise_trail", reasoning="no valid tool call")
                conversation.append({
                    "role": "assistant",
                    "content": response.message.content or "",
                })

            result = env.step(action)
            obs = result.observation

            if not result.done:
                if real_calls:
                    conversation.append({"role": "tool", "content": format_observation(obs)})
                else:
                    conversation.append({
                        "role": "user",
                        "content": "Use one of the four tools.\n\n" + format_observation(obs),
                    })

            obs_dict = obs.model_dump()
            obs_dict["action_taken"] = action.action_type
            obs_dict["action_reasoning"] = action.reasoning
            obs_dict["reward"] = result.reward
            obs_dict["done"] = result.done
            if env._flagged_allele_ids:
                aid = env._flagged_allele_ids[-1]
                fv = next((v for v in env._case.candidate_variants if v.allele_id == aid), None)
                obs_dict["flagged_variant"] = fv.model_dump() if fv else {"id": f"VAR:{aid}", "allele_id": aid}
            else:
                obs_dict["flagged_variant"] = None
            obs_dict["success"] = (result.reward > 0.5) if result.done else None

            yield f"data: {json.dumps(obs_dict)}\n\n"

        del _active_episodes[episode_id]

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("Warming up graph...")
    _get_graph()
    print(f"Launching GenoPath API on http://0.0.0.0:8000")
    print(f"Local network: http://{_local_ip()}:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
