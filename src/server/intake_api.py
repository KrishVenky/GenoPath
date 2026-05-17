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

import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.agent.intake import extract_phenotypes_from_image, match_to_hpo
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


class ImagePayload(BaseModel):
    image_b64: str  # base64-encoded image data (no data-URI prefix)
    mime: str = "image/jpeg"


@app.post("/extract-phenotypes")
async def extract_phenotypes(payload: ImagePayload) -> Dict[str, Any]:
    """
    Receive a base64-encoded image from any phone browser.
    Gemma 4 E4B vision extracts phenotype terms; HPO matcher resolves them.
    Returns matched phenotypes ready to pass straight to /intake.
    """
    import tempfile, os
    img_bytes = base64.b64decode(payload.image_b64)
    suffix = ".jpg" if "jpeg" in payload.mime else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(img_bytes)
        tmp = f.name
    try:
        matched = extract_phenotypes_from_image(tmp)
    finally:
        os.unlink(tmp)
    return {"matched": matched}


_CAMERA_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GenoPath — Capture Report</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0a0f1e;color:#e2e8f0;font-family:system-ui,sans-serif;
       min-height:100dvh;display:flex;flex-direction:column;align-items:center;padding:24px 16px}
  h1{font-size:1.2rem;font-weight:600;margin-bottom:4px;color:#a5b4fc}
  .sub{font-size:.85rem;color:#64748b;margin-bottom:28px;text-align:center}
  .card{background:#111827;border:1px solid #1e293b;border-radius:14px;
        padding:20px;width:100%;max-width:420px;margin-bottom:16px}
  label.capture-btn{display:block;background:#4f46e5;color:#fff;text-align:center;
        padding:14px;border-radius:10px;font-size:1rem;font-weight:600;cursor:pointer;
        transition:background .2s}
  label.capture-btn:active{background:#4338ca}
  input[type=file]{display:none}
  #preview{width:100%;border-radius:8px;margin-top:14px;display:none}
  #status{font-size:.85rem;color:#94a3b8;margin-top:10px;min-height:20px;text-align:center}
  .tag-wrap{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .tag{background:#1e1b4b;border:1px solid #4f46e5;color:#a5b4fc;
       padding:4px 10px;border-radius:20px;font-size:.8rem}
  .tag.no-match{border-color:#475569;color:#94a3b8;background:#1e293b}
  .confirm-btn{width:100%;background:#059669;color:#fff;border:none;border-radius:10px;
        padding:14px;font-size:1rem;font-weight:600;cursor:pointer;display:none;
        margin-top:6px;transition:background .2s}
  .confirm-btn:active{background:#047857}
  .spinner{display:inline-block;width:16px;height:16px;border:2px solid #4f46e5;
           border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;
           vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .back{font-size:.8rem;color:#64748b;text-decoration:none;margin-top:auto;padding-top:24px}
  .back:hover{color:#a5b4fc}
</style>
</head>
<body>
<h1>GenoPath</h1>
<p class="sub">Photograph a clinical report — Gemma extracts the phenotypes.</p>

<div class="card">
  <label class="capture-btn" for="cam">
    📷 &nbsp;Capture Clinical Report
  </label>
  <input type="file" id="cam" accept="image/*" capture="environment">
  <img id="preview" alt="captured report">
  <div id="status"></div>
</div>

<div class="card" id="result-card" style="display:none">
  <div style="font-size:.8rem;color:#64748b;margin-bottom:8px">EXTRACTED PHENOTYPES</div>
  <div class="tag-wrap" id="tags"></div>
  <button class="confirm-btn" id="confirm-btn" onclick="confirm()">
    Run GenoPath Analysis →
  </button>
</div>

<a href="/" class="back">← Back to main UI</a>

<script>
let matched = [];

document.getElementById('cam').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Show preview
  const url = URL.createObjectURL(file);
  const prev = document.getElementById('preview');
  prev.src = url;
  prev.style.display = 'block';

  document.getElementById('result-card').style.display = 'none';
  document.getElementById('status').innerHTML =
    '<span class="spinner"></span>Gemma is reading the report…';

  // Base64 encode
  const b64 = await new Promise(res => {
    const reader = new FileReader();
    reader.onload = () => res(reader.result.split(',')[1]);
    reader.readAsDataURL(file);
  });

  try {
    const r = await fetch('/extract-phenotypes', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({image_b64: b64, mime: file.type}),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    matched = data.matched || [];

    document.getElementById('status').textContent =
      matched.length + ' phenotype' + (matched.length !== 1 ? 's' : '') + ' found';

    const tagWrap = document.getElementById('tags');
    tagWrap.innerHTML = matched.map(m =>
      `<span class="tag${m.hpo_id ? '' : ' no-match'}">${m.name}</span>`
    ).join('');

    document.getElementById('confirm-btn').style.display = 'block';
    document.getElementById('result-card').style.display = 'block';
  } catch(err) {
    document.getElementById('status').textContent = 'Error: ' + err.message;
  }
});

async function confirm() {
  const btn = document.getElementById('confirm-btn');
  btn.disabled = true;
  btn.textContent = 'Starting episode…';

  const r = await fetch('/intake', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      raw_phenotypes: matched.map(m => m.raw),
      source: 'camera',
      task_type: 'monogenic',
    }),
  });
  const data = await r.json();
  // Hand off to main UI with episode_id so it can connect to the stream
  window.location.href = '/?episode_id=' + encodeURIComponent(data.episode_id);
}
</script>
</body>
</html>
"""


@app.get("/camera")
async def camera_page():
    return HTMLResponse(_CAMERA_HTML)


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
