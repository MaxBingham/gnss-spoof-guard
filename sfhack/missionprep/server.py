"""Mission planner web app (FastAPI, no build step): Leaflet page + JSON API.

    python -m sfhack.missionprep serve            # http://127.0.0.1:8765/

API (all JSON):
  GET  /api/defaults                       default A, B, radius, link bandwidth
  GET  /api/search?q=...                   Nominatim proxy (User-Agent, <= 1 request/s, cached)
  POST /api/prepare {a, b, radius_m, mission_id?, force?}  -> {job}
  POST /api/upload {mission_id, mbps}      -> {job}
  GET  /api/jobs/{job}?since=N             status, progress, log lines from N, upload events, result
  GET  /api/missions                       prepared missions
  GET  /api/missions/{id}                  build report + mission.json + route/bbox in lat/lon
  POST /api/fly {mission_id, night?}       start the live viewer (demos/demo_live.py --mission id [--night]), one at a time
  GET  /api/fly/status                     running / exited, exit code, viewer.log tail after a crash
  POST /api/fly/stop                       stop the viewer
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import REPO_ROOT
from ..geo import GeoOrigin, LatLon
from .build import (DEFAULT_A, DEFAULT_B, DEFAULT_RADIUS_M, default_mission_id, mission_dir, missions_dir,
                    prepare_mission, summarize)
from .sources import Fetcher, Nominatim
from .upload import FLY_COMMAND, is_uploaded, upload

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="sfhack mission planner")
_fetcher = Fetcher()
_geocoder = Nominatim(_fetcher)
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class Point(BaseModel):
    lat: float
    lon: float


class PrepareReq(BaseModel):
    a: Point
    b: Point
    radius_m: float = DEFAULT_RADIUS_M
    mission_id: str | None = None
    force: bool = False


class UploadReq(BaseModel):
    mission_id: str
    mbps: float = 2.0


def _new_job(kind: str) -> dict[str, Any]:
    job = {"id": uuid.uuid4().hex[:10], "kind": kind, "status": "running", "progress": 0.0, "stage": "",
           "log": [], "events": [], "result": None, "error": "", "started": time.time()}
    with _lock:
        _jobs[job["id"]] = job
    return job


def _log(job: dict[str, Any], text: str) -> None:
    for line in text.splitlines():
        job["log"].append(f"{time.time() - job['started']:6.1f}s  {line}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    return {"a": {"lat": DEFAULT_A.lat, "lon": DEFAULT_A.lon}, "b": {"lat": DEFAULT_B.lat, "lon": DEFAULT_B.lon},
            "radius_m": DEFAULT_RADIUS_M, "mbps": 2.0, "center": [37.8727, -122.2592], "zoom": 16}


@app.get("/api/search")
def search(q: str) -> list[dict[str, Any]]:
    q = q.strip()
    if not q or len(q) > 200:
        return []
    try:
        return _geocoder.search(q)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"search failed: {e}") from e


@app.post("/api/prepare")
def prepare(req: PrepareReq) -> dict[str, str]:
    job = _new_job("prepare")

    def run() -> None:
        try:
            _release(req.mission_id or default_mission_id(LatLon(req.a.lat, req.a.lon), LatLon(req.b.lat, req.b.lon),
                                                          req.radius_m))
            rep = prepare_mission(LatLon(req.a.lat, req.a.lon), LatLon(req.b.lat, req.b.lon), req.radius_m,
                                  req.mission_id or None, force=req.force, log=lambda s: _log(job, s),
                                  progress=lambda f, s: job.update(progress=f, stage=s))
            _log(job, summarize(rep))
            job["result"] = {**rep, "mission": _mission_view(rep["mission_id"])}
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["error"] = f"{type(e).__name__}: {e}"
            _log(job, f"FAILED: {job['error']}")
            job["status"] = "failed"

    threading.Thread(target=run, daemon=True).start()
    return {"job": job["id"]}


@app.post("/api/upload")
def upload_pkg(req: UploadReq) -> dict[str, str]:
    try:
        d = mission_dir(req.mission_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not (d / "onboard" / "manifest.json").exists():
        raise HTTPException(404, f"mission {req.mission_id} not prepared")
    job = _new_job("upload")

    def on_event(e: dict[str, Any]) -> None:
        job["events"].append(e)
        if e["type"] == "progress":
            job["progress"] = e["total_sent"] / max(e["total_bytes"], 1)
        if "text" in e:
            _log(job, e["text"])

    def run() -> None:
        try:
            _release(req.mission_id)
            ok = upload(req.mission_id, req.mbps, on_event)
            job["result"] = {"ok": ok, "command": FLY_COMMAND.format(id=req.mission_id)}
            job["status"] = "done" if ok else "failed"
        except Exception as e:  # noqa: BLE001
            job["error"] = f"{type(e).__name__}: {e}"
            job["status"] = "failed"

    threading.Thread(target=run, daemon=True).start()
    return {"job": job["id"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0, events_since: int = 0) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {k: job[k] for k in ("id", "kind", "status", "progress", "stage", "result", "error")} | {
        "log": job["log"][since:], "log_len": len(job["log"]),
        "events": job["events"][events_since:], "events_len": len(job["events"])}


def _mission_view(mid: str) -> dict[str, Any]:
    d = mission_dir(mid)
    rep = json.loads((d / "build_report.json").read_text(encoding="utf-8"))
    man = json.loads((d / "onboard" / "manifest.json").read_text(encoding="utf-8"))
    mission = json.loads((d / "onboard" / "mission.json").read_text(encoding="utf-8"))
    o = GeoOrigin(man["origin"]["lat"], man["origin"]["lon"])
    route = [{"phase": w["phase"], "z": w["enu"][2], "latlon": list(o.to_latlon(w["enu"][0], w["enu"][1]))}
             for w in mission["route"]]
    layers = [{"key": k, "file": v["file"], "bytes": v["bytes"], "description": v["description"]}
              for k, v in man["layers"].items()]
    return {"mission_id": mid, "bbox": man["bbox"], "route": route, "layers": layers, "report": rep,
            "a": mission["a"], "b": mission["b"], "warnings": mission.get("warnings", []),
            "uploaded": is_uploaded(mid), "command": FLY_COMMAND.format(id=mid),
            "cloud": man["map_cloud"]["kind"]}


@app.get("/api/missions")
def missions() -> list[dict[str, Any]]:
    out = []
    for d in sorted(missions_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        rep = d / "build_report.json"
        if rep.exists():
            r = json.loads(rep.read_text(encoding="utf-8"))
            out.append({"mission_id": d.name, "a": r["inputs"]["a"], "b": r["inputs"]["b"],
                        "radius_m": r["inputs"]["radius_m"], "created_at": r["created_at"],
                        "uploaded": is_uploaded(d.name)})
    return out


@app.get("/api/missions/{mid}")
def mission(mid: str) -> dict[str, Any]:
    try:
        d = mission_dir(mid)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not (d / "build_report.json").exists():
        raise HTTPException(404, f"mission {mid} not prepared")
    return _mission_view(mid)


# ------------------------------------------------------------------------------------------- live viewer

VIEWER = REPO_ROOT / "demos" / "demo_live.py"
_viewer: dict[str, Any] = {"proc": None, "mission_id": None, "log": None, "started": 0.0, "fh": None, "stopped": False}
_viewer_lock = threading.Lock()


class FlyReq(BaseModel):
    mission_id: str
    night: bool = False  # night flight: dark scene, thermal camera, key N = GNSS off / on


def _stop_viewer() -> int | None:
    """Terminate the running viewer (if any); returns its exit code. Caller holds _viewer_lock."""
    proc = _viewer["proc"]
    if proc is None:
        return None
    if proc.poll() is None:
        _viewer["stopped"] = True  # our doing: terminate() exits with 1 on Windows, which is not a crash
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if _viewer["fh"] is not None:
        _viewer["fh"].close()
        _viewer["fh"] = None
    return proc.returncode


def _release(mission_id: str) -> None:
    """Stop the viewer if it flies this mission: a rebuild / re-upload replaces files it holds open (Windows)."""
    with _viewer_lock:
        if _viewer["mission_id"] == mission_id:
            _stop_viewer()


def _log_tail(path: Path | None, n: int = 30) -> list[str]:
    if path is None or not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


@app.post("/api/fly")
def fly(req: FlyReq) -> dict[str, Any]:
    """Start demos/demo_live.py --mission <id> with this server's interpreter (the same venv), detached
    from the server's console; one viewer at a time. Output goes to <mission>/viewer.log."""
    try:
        d = mission_dir(req.mission_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not ((d / "world" / "manifest.json").exists() and (d / "onboard" / "manifest.json").exists()):
        raise HTTPException(404, f"mission {req.mission_id} not prepared")
    with _viewer_lock:
        _stop_viewer()
        log = d / "viewer.log"
        fh = open(log, "wb")  # noqa: SIM115 - stays open for the child's lifetime, closed in _stop_viewer
        kw: dict[str, Any] = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        argv = [sys.executable, str(VIEWER), "--mission", req.mission_id] + (["--night"] if req.night else [])
        proc = subprocess.Popen(argv, cwd=str(REPO_ROOT),
                                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"}, **kw)
        _viewer.update(proc=proc, mission_id=req.mission_id, log=log, started=time.time(), fh=fh, stopped=False)
    return {"pid": proc.pid, "mission_id": req.mission_id, "log": str(log), "night": req.night,
            "command": FLY_COMMAND.format(id=req.mission_id) + (" --night" if req.night else "")}


@app.get("/api/fly/status")
def fly_status() -> dict[str, Any]:
    with _viewer_lock:
        proc = _viewer["proc"]
        if proc is None:
            return {"state": "none"}
        code = proc.poll()
        state = "running" if code is None else "stopped" if _viewer["stopped"] else "exited"
        out = {"state": state, "pid": proc.pid, "mission_id": _viewer["mission_id"],
               "exit_code": code, "uptime_s": round(time.time() - _viewer["started"], 1), "log": str(_viewer["log"])}
        if out["state"] == "exited" and code != 0:  # crashed: show why
            out["log_tail"] = _log_tail(_viewer["log"])
        return out


@app.post("/api/fly/stop")
def fly_stop() -> dict[str, Any]:
    with _viewer_lock:
        if _viewer["proc"] is None:
            return {"state": "none"}
        return {"state": "stopped", "pid": _viewer["proc"].pid, "exit_code": _stop_viewer()}
