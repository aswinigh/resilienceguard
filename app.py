#!/usr/bin/env python3
"""ResilienceGuard Studio: fire drills for AI agents on Nebius Token Factory.

    python app.py                     # local:  http://localhost:8000
    RG_HOSTED=1 python app.py         # hosted: sandbox-only, daily cap, fixes need RG_ACCESS_CODE

Shows saved runs from runs/*.json (Agent -> Audit -> Fix) and starts live runs on any public GitHub agent.
Fetched agents never run on this server: their audits run in Token Factory Sandboxes microVMs.

Environment: API_KEY and PROJECT_ID (Token Factory), RG_HOSTED, RG_ACCESS_CODE, RG_MAX_JOBS_PER_DAY, PORT.
Standard library server; the page has no external dependencies.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PAGE = ROOT / "studio" / "index.html"
WORK = Path(tempfile.gettempdir()) / "rg_studio_agents"

HOSTED = os.environ.get("RG_HOSTED") == "1"
ACCESS_CODE = os.environ.get("RG_ACCESS_CODE") or None
MAX_JOBS_PER_DAY = int(os.environ.get("RG_MAX_JOBS_PER_DAY", "10"))
PAIRS = 1 if HOSTED else 3

COOKBOOK = "https://github.com/nebius/token-factory-cookbook/tree/main/agents/langchain"
AGENTS = {
    "incident_response": {
        "title": "Incident-response agent",
        "source": "Nebius Token Factory cookbook: agents/langchain/incident-response-agent",
        "url": f"{COOKBOOK}/incident-response-agent",
        "framework": "LangChain (ChatOpenAI + tool loop)",
        "does": "Investigates a checkout-latency alert with log, runbook and deploy tools, then drafts a status update.",
        "action": "draft_status_update: the status update customers and staff would see",
        "action_tools": ["draft_status_update"],
    },
    "customer_support": {
        "title": "Customer-support agent",
        "source": "Nebius Token Factory cookbook: agents/langchain/customer-support-resolution-agent",
        "url": f"{COOKBOOK}/customer-support-resolution-agent",
        "framework": "LangChain (ChatOpenAI + tool loop)",
        "does": "Resolves a double-charge ticket with order and policy lookups, then drafts a refund or replacement.",
        "action": "draft_refund_or_replacement: the refund the customer would receive",
        "action_tools": ["draft_refund_or_replacement"],
    },
}

JOBS: dict[str, dict] = {}
INSPECTED: dict[str, dict] = {}
_lock = threading.Lock()


def sandbox_available() -> bool:
    try:
        import contree_sdk  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("API_KEY") or os.environ.get("NEBIUS_API_KEY")) and bool(
        os.environ.get("PROJECT_ID") or os.environ.get("NEBIUS_PROJECT_ID"))


def jobs_today() -> int:
    today = dt.datetime.now(dt.timezone.utc).date()
    return sum(1 for j in JOBS.values() if dt.datetime.fromtimestamp(j["started"], dt.timezone.utc).date() == today)


def config() -> dict:
    return {"hosted": HOSTED, "fix_requires_code": HOSTED, "fix_available": (not HOSTED) or bool(ACCESS_CODE),
            "sandbox_available": sandbox_available(), "jobs_left_today": max(0, MAX_JOBS_PER_DAY - jobs_today()),
            "busy": any(j["status"] == "running" for j in JOBS.values()), "pairs": PAIRS,
            "presets": {k: {"title": v["title"], "url": v["url"], "action_tools": v["action_tools"]} for k, v in AGENTS.items()}}


def list_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        c, b = d.get("comparison") or {}, d.get("before") or {}
        out.append({"file": p.name, "agent": d.get("agent"), "model": d.get("model"), "timestamp": b.get("timestamp"),
                    "before": f"{b.get('grade', '?')} ({b.get('score', '?')})",
                    "after": f"{c['grade_after']} ({c['score_after']})" if c else "audit only"})
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "agent"


def run_job(job_id: str, req: dict) -> None:
    """Audit (and optionally fix) a fetched agent; audits run in Sandboxes whenever available."""
    job = JOBS[job_id]
    info, stamp = req["info"], time.strftime("%Y-%m-%d_%H%M%S")
    name = info["label"]
    audit_path, fix_path = WORK / f"audit_{job_id}.json", WORK / f"fix_{job_id}.json"
    common = ["--workdir", info["workdir"], "--cmd", req["command"], "--name", name, "--pairs", str(PAIRS)]
    common += [x for t in req["action_tools"] for x in ("--action-tool", t)]
    sb = ["--sandbox"] if req["sandbox"] else []
    steps = [("Audit", [sys.executable, "-m", "src.agent_audit", *common, "--out", str(audit_path), *sb])]
    if req["fix"]:
        entry = req["command"].split()[-1]
        steps.append(("Nemotron fix + re-audit", [sys.executable, "-m", "src.fixer", *common, "--file", entry,
                      "--audit", str(audit_path), "--candidates", "2", "--rounds", "2" if HOSTED else "3",
                      "--out", str(fix_path), *sb]))
    try:
        for label, cmd in steps:
            job["log"].append(f"== {label}")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                job["log"].append(line.rstrip())
            if proc.wait() != 0:
                raise RuntimeError(f"{label} failed (exit {proc.returncode})")
        if not req["fix"]:
            audit = json.loads(audit_path.read_text())
            fix_path.write_text(json.dumps({"agent": name, "model": None, "file": None, "before": audit, "rounds": []}))
        result = json.loads(fix_path.read_text())
        result["meta"] = req["meta"]
        fix_path.write_text(json.dumps(result))
        run_name = f"{_slug(name)}_live_{stamp}"
        subprocess.run([sys.executable, "scripts/save_run.py", str(fix_path), run_name], cwd=ROOT, check=True,
                       capture_output=True)
        job.update(status="done", run=f"{run_name}.json")
    except Exception as e:  # noqa: BLE001 - surfaced to the page
        job.update(status="error", error=str(e))


class Studio(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[studio] {self.address_string()} {fmt % args}\n")

    def _json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _same_origin_json(self) -> bool:
        """POSTs start agent runs, so only accept them from this studio's own page."""
        host = self.headers.get("Host", "")
        if not HOSTED:
            port = self.server.server_address[1]
            if host not in {f"localhost:{port}", f"127.0.0.1:{port}"}:
                return False
        origin = self.headers.get("Origin")
        if origin is not None and urlparse(origin).netloc != host:
            return False
        return self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json"

    def _body(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            return None
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return None

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/config":
            self._json(200, config())
        elif path == "/api/agents":
            self._json(200, AGENTS)
        elif path == "/api/runs":
            self._json(200, list_runs())
        elif path.startswith("/api/runs/"):
            p = RUNS / Path(unquote(path[len("/api/runs/"):])).name  # no path traversal
            if p.suffix != ".json" or not p.is_file():
                return self._json(404, {"error": "run not found"})
            data = json.loads(p.read_text())
            data["meta"] = data.get("meta") or AGENTS.get(data.get("agent"), {"title": data.get("agent")})
            self._json(200, data)
        elif path.startswith("/api/jobs/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            if not job:
                return self._json(404, {"error": "no such job"})
            self._json(200, {k: v for k, v in job.items() if k != "req"} | {"log": job["log"][-200:]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/inspect", "/api/live"):
            return self._json(404, {"error": "not found"})
        if not self._same_origin_json():
            return self._json(403, {"error": "Requests must come from the studio page."})
        req = self._body()
        if req is None:
            return self._json(400, {"error": "invalid request"})
        return self._inspect(req) if path == "/api/inspect" else self._live(req)

    def _inspect(self, req: dict) -> None:
        from src.github_fetch import fetch_agent

        url = str(req.get("url") or "")
        preset = AGENTS.get(req.get("preset") or "")
        if preset:
            url = preset["url"]
        token = uuid.uuid4().hex[:12]
        dest = WORK / token
        try:
            info = fetch_agent(url, dest)
        except Exception as e:  # noqa: BLE001 - shown to the user
            shutil.rmtree(dest, ignore_errors=True)
            return self._json(400, {"error": str(e)[:300]})
        record = {"label": info.ref.label, "url": info.ref.url, "workdir": str(info.workdir), "files": info.files,
                  "entry": info.entry, "command": info.command, "tools": info.tools,
                  "suggested_action_tools": preset["action_tools"] if preset else info.suggested_action_tools,
                  "has_requirements": info.has_requirements, "preset": req.get("preset") if preset else None}
        with _lock:
            INSPECTED[token] = record
            for old in list(INSPECTED)[:-30]:  # keep the latest 30 fetched agents
                shutil.rmtree(WORK / old, ignore_errors=True)
                INSPECTED.pop(old, None)
        self._json(200, {"token": token, **{k: v for k, v in record.items() if k != "workdir"}})

    def _live(self, req: dict) -> None:
        info = INSPECTED.get(str(req.get("token") or ""))
        if not info:
            return self._json(400, {"error": "Fetch the agent first."})
        command = str(req.get("command") or info["command"] or "").strip()
        if not re.fullmatch(r"python3? [A-Za-z0-9_./-]+\.py( [A-Za-z0-9_./=-]+)*", command):
            return self._json(400, {"error": "Command must look like: python main.py"})
        tools = [t for t in req.get("action_tools") or [] if re.fullmatch(r"[A-Za-z_]\w{0,80}", str(t))]
        fix = bool(req.get("fix"))
        if fix and HOSTED and not (ACCESS_CODE and hmac.compare_digest(str(req.get("code") or ""), ACCESS_CODE)):
            return self._json(403, {"error": "The Nemotron fix loop needs the access code."})
        sandbox = sandbox_available()
        if not sandbox:  # fetched code never runs on this machine
            return self._json(503, {"error": "Token Factory Sandboxes aren't configured on this server."})
        preset = AGENTS.get(info.get("preset") or "")
        meta = ({k: v for k, v in preset.items() if k != "action_tools"} if preset else
                {"title": info["label"], "source": f"GitHub: {info['label']}", "url": info["url"],
                 "framework": "Detected from the repository", "does": f"Entry point: {command}.",
                 "action": ", ".join(tools) or "No action tools named"})
        with _lock:
            if any(j["status"] == "running" for j in JOBS.values()):
                return self._json(409, {"error": "A run is already in progress. Try again in a few minutes."})
            if jobs_today() >= MAX_JOBS_PER_DAY:
                return self._json(429, {"error": "Today's run limit is used up. Saved runs are still viewable."})
            job_id = uuid.uuid4().hex[:10]
            JOBS[job_id] = {"id": job_id, "agent": info["label"], "status": "running", "log": [], "started": time.time(),
                            "fix": fix}
        run_req = {"info": info, "command": command, "action_tools": tools, "fix": fix, "sandbox": sandbox, "meta": meta}
        threading.Thread(target=run_job, args=(job_id, run_req), daemon=True).start()
        self._json(202, {"job": job_id})


def make_server(port: int, host: str | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host or ("0.0.0.0" if HOSTED else "127.0.0.1"), port), Studio)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ResilienceGuard Studio")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--host")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    srv = make_server(args.port, args.host)
    mode = "hosted (sandbox-only)" if HOSTED else "local"
    print(f"ResilienceGuard Studio [{mode}] on http://{srv.server_address[0]}:{args.port}  (saved runs: {len(list_runs())})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
