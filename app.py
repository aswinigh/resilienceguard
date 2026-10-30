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

def _load_env_file() -> None:
    """Local runs: read API_KEY / PROJECT_ID from .env (hosted deployments set real environment variables)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()
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


def drills() -> list[dict]:
    try:
        from src.agent_audit import SCENARIOS
        return [{"id": d.id, "name": d.name, "simulates": d.simulates} for d in SCENARIOS]
    except Exception:  # noqa: BLE001 - config must still load
        return []


def config() -> dict:
    return {"drills": drills(), "fix_eta": "usually 5 to 10 minutes", "audit_eta": "usually about 2 minutes","hosted": HOSTED, "fix_requires_code": HOSTED, "fix_available": (not HOSTED) or bool(ACCESS_CODE),
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


def _run_steps(job: dict, steps: list[tuple[str, list[str]]]) -> None:
    for label, cmd in steps:
        job["log"].append(f"== {label}")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            job["log"].append(line.rstrip())
        if proc.wait() != 0:
            tail = next((l for l in reversed(job["log"]) if l.strip() and not l.startswith("==")), "")
            raise RuntimeError(f"{label} failed: {tail[:300]}")


def _save(job: dict, result: dict, req: dict, suffix: str) -> str:
    result["meta"] = req["meta"]
    out = WORK / f"result_{job['id']}.json"
    out.write_text(json.dumps(result))
    run_name = f"{_slug(req['info']['label'])}_{suffix}_{time.strftime('%Y-%m-%d_%H%M%S')}"
    subprocess.run([sys.executable, "scripts/save_run.py", str(out), run_name], cwd=ROOT, check=True,
                   capture_output=True)
    return f"{run_name}.json"


def _common_args(req: dict) -> list[str]:
    args = ["--workdir", req["info"]["workdir"], "--cmd", req["command"], "--name", req["info"]["label"],
            "--pairs", str(PAIRS)]
    args += [x for t in req["action_tools"] for x in ("--action-tool", t)]
    return args + (["--sandbox"] if req["sandbox"] else [])


def run_audit_job(job_id: str, req: dict) -> None:
    """Step 1: audit the fetched agent (in Sandboxes). The result becomes an audit-only run."""
    job = JOBS[job_id]
    audit_path = WORK / f"audit_{job_id}.json"
    try:
        _run_steps(job, [("Audit", [sys.executable, "-m", "src.agent_audit", *_common_args(req),
                                    "--out", str(audit_path)])])
        audit = json.loads(audit_path.read_text())
        run = _save(job, {"agent": req["info"]["label"], "model": None, "file": None, "before": audit, "rounds": []},
                    req, "audit")
        job.update(status="done", run=run, audit_path=str(audit_path))
    except Exception as e:  # noqa: BLE001 - surfaced to the page
        job.update(status="error", error=str(e))


def run_fix_job(job_id: str, req: dict, audit_path: str) -> None:
    """Step 2: Nemotron writes fixes for that audit; each is re-audited (in Sandboxes branches)."""
    job = JOBS[job_id]
    fix_path = WORK / f"fix_{job_id}.json"
    entry = req["command"].split()[1]
    try:
        _run_steps(job, [("Nemotron fix + re-audit", [sys.executable, "-m", "src.fixer", *_common_args(req),
                          "--file", entry, "--audit", audit_path, "--candidates", "2",
                          "--rounds", "2" if HOSTED else "3", "--out", str(fix_path)])])
        result = json.loads(fix_path.read_text())
        result["agent"] = req["info"]["label"]
        job.update(status="done", run=_save(job, result, req, "fix"))
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=str(e))


def _start_job(kind: str, label: str, target, args: tuple) -> tuple[int, dict]:
    with _lock:
        if any(j["status"] == "running" for j in JOBS.values()):
            return 409, {"error": "A run is already in progress. Try again in a few minutes."}
        if jobs_today() >= MAX_JOBS_PER_DAY:
            return 429, {"error": "Today's run limit is used up. Example results are still viewable."}
        job_id = uuid.uuid4().hex[:10]
        JOBS[job_id] = {"id": job_id, "kind": kind, "agent": label, "status": "running", "log": [],
                        "started": time.time()}
    threading.Thread(target=target, args=(job_id, *args), daemon=True).start()
    return 202, {"job": job_id}


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
            public = {k: v for k, v in job.items() if k not in ("req", "audit_path")}
            self._json(200, public | {"log": job["log"][-200:], "elapsed": round(time.time() - job["started"])})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/inspect", "/api/live", "/api/fix"):
            return self._json(404, {"error": "not found"})
        if not self._same_origin_json():
            return self._json(403, {"error": "Requests must come from the studio page."})
        req = self._body()
        if req is None:
            return self._json(400, {"error": "invalid request"})
        handler = {"/api/inspect": self._inspect, "/api/live": self._live, "/api/fix": self._fix}[path]
        return handler(req)

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
        """Start step 1, the audit, for an agent fetched with /api/inspect."""
        info = INSPECTED.get(str(req.get("token") or ""))
        if not info:
            return self._json(400, {"error": "Fetch the agent first."})
        command = str(req.get("command") or info["command"] or "").strip()
        if not re.fullmatch(r"python3? [A-Za-z0-9_./-]+\.py( [A-Za-z0-9_./=-]+)*", command):
            return self._json(400, {"error": "Command must look like: python main.py"})
        tools = [t for t in req.get("action_tools") or [] if re.fullmatch(r"[A-Za-z_]\w{0,80}", str(t))]
        if not sandbox_available():  # fetched code never runs on this machine
            return self._json(503, {"error": "Token Factory Sandboxes aren't configured on this server."})
        preset = AGENTS.get(info.get("preset") or "")
        meta = ({k: v for k, v in preset.items() if k != "action_tools"} if preset else
                {"title": info["label"], "source": f"GitHub: {info['label']}", "url": info["url"],
                 "framework": "Detected from the repository", "does": f"Entry point: {command}.",
                 "action": ", ".join(tools) or "No action tools named"})
        run_req = {"info": info, "command": command, "action_tools": tools, "sandbox": True, "meta": meta}
        status, body = _start_job("audit", info["label"], run_audit_job, (run_req,))
        if status == 202:
            JOBS[body["job"]]["req"] = run_req
        self._json(status, body)

    def _fix(self, req: dict) -> None:
        """Start step 2: Nemotron fixes the agent that a finished audit job just audited."""
        audit_job = JOBS.get(str(req.get("job") or ""))
        if not audit_job or audit_job.get("kind") != "audit" or audit_job.get("status") != "done":
            return self._json(400, {"error": "Run an audit first; the fix builds on it."})
        if HOSTED and not (ACCESS_CODE and hmac.compare_digest(str(req.get("code") or ""), ACCESS_CODE)):
            return self._json(403, {"error": "The Nemotron fix needs the access code."})
        if not HOSTED and not sandbox_available():
            return self._json(503, {"error": "Token Factory Sandboxes aren't configured on this server."})
        run_req = audit_job["req"]
        self._json(*_start_job("fix", run_req["info"]["label"], run_fix_job, (run_req, audit_job["audit_path"])))


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
