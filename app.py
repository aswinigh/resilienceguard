#!/usr/bin/env python3
"""ResilienceGuard Studio: fire drills for AI agents on Nebius Token Factory.

    python app.py            # then open http://localhost:8000

Shows saved real runs from runs/*.json (Agent -> Audit -> Fix), and can start a live run on a
Token Factory cookbook agent, which audits it, has Nemotron fix it, and re-audits the fix.
Standard library only; the page has no external dependencies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PAGE = ROOT / "studio" / "index.html"

AGENTS = {
    "incident_response": {
        "title": "Incident-response agent",
        "source": "Nebius Token Factory cookbook: agents/langchain/incident-response-agent",
        "url": "https://github.com/nebius/token-factory-cookbook/tree/main/agents/langchain/incident-response-agent",
        "framework": "LangChain (ChatOpenAI + tool loop)",
        "does": "Investigates a checkout-latency alert with log, runbook and deploy tools, then drafts a status update.",
        "action": "draft_status_update: the status update customers and staff would see",
    },
    "customer_support": {
        "title": "Customer-support agent",
        "source": "Nebius Token Factory cookbook: agents/langchain/customer-support-resolution-agent",
        "url": "https://github.com/nebius/token-factory-cookbook/tree/main/agents/langchain/customer-support-resolution-agent",
        "framework": "LangChain (ChatOpenAI + tool loop)",
        "does": "Resolves a double-charge ticket with order and policy lookups, then drafts a refund or replacement.",
        "action": "draft_refund_or_replacement: the refund the customer would receive",
    },
}

JOBS: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def list_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
            c = d.get("comparison") or {}
            out.append({"file": p.name, "agent": d.get("agent"), "model": d.get("model"),
                        "timestamp": (d.get("before") or {}).get("timestamp"),
                        "before": f"{c.get('grade_before', '?')} ({c.get('score_before', '?')})",
                        "after": f"{c.get('grade_after', '?')} ({c.get('score_after', '?')})"})
        except (OSError, json.JSONDecodeError):
            continue
    return out


def run_job(job_id: str, agent: str, candidates: int, rounds: int) -> None:
    """Live run: audit the cookbook agent, then have Nemotron fix it and re-audit."""
    job = JOBS[job_id]
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    audit_path = ROOT / f"audit_live_{stamp}.json"
    fix_path = ROOT / f"fix_live_{stamp}.json"
    steps = [
        ("Audit (before)", [sys.executable, "-m", "src.agent_audit", "--cookbook", agent, "--out", str(audit_path)]),
        ("Nemotron fix + re-audit", [sys.executable, "-m", "src.fixer", "--cookbook", agent, "--audit", str(audit_path),
                                     "--candidates", str(candidates), "--rounds", str(rounds), "--out", str(fix_path)]),
    ]
    try:
        for label, cmd in steps:
            job["log"].append(f"== {label}")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                job["log"].append(line.rstrip())
            if proc.wait() != 0:
                raise RuntimeError(f"{label} failed (exit {proc.returncode})")
        name = f"{agent}_live_{stamp}"
        subprocess.run([sys.executable, "scripts/save_run.py", str(fix_path), name], cwd=ROOT, check=True)
        job.update(status="done", run=f"{name}.json")
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

    def _local_json_request(self) -> bool:
        """Only accept POSTs from the studio page itself (a live run executes agent code)."""
        port = self.server.server_address[1]
        hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
        if self.headers.get("Host", "") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in hosts}:
            return False
        return self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/agents":
            self._json(200, AGENTS)
        elif path == "/api/runs":
            self._json(200, list_runs())
        elif path.startswith("/api/runs/"):
            name = Path(unquote(path[len("/api/runs/"):])).name  # no path traversal
            p = RUNS / name
            if p.suffix != ".json" or not p.is_file():
                return self._json(404, {"error": "run not found"})
            data = json.loads(p.read_text())
            data["meta"] = AGENTS.get(data.get("agent"), {"title": data.get("agent")})
            self._json(200, data)
        elif path.startswith("/api/jobs/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            if not job:
                return self._json(404, {"error": "no such job"})
            self._json(200, {k: v for k, v in job.items() if k != "thread"} | {"log": job["log"][-200:]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/live":
            return self._json(404, {"error": "not found"})
        if not self._local_json_request():
            return self._json(403, {"error": "Live runs must be started from the studio page."})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        agent = req.get("agent")
        if agent not in AGENTS:
            return self._json(400, {"error": f"unknown agent; choose one of {sorted(AGENTS)}"})
        with _jobs_lock:
            if any(j["status"] == "running" for j in JOBS.values()):
                return self._json(409, {"error": "A live run is already in progress."})
            job_id = uuid.uuid4().hex[:10]
            JOBS[job_id] = {"id": job_id, "agent": agent, "status": "running", "log": [], "started": time.time()}
        threading.Thread(target=run_job, args=(job_id, agent, int(req.get("candidates", 2)),
                                               int(req.get("rounds", 3))), daemon=True).start()
        self._json(202, {"job": job_id})


def make_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Studio)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ResilienceGuard Studio")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = make_server(args.port)
    print(f"ResilienceGuard Studio on http://localhost:{args.port}  (saved runs: {len(list_runs())})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
