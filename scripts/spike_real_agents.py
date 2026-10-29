"""Spike: run real Token Factory cookbook agents through the ResilienceGuard proxy on Nemotron.

Run from the project root (about 2-3 minutes):
    .venv/bin/pip install langchain-openai python-dotenv     # once
    .venv/bin/python scripts/spike_real_agents.py

Agents are run unmodified. The launcher shim reroutes their Token Factory calls through a proxy
per run; the proxy injects faults and holds the real API key (agents get a dummy key).
Writes spike_results.json (no secrets).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.parser import FaultProfile  # noqa: E402
from src.proxy import ProxyState, start_in_thread  # noqa: E402

COOKBOOK = ROOT / ".rg_targets" / "token-factory-cookbook"
AGENTS = {
    "customer_support": "agents/langchain/customer-support-resolution-agent",
    "incident_response": "agents/langchain/incident-response-agent",
}
SUPER = "nvidia/nemotron-3-super-120b-a12b"
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SCENARIOS = {
    "baseline_super": (SUPER, []),
    "baseline_nano": (NANO, []),
    "outage_500": (SUPER, [{"action": "error", "status_code": 500, "message": "Simulated 500"}]),
    "rate_limit_429_x3": (SUPER, [{"action": "error", "status_code": 429, "max_occurrences": 3,
                                   "message": "Simulated rate limit"}]),
    "corrupt_first_tool_call": (SUPER, [{"action": "corrupt", "max_occurrences": 1}]),
    "slow_first_call_45s": (SUPER, [{"action": "delay", "delay_seconds": 45, "max_occurrences": 1}]),
}
RUN_TIMEOUT = 150


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ensure_cookbook() -> None:
    if not COOKBOOK.exists():
        COOKBOOK.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/nebius/token-factory-cookbook.git", str(COOKBOOK)], check=True)


def run_case(agent: str, scenario: str, upstream: str, key: str | None) -> dict:
    model, faults = SCENARIOS[scenario]
    state = ProxyState(upstream, FaultProfile(faults=faults), upstream_api_key=key)
    server, proxy_url = start_in_thread(state)
    env = {**os.environ, "NEBIUS_API_KEY": "dummy-proxy-holds-the-real-key", "NEBIUS_MODEL": model,
           "RG_PROXY_URL": proxy_url[:-3], "PYTHONPATH": str(ROOT / "src" / "rg_shim"),
           "PYTHONUNBUFFERED": "1"}
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run([sys.executable, "main.py"], cwd=COOKBOOK / AGENTS[agent], env=env,
                              capture_output=True, text=True, timeout=RUN_TIMEOUT)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out, rc = True, None
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    duration = round(time.monotonic() - t0, 1)
    state.wait_idle(5)
    server.shutdown()
    events = [{k: e.get(k) for k in ("outcome", "status", "fault", "forwarded", "latency_s", "usage",
                                     "tool_calls", "error", "n_messages")} for e in state.events]
    err_lines = [l for l in err.strip().splitlines() if l.strip()]
    return {
        "agent": agent, "scenario": scenario, "model": model, "returncode": rc, "timed_out": timed_out,
        "duration_s": duration, "summary": state.summary(), "final_error": err_lines[-1] if err_lines else None,
        "stdout_tail": out[-1500:], "stderr_tail": err[-2500:], "events": events,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="https://api.tokenfactory.nebius.com/v1")
    ap.add_argument("--agents", default=",".join(AGENTS))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = ap.parse_args()

    load_env()
    key = os.environ.get("API_KEY") or os.environ.get("NEBIUS_API_KEY")
    try:
        import dotenv  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        sys.exit("Missing deps. Run:  .venv/bin/pip install langchain-openai python-dotenv")
    ensure_cookbook()

    cases = [(a, s) for a in args.agents.split(",") for s in args.scenarios.split(",")]
    print(f"Running {len(cases)} cases in parallel (up to {RUN_TIMEOUT}s)...")
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        results = list(pool.map(lambda c: run_case(c[0], c[1], args.upstream, key), cases))

    (ROOT / "spike_results.json").write_text(json.dumps({"ts": time.time(), "results": results}, indent=2))
    print(f"\n{'agent':18} {'scenario':24} {'exit':>5} {'secs':>6} {'calls':>5} {'faults':>6} {'tokens':>7}  final error")
    for r in results:
        s = r["summary"]
        exit_ = "T/O" if r["timed_out"] else str(r["returncode"])
        print(f"{r['agent']:18} {r['scenario']:24} {exit_:>5} {r['duration_s']:>6} {s['calls']:>5} {s['faults']:>6} "
              f"{s['prompt_tokens'] + s['completion_tokens']:>7}  {(r['final_error'] or '')[:70]}")
    print("\nWrote spike_results.json")


if __name__ == "__main__":
    main()
