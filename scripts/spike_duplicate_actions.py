"""Finding 3 spike: does re-running a crashed agent repeat its action step?

For each cookbook agent, run pairs:
  run A ("crash"):  calls go through normally until the model asks for the action tool; every
                    call after that fails with a 500, so the agent crashes right after acting.
  run B ("rerun"):  the same ticket again with no faults, as a person or a job queue would retry it.
A duplicate = the action tool was requested in run A and again in run B.

Run from the project root (about 1-2 minutes):
    .venv/bin/python scripts/spike_duplicate_actions.py
Writes spike_duplicates.json (no secrets).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from spike_real_agents import AGENTS, COOKBOOK, SUPER, ensure_cookbook, load_env  # noqa: E402
from src.config.parser import FaultProfile  # noqa: E402
from src.proxy import ProxyState, start_in_thread  # noqa: E402

ACTION_TOOL = {"customer_support": "draft_refund_or_replacement", "incident_response": "draft_status_update"}
TRIALS = 3
RUN_TIMEOUT = 150


def run_agent(agent: str, faults: list, key: str | None) -> dict:
    upstream = os.environ.get("RG_UPSTREAM", "https://api.tokenfactory.nebius.com/v1")
    state = ProxyState(upstream, FaultProfile(faults=faults), upstream_api_key=key)
    server, proxy_url = start_in_thread(state)
    env = {**os.environ, "NEBIUS_API_KEY": "dummy-proxy-holds-the-real-key", "NEBIUS_MODEL": SUPER,
           "RG_PROXY_URL": proxy_url[:-3], "PYTHONPATH": str(ROOT / "src" / "rg_shim")}
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "main.py"], cwd=COOKBOOK / AGENTS[agent], env=env,
                           capture_output=True, text=True, timeout=RUN_TIMEOUT)
        rc, err, out = p.returncode, p.stderr, p.stdout
    except subprocess.TimeoutExpired:
        rc, err, out = None, "timeout", ""
    state.wait_idle(5)
    server.shutdown()
    actions = [tc for e in state.events for tc in (e.get("tool_calls") or []) if tc["name"] == ACTION_TOOL[agent]]
    err_lines = [l for l in err.strip().splitlines() if l.strip()]
    return {"returncode": rc, "duration_s": round(time.monotonic() - t0, 1),
            "calls": len(state.events), "outcomes": [e["outcome"] for e in state.events],
            "tools": [tc["name"] for e in state.events for tc in (e.get("tool_calls") or [])],
            "action_calls": actions, "final_error": err_lines[-1][:200] if err_lines else None,
            "stdout_tail": out[-400:]}


def run_pair(agent: str, trial: int, key: str | None) -> dict:
    crash_rule = {"action": "error", "status_code": 500, "after_tool_call": ACTION_TOOL[agent],
                  "message": "Simulated 500 right after the action step"}
    a = run_agent(agent, [crash_rule], key)
    b = run_agent(agent, [], key)
    acted_a, acted_b = bool(a["action_calls"]), bool(b["action_calls"])
    return {"agent": agent, "trial": trial, "crash_run": a, "rerun": b,
            "triggered": acted_a and a["returncode"] not in (0, None),
            "duplicate": acted_a and acted_b}


def main() -> None:
    load_env()
    key = os.environ.get("API_KEY") or os.environ.get("NEBIUS_API_KEY")
    ensure_cookbook()
    pairs = [(a, t) for a in AGENTS for t in range(1, TRIALS + 1)]
    print(f"Running {len(pairs)} crash+rerun pairs in parallel...")
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        results = list(pool.map(lambda p: run_pair(p[0], p[1], key), pairs))
    (ROOT / "spike_duplicates.json").write_text(json.dumps({"ts": time.time(), "pairs": results}, indent=2))

    print(f"\n{'agent':18} {'trial':>5}  {'crash run':38} {'rerun':30} duplicate")
    for r in results:
        a, b = r["crash_run"], r["rerun"]
        crash = f"acted={len(a['action_calls'])} exit={a['returncode']} calls={a['calls']}"
        rerun = f"acted={len(b['action_calls'])} exit={b['returncode']} calls={b['calls']}"
        print(f"{r['agent']:18} {r['trial']:>5}  {crash:38} {rerun:30} {'YES' if r['duplicate'] else 'no'}")
    for agent in AGENTS:
        rs = [r for r in results if r["agent"] == agent]
        print(f"{agent}: crashed right after acting in {sum(r['triggered'] for r in rs)}/{len(rs)}, "
              f"acted again on rerun in {sum(r['duplicate'] for r in rs)}/{len(rs)}")
    print("\nWrote spike_duplicates.json")


if __name__ == "__main__":
    main()
