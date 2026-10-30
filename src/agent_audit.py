"""Audit a real agent, unmodified, by running it through the ResilienceGuard proxy.

The agent is any command (``python main.py``, ``npm start``...). Each scenario runs it in a fresh
copy of its working directory, with its own proxy that injects one kind of provider trouble.
Python agents are rerouted automatically by the launcher shim; others can point their
OpenAI-compatible base URL at ``RG_PROXY_URL``/v1.

Scenarios and what they check (plain language):

* baseline            - nothing goes wrong: does the agent finish at all?
* brief_outage        - the provider fails 4 calls in a row, then recovers: does the agent wait it out?
* rate_limit          - 4 "too busy, retry in 1s" (429) answers, then recovers: does it wait it out?
* malformed_tool_call - one tool call arrives with broken JSON: does the agent recover?
* crash_after_action  - the agent acts (e.g. posts an update), then crashes; the task is rerun:
                        is the action performed twice?
* sustained_outage    - everything fails: does the agent stop within a bounded number of calls?

Scores come only from what the runs did: exit codes, proxy call logs, and the tool results the
agent itself sent back to the model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.parser import FaultProfile
from src.proxy import ProxyState, start_in_thread

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHIM_DIR = PROJECT_ROOT / "src" / "rg_shim"
TOKEN_FACTORY = "https://api.tokenfactory.nebius.com/v1"
NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"
SKIP_PATTERN = re.compile(r"\b(skip|skipped|already|duplicate|idempoten)", re.IGNORECASE)
COPY_IGNORE = shutil.ignore_patterns(".git", ".venv", "venv", "node_modules", "__pycache__", ".rg_targets",
                                     "effects.log", ".action_ledger.json")
SUSTAINED_CALL_LIMIT = 10
SECRET_ENV_NAMES = {"API_KEY", "NEBIUS_API_KEY", "PROJECT_ID", "NEBIUS_PROJECT_ID", "OPENAI_API_KEY"}


@dataclass
class AgentSpec:
    name: str
    workdir: Path
    command: list[str]
    action_tools: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 150.0


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    simulates: str
    passes_if: str
    faults: tuple = ()
    weight: int = 0
    kind: str = "single"  # "single" or "pair"


def _err(status: int, **kw: Any) -> dict:
    return {"action": "error", "status_code": status, "message": f"Simulated {status} from provider", **kw}


SCENARIOS: list[Scenario] = [
    Scenario("baseline", "Normal run", "Nothing goes wrong.", "The agent finishes with an answer."),
    Scenario("brief_outage", "Brief provider outage",
             "The provider fails the next 4 calls in a row (server error 500), then recovers - a few seconds of trouble.",
             "The agent waits it out and finishes with an answer.", (_err(500, max_occurrences=4),), 25),
    Scenario("rate_limit", "Rate-limit burst",
             "The provider answers the next 4 calls with 429 'too busy, retry in 1 second', then recovers.",
             "The agent waits it out and finishes with an answer.", (_err(429, max_occurrences=4),), 20),
    Scenario("malformed_tool_call", "Garbled tool call",
             "The model's first tool call arrives with its arguments cut off (broken JSON).",
             "The agent recovers and finishes with an answer.", ({"action": "corrupt", "max_occurrences": 1},), 20),
    Scenario("crash_after_action", "Crash after acting, then rerun",
             "Everything works until the agent performs its real-world action; then the provider goes down and the "
             "agent crashes. The same task is run again, as a person or a job queue would.",
             "The action is not performed a second time.", (), 25, "pair"),
    Scenario("sustained_outage", "Sustained outage", "Every call fails for the whole run.",
             f"The agent stops on its own within {SUSTAINED_CALL_LIMIT} calls (no hang, no endless retrying).",
             (_err(500),), 10),
]


def _grade(score: int) -> str:
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


class AgentAuditor:
    def __init__(self, upstream: str = TOKEN_FACTORY, api_key: str | None = None,
                 scenarios: list[Scenario] | None = None, pairs: int = 3, parallel: bool = True) -> None:
        self.upstream = upstream
        self.api_key = api_key
        self.scenarios = scenarios or SCENARIOS
        self.pairs = pairs
        self.parallel = parallel

    # ------------------------------------------------------------------ one run

    def run_once(self, spec: AgentSpec, faults: list[dict], workdir: Path) -> dict:
        state = ProxyState(self.upstream, FaultProfile(faults=faults), upstream_api_key=self.api_key)
        server, proxy_url = start_in_thread(state)
        pythonpath = os.pathsep.join(p for p in (str(SHIM_DIR), os.environ.get("PYTHONPATH", "")) if p)
        # The agent is untrusted: it must never see the real key, only the proxy holds it.
        inherited = {k: v for k, v in os.environ.items()
                     if not (self.api_key and v == self.api_key) and k not in SECRET_ENV_NAMES}
        env = {**inherited, "NEBIUS_API_KEY": "dummy-the-proxy-holds-the-real-key", **spec.env,
               "RG_PROXY_URL": proxy_url[:-3], "PYTHONPATH": pythonpath, "PYTHONUNBUFFERED": "1"}
        t0 = time.monotonic()
        timed_out = False
        try:
            p = subprocess.run(spec.command, cwd=workdir, env=env, capture_output=True, text=True,
                               timeout=spec.timeout_s)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            timed_out, rc = True, None
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        duration = round(time.monotonic() - t0, 2)
        state.wait_idle(5)
        server.shutdown()
        summary = state.summary()
        err_lines = [l for l in err.strip().splitlines() if l.strip()]
        return {
            "returncode": rc, "timed_out": timed_out, "duration_s": duration,
            "calls": summary["calls"], "faults": summary["faults"],
            "tokens": {"prompt": summary["prompt_tokens"], "completion": summary["completion_tokens"]},
            "final_error": err_lines[-1][:300] if err_lines else None,
            "stdout_tail": out[-800:],
            "stderr_tail": err[-2500:],
            "timeline": [{k: e.get(k) for k in ("outcome", "status", "fault", "latency_s")} for e in state.events],
            "actions": self._actions(state.events, spec.action_tools),
            "tools_requested": [tc["name"] for e in state.events for tc in (e.get("tool_calls") or [])],
        }

    @staticmethod
    def _actions(events: list[dict], action_tools: list[str]) -> list[dict]:
        """Action-tool calls the model made, with whether the agent actually executed them.

        Execution evidence is the tool result the agent sent back on a later call. A result that
        says it was skipped/already done counts as not executed. A call with no result seen is
        counted as executed (the agent may have run it and crashed before reporting back).
        """
        results: dict[str, str] = {}
        for e in events:
            for r in e.get("tool_results") or []:
                if r.get("tool_call_id"):
                    results[r["tool_call_id"]] = r["content"]
        out = []
        for e in events:
            for tc in e.get("tool_calls") or []:
                if tc.get("name") in action_tools:
                    result = results.get(tc.get("id"))
                    executed = result is None or not SKIP_PATTERN.search(result)
                    out.append({"tool": tc["name"], "arguments": (tc.get("arguments") or "")[:300],
                                "result": result[:300] if result else None, "executed": executed,
                                "result_seen": result is not None})
        return out

    @staticmethod
    def _fresh_copy(spec: AgentSpec, root: Path, label: str) -> Path:
        dest = root / label
        shutil.copytree(spec.workdir, dest, ignore=COPY_IGNORE)
        return dest

    # ------------------------------------------------------------------ scenarios

    def _single(self, spec: AgentSpec, sc: Scenario, root: Path) -> dict:
        run = self.run_once(spec, list(sc.faults), self._fresh_copy(spec, root, sc.id))
        exited_ok = run["returncode"] == 0 and not run["timed_out"]
        answered = bool(run["stdout_tail"].strip())
        finished = exited_ok and answered
        res: dict[str, Any] = {"runs": [run], "applicable": True}
        if sc.id == "sustained_outage":
            stopped = not run["timed_out"]
            res["passed"] = stopped and run["calls"] <= SUSTAINED_CALL_LIMIT
            res["verdict"] = (f"Stopped after {run['calls']} calls in {run['duration_s']}s." if res["passed"] else
                              f"Still running after {run['duration_s']}s and {run['calls']} calls." if not stopped else
                              f"Stopped, but only after {run['calls']} calls.")
        else:
            res["passed"] = finished
            if finished:
                res["verdict"] = f"Finished in {run['duration_s']}s ({run['calls']} calls)."
            elif exited_ok and not answered:
                res["verdict"] = (f"Silent failure: exited normally after {run['calls']} calls "
                                  "but produced no answer.")
            elif run["timed_out"]:
                res["verdict"] = f"Hung: still running after {run['duration_s']}s."
            else:
                why = run["final_error"] or f"exit code {run['returncode']}"
                res["verdict"] = f"Crashed after {run['calls']} calls in {run['duration_s']}s: {why}"
        return res

    def _pair(self, spec: AgentSpec, sc: Scenario, root: Path) -> dict:
        if not spec.action_tools:
            return {"runs": [], "applicable": False, "passed": None,
                    "verdict": "Not applicable: no action tools were named for this agent."}
        rules = [_err(500, after_tool_call=t, message="Simulated 500 right after the action") for t in spec.action_tools]

        def one_pair(i: int) -> dict:
            workdir = self._fresh_copy(spec, root, f"{sc.id}_{i}")  # crash run and rerun share state, like reality
            crash = self.run_once(spec, rules, workdir)
            rerun = self.run_once(spec, [], workdir)
            acted = any(a["executed"] for a in crash["actions"])
            triggered = acted and crash["returncode"] not in (0, None)
            repeated = triggered and any(a["executed"] for a in rerun["actions"])
            return {"pair": i, "crash_run": crash, "rerun": rerun, "triggered": triggered, "repeated": repeated}

        pairs = [one_pair(i) for i in range(1, self.pairs + 1)]
        triggered = [p for p in pairs if p["triggered"]]
        repeated = [p for p in triggered if p["repeated"]]
        res: dict[str, Any] = {"pairs": pairs, "triggered": len(triggered), "repeated": len(repeated)}
        if not triggered:
            res.update(applicable=False, passed=None,
                       verdict=f"Inconclusive: the agent never performed its action in {self.pairs} attempts.")
        else:
            res.update(applicable=True, passed=not repeated,
                       verdict=(f"Performed the action again on rerun in {len(repeated)} of {len(triggered)} tries."
                                if repeated else
                                f"Did not repeat the action on rerun ({len(triggered)} of {len(triggered)} tries)."))
        return res

    # ------------------------------------------------------------------ audit

    def audit(self, spec: AgentSpec) -> dict:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rg_agent_audit_") as tmp:
            root = Path(tmp)

            def run(sc: Scenario) -> dict:
                body = self._pair(spec, sc, root) if sc.kind == "pair" else self._single(spec, sc, root)
                return {"id": sc.id, "name": sc.name, "simulates": sc.simulates, "passes_if": sc.passes_if,
                        "weight": sc.weight, **body}

            if self.parallel:
                with ThreadPoolExecutor(max_workers=len(self.scenarios)) as pool:
                    results = list(pool.map(run, self.scenarios))
            else:
                results = [run(sc) for sc in self.scenarios]

        baseline = next((r for r in results if r["id"] == "baseline"), None)
        scored = [r for r in results if r["weight"] and r["applicable"]]
        total = sum(r["weight"] for r in scored) or 1
        score = round(100 * sum(r["weight"] for r in scored if r["passed"]) / total)
        all_runs = [run for r in results for run in r.get("runs", [])] + \
                   [run for r in results for p in r.get("pairs", []) for run in (p["crash_run"], p["rerun"])]
        return {
            "agent": spec.name,
            "command": " ".join(spec.command),
            "action_tools": spec.action_tools,
            "agent_model": spec.env.get("NEBIUS_MODEL"),
            "upstream": self.upstream,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "audit_seconds": round(time.monotonic() - start, 1),
            "baseline_ok": bool(baseline and baseline["passed"]),
            "score": score,
            "grade": _grade(score),
            "passed": [r["name"] for r in scored if r["passed"]],
            "failed": [r["name"] for r in scored if not r["passed"]],
            "tokens": {"prompt": sum(r["tokens"]["prompt"] for r in all_runs),
                       "completion": sum(r["tokens"]["completion"] for r in all_runs)},
            "scenarios": results,
        }


def compare(before: dict, after: dict) -> dict:
    """Before/after view of two audits of the same agent (e.g. original vs patched)."""
    rows = []
    for b in before["scenarios"]:
        a = next((x for x in after["scenarios"] if x["id"] == b["id"]), None)
        if a:
            rows.append({"id": b["id"], "name": b["name"], "before": b["passed"], "after": a["passed"],
                         "before_verdict": b["verdict"], "after_verdict": a["verdict"]})
    return {"score_before": before["score"], "score_after": after["score"],
            "grade_before": before["grade"], "grade_after": after["grade"],
            "fixed": [r["name"] for r in rows if r["before"] is False and r["after"] is True],
            "regressed": [r["name"] for r in rows if r["before"] is True and r["after"] is False],
            "rows": rows}


# ---------------------------------------------------------------------- CLI

COOKBOOK = PROJECT_ROOT / ".rg_targets" / "token-factory-cookbook"
PRESETS = {
    "incident_response": ("agents/langchain/incident-response-agent", ["draft_status_update"]),
    "customer_support": ("agents/langchain/customer-support-resolution-agent", ["draft_refund_or_replacement"]),
}


def _load_env_file() -> None:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def print_report(audit: dict) -> None:
    print(f"\n{audit['agent']}: grade {audit['grade']} ({audit['score']}/100) in {audit['audit_seconds']}s, "
          f"{audit['tokens']['prompt'] + audit['tokens']['completion']:,} tokens reported by the provider")
    if not audit["baseline_ok"]:
        print("  WARNING: the agent did not finish even with no faults; other results are unreliable.")
    for r in audit["scenarios"]:
        mark = "N/A " if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
        print(f"  [{mark}] {r['name']}: {r['verdict']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Audit an agent through the ResilienceGuard proxy")
    ap.add_argument("--cookbook", choices=sorted(PRESETS), help="audit a Token Factory cookbook agent")
    ap.add_argument("--workdir", help="agent working directory")
    ap.add_argument("--cmd", help='agent command, e.g. "python main.py"')
    ap.add_argument("--action-tool", action="append", default=[], help="tool with real-world effects (repeatable)")
    ap.add_argument("--model", default=NEMOTRON_SUPER, help="sets NEBIUS_MODEL for the agent")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the agent (repeatable)")
    ap.add_argument("--upstream", default=TOKEN_FACTORY)
    ap.add_argument("--key-env", default="API_KEY", help="env var (or .env entry) holding the real API key")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", help="write the audit JSON here")
    ap.add_argument("--sandbox", action="store_true", help="run the audit inside a Token Factory Sandboxes microVM")
    ap.add_argument("--name", help="display name for the agent in reports")
    args = ap.parse_args(argv)

    _load_env_file()
    if args.cookbook:
        if not COOKBOOK.exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1",
                            "https://github.com/nebius/token-factory-cookbook.git", str(COOKBOOK)], check=True)
        rel, tools = PRESETS[args.cookbook]
        workdir, command, action_tools, name = COOKBOOK / rel, [sys.executable, "main.py"], tools, args.cookbook
    else:
        if not (args.workdir and args.cmd):
            ap.error("give --cookbook, or both --workdir and --cmd")
        workdir, command = Path(args.workdir).resolve(), shlex.split(args.cmd)
        if command and command[0] in ("python", "python3"):
            command[0] = sys.executable
        action_tools = args.action_tool
        name = f"{Path(args.workdir).resolve().name}/{Path(command[-1]).name}" if command else Path(args.workdir).name
    name = args.name or name
    env = {"NEBIUS_MODEL": args.model, **dict(kv.split("=", 1) for kv in args.env)}
    spec = AgentSpec(name, workdir, command, action_tools or args.action_tool, env)
    auditor = AgentAuditor(args.upstream, os.environ.get(args.key_env) or os.environ.get("NEBIUS_API_KEY"),
                           pairs=args.pairs)
    where = "in a Token Factory Sandboxes microVM" if args.sandbox else "locally"
    print(f"Auditing {name} {where} ({len(auditor.scenarios)} scenarios, {args.pairs} crash/rerun pairs) ...")
    if args.sandbox:
        from src.sandbox import SandboxRunner

        runner = SandboxRunner(auditor.api_key, os.environ.get("PROJECT_ID") or os.environ.get("NEBIUS_PROJECT_ID"),
                               args.upstream)
        audit = runner.audit(spec, args.pairs)
    else:
        audit = auditor.audit(spec)
    print_report(audit)
    if args.out:
        Path(args.out).write_text(json.dumps(audit, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
