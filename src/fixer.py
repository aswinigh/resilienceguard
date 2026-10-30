"""Nemotron fix loop: read the agent and its audit, write fixes, prove them, repair until they pass.

Round 1: Nemotron gets the agent's source, small input files from its folder, the audit evidence and
a resilience playbook, and writes several candidate fixes in parallel.
Each candidate goes into a fresh copy of the agent (the original is never touched), is
syntax-checked, must pass a quick normal run, then gets the full audit.
Rounds 2+: every candidate that still fails goes back to Nemotron with its own code, its audit and
the full error output, and Nemotron repairs it. Stops when a candidate passes everything.

    python -m src.fixer --cookbook incident_response --audit audit_incident.json --out fix_incident.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import openai

from src.agent_audit import (COPY_IGNORE, NEMOTRON_SUPER, PRESETS, PROJECT_ROOT, SCENARIOS, TOKEN_FACTORY,
                             AgentAuditor, AgentSpec, _load_env_file, compare, print_report)

RUNS_DIR = PROJECT_ROOT / ".rg_runs"

PLAYBOOK = """Resilience playbook (apply what the audit evidence calls for):

A. Provider errors (500/502/503/429, timeouts): a brief outage or rate-limit burst must not kill the run.
   Give the model client a real retry budget with exponential backoff that honours Retry-After. For
   LangChain's ChatOpenAI or the OpenAI SDK the simplest correct fix is constructor arguments:
   max_retries around 6 and an explicit timeout such as 60 seconds. Keep it bounded so a sustained
   outage still stops within about 10 calls.
B. Garbled tool calls (arguments that are not valid JSON): never send the broken call back to the provider
   inside the conversation history; the provider rejects the whole request with a 400 and every later
   call fails the same way. In LangChain a garbled call is moved to `response.invalid_tool_calls` and
   `response.tool_calls` is EMPTY, so check `invalid_tool_calls` BEFORE treating the reply as a final
   answer. Append a cleaned message instead, e.g. AIMessage(content=response.content or "",
   tool_calls=response.tool_calls), then a HumanMessage asking the model to call the tool again with
   valid JSON. Bound these retries.
C. Repeated real-world actions after a crash and rerun: make each action tool idempotent. Before acting,
   check a small persistent ledger (a JSON file in the agent's directory) for a key built from stable
   business identifiers taken from the actual input data (look at the input files below for the real
   field names). Never use model-written text, and never use Python's built-in hash(), which differs
   between processes; if no ID field exists, use hashlib.sha256 of the canonical JSON input. If the key
   exists, return a message starting with "skipped: already done" instead of acting; otherwise act,
   then record the key.

Constraints: change as little as possible. Keep every import, data model, prompt string, tool signature,
input and output format exactly as it is unless a fix requires touching it. Use only the standard library
and packages the file already imports. Do not add files other than the ledger."""

SYSTEM = """You are a senior engineer fixing reliability bugs in an AI agent. You are given the agent's source
file, its input files, an audit that ran it under injected provider faults, and a resilience playbook.
Make the smallest changes that fix the failures the audit found, without breaking anything else.

Reply in exactly this format:
CHANGES:
- <one plain-language sentence per change, naming the failure it fixes>
CODE:
```python
<the complete fixed file, with every unchanged line copied exactly>
```"""


def context_files(workdir: Path, exclude: str, limit: int = 6, max_chars: int = 1500) -> str:
    """Small input/data files from the agent's folder, so fixes use the real field names."""
    out = []
    for p in sorted(workdir.rglob("*")):
        if len(out) >= limit:
            break
        rel = p.relative_to(workdir)
        if (not p.is_file() or str(rel) == exclude or any(part.startswith(".") for part in rel.parts)
                or p.suffix.lower() not in (".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml")
                or p.stat().st_size > 20_000):
            continue
        out.append(f"--- {rel} ---\n{p.read_text(errors='replace')[:max_chars]}")
    return "\n".join(out) or "(none)"


def audit_evidence(audit: dict, with_traces: bool = False) -> str:
    """Model-readable summary of what failed, optionally with the error output of failed runs."""
    lines = [f"Audit grade {audit['grade']} ({audit['score']}/100). Action tools: {audit['action_tools']}."]
    for r in audit["scenarios"]:
        status = "N/A" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
        lines.append(f"- [{status}] {r['name']}: simulated: {r['simulates']} Result: {r['verdict']}")
        if r["passed"] is False and with_traces:
            for run in r.get("runs", [])[:1]:
                trace = (run.get("stderr_tail") or "").strip().splitlines()[-15:]
                if trace:
                    lines.append("    error output:\n      " + "\n      ".join(trace))
        if r["passed"] is False and r.get("pairs"):
            for p in r["pairs"][:2]:
                for k in ("crash_run", "rerun"):
                    for a in p[k]["actions"]:
                        lines.append(f"    {k} executed {a['tool']}({a['arguments'][:120]}) -> returned: "
                                     f"{(a['result'] or '(no result seen)')[:120]}")
            lines.append("    The rerun performed the action again: its result did not start with "
                         "'skipped: already done', so the idempotency check did not match.")
    return "\n".join(lines)


def build_messages(source: str, filename: str, audit: dict, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{PLAYBOOK}\n\nAUDIT EVIDENCE:\n{audit_evidence(audit)}\n\n"
                                    f"INPUT FILES:\n{context}\n\n"
                                    f"SOURCE FILE {filename}:\n```python\n{source}\n```"},
    ]


def build_repair_messages(original: str, attempt: str, filename: str, attempt_audit: dict,
                          context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{PLAYBOOK}\n\nYour previous fix of {filename} was audited and still fails. "
                                    "Fix the remaining problems, including any bugs your previous change "
                                    "introduced. Keep the parts that worked.\n\n"
                                    f"AUDIT OF YOUR PREVIOUS FIX:\n{audit_evidence(attempt_audit, with_traces=True)}\n\n"
                                    f"INPUT FILES:\n{context}\n\n"
                                    f"ORIGINAL FILE (for reference):\n```python\n{original}\n```\n\n"
                                    f"YOUR PREVIOUS FIX (start from this):\n```python\n{attempt}\n```"},
    ]


def parse_reply(text: str) -> tuple[list[str], str | None]:
    """Split the reply into the change list and the code.

    The code runs from the first fence after "CODE:" to the LAST fence in the reply. Agent source
    often contains ``` itself (e.g. regexes that strip Markdown fences), so stopping at the first
    closing fence would truncate it.
    """
    text = text or ""
    code_part = text.split("CODE:", 1)[1] if "CODE:" in text else text
    code = None
    opening = re.search(r"```[a-zA-Z]*\n", code_part)
    if opening:
        body = code_part[opening.end():]
        end = body.rfind("```")
        code = body[:end] if end != -1 else body
    changes_part = text.split("CODE:")[0]
    changes = [l.strip()[2:].strip() for l in changes_part.splitlines() if l.strip().startswith("- ")]
    return changes, code


def unified_diff(old: str, new: str, filename: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f"a/{filename}", tofile=f"b/{filename}"))


def fully_passing(audit: dict) -> bool:
    return audit["baseline_ok"] and not audit["failed"]


class Fixer:
    def __init__(self, client: Any, model: str = NEMOTRON_SUPER, auditor: AgentAuditor | None = None,
                 candidates: int = 2, rounds: int = 3, max_tokens: int = 16000, sandbox: Any = None) -> None:
        self.client = client
        self.sandbox = sandbox  # a SandboxRunner: each candidate is audited in its own microVM branch
        self.model = model
        self.auditor = auditor or AgentAuditor()
        self.smoke = AgentAuditor(self.auditor.upstream, self.auditor.api_key,
                                  scenarios=[s for s in SCENARIOS if s.id == "baseline"], pairs=1)
        self.candidates = candidates
        self.rounds = rounds
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict], temperature: float) -> dict:
        t0 = time.monotonic()
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   temperature=temperature, max_tokens=self.max_tokens)
        text = resp.choices[0].message.content or ""
        changes, code = parse_reply(text)
        usage = getattr(resp, "usage", None)
        return {"changes": changes, "code": code, "raw": text,
                "gen_seconds": round(time.monotonic() - t0, 1),
                "tokens": {"prompt": getattr(usage, "prompt_tokens", 0) or 0,
                           "completion": getattr(usage, "completion_tokens", 0) or 0}}

    def evaluate(self, spec: AgentSpec, source: str, gen: dict, filename: str, workdir: Path) -> dict:
        """Syntax check, quick normal run, then the full audit."""
        if not gen["code"]:
            return {"valid": False, "problem": "Nemotron's reply had no code block."}
        try:
            compile(gen["code"], filename, "exec")
        except SyntaxError as e:
            return {"valid": False, "problem": f"Syntax error: {e}"}
        diff = unified_diff(source, gen["code"], filename)
        changed = sum(1 for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))
        if self.sandbox is not None:
            try:
                audit = self.sandbox.audit(spec, self.auditor.pairs, patched={filename: gen["code"]},
                                           label=f"{spec.name} ({workdir.name})")
            except Exception as e:  # noqa: BLE001 - a failed sandbox run is reported, not fatal
                return {"valid": False, "problem": f"Sandbox run failed: {str(e)[:500]}"}
            return {"valid": True, "smoke_failed": not audit["baseline_ok"], "audit": audit, "diff": diff,
                    "changed_lines": changed, "sandbox": audit.get("sandbox")}
        shutil.copytree(spec.workdir, workdir, ignore=COPY_IGNORE)
        (workdir / filename).write_text(gen["code"])
        patched = AgentSpec(spec.name, workdir, spec.command, spec.action_tools, spec.env, spec.timeout_s)
        smoke = self.smoke.audit(patched)
        if not smoke["baseline_ok"]:
            shutil.rmtree(workdir, ignore_errors=True)
            return {"valid": True, "smoke_failed": True, "audit": smoke, "diff": diff, "changed_lines": changed,
                    "problem": "Failed the normal run, so the full audit was skipped."}
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(spec.workdir, workdir, ignore=COPY_IGNORE)  # fresh copy: no ledger from the smoke run
        (workdir / filename).write_text(gen["code"])
        return {"valid": True, "smoke_failed": False, "workdir": str(workdir), "audit": self.auditor.audit(patched),
                "diff": diff, "changed_lines": changed}

    def fix(self, spec: AgentSpec, audit: dict, filename: str, run_dir: Path) -> dict:
        source = (spec.workdir / filename).read_text()
        context = context_files(spec.workdir, filename)
        temps = [0.2, 0.6, 0.9, 0.4][: self.candidates]
        lineages: list[list[dict]] = [[] for _ in range(self.candidates)]
        rounds_log = []

        def step(i: int, rnd: int) -> dict:
            prev = lineages[i][-1] if lineages[i] else None
            if prev is None or not prev.get("code"):
                messages = build_messages(source, filename, audit, context)
            else:
                prev_audit = prev.get("audit") or {"grade": "F", "score": 0, "action_tools": spec.action_tools,
                                                   "scenarios": [], "baseline_ok": False, "failed": []}
                if not prev.get("audit"):  # rejected before auditing: report the reason instead
                    prev_audit = {**prev_audit, "scenarios": [{"name": "Code check", "simulates": "-",
                                                               "verdict": prev["problem"], "passed": False}]}
                messages = build_repair_messages(source, prev["code"], filename, prev_audit, context)
            gen = self.generate(messages, temps[i])
            ev = self.evaluate(spec, source, gen, filename, run_dir / f"round{rnd}_candidate{i + 1}")
            return {"candidate": i + 1, "round": rnd, "temperature": temps[i], **gen, **ev}

        best = None
        for rnd in range(1, self.rounds + 1):
            active = [i for i in range(self.candidates)
                      if not (lineages[i] and lineages[i][-1].get("audit") and fully_passing(lineages[i][-1]["audit"]))]
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                results = list(pool.map(lambda i: step(i, rnd), active))
            for r in results:
                lineages[r["candidate"] - 1].append(r)
            rounds_log.append(results)
            done = [r for r in results if r.get("audit") and not r.get("smoke_failed") and fully_passing(r["audit"])]
            if done:
                break

        attempts = [a for lin in lineages for a in lin if a.get("audit") and not a.get("smoke_failed")]
        if attempts:
            best = max(attempts, key=lambda a: (a["audit"]["baseline_ok"], a["audit"]["score"], -a["round"],
                                                -a["changed_lines"]))
        result = {"agent": spec.name, "file": filename, "model": self.model, "before": audit,
                  "rounds": rounds_log, "best": {"round": best["round"], "candidate": best["candidate"]} if best else None}
        if best:
            result["comparison"] = compare(audit, best["audit"])
            result["best_changes"] = best["changes"]
            result["best_diff"] = best["diff"]
            (run_dir / "best.diff").write_text(best["diff"])
            (run_dir / f"best_{filename}").write_text(best["code"])
        return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Have Nemotron fix an audited agent and prove it")
    ap.add_argument("--cookbook", choices=sorted(PRESETS))
    ap.add_argument("--workdir")
    ap.add_argument("--cmd")
    ap.add_argument("--file", default="main.py", help="the agent file Nemotron may change")
    ap.add_argument("--action-tool", action="append", default=[])
    ap.add_argument("--audit", required=True, help="audit JSON from src.agent_audit (the 'before')")
    ap.add_argument("--model", default=NEMOTRON_SUPER, help="model that writes the fix")
    ap.add_argument("--agent-model", default=NEMOTRON_SUPER, help="model the agent runs on during re-audit")
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--key-env", default="API_KEY")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sandbox", action="store_true",
                    help="audit each candidate in its own Token Factory Sandboxes branch")
    ap.add_argument("--name", help="display name for the agent in reports")
    args = ap.parse_args(argv)

    _load_env_file()
    key = os.environ.get(args.key_env) or os.environ.get("NEBIUS_API_KEY")
    if args.cookbook:
        from src.agent_audit import COOKBOOK
        rel, tools = PRESETS[args.cookbook]
        spec = AgentSpec(args.cookbook, COOKBOOK / rel, [sys.executable, "main.py"], tools,
                         {"NEBIUS_MODEL": args.agent_model})
    else:
        import shlex
        cmd = shlex.split(args.cmd)
        if cmd and cmd[0] in ("python", "python3"):
            cmd[0] = sys.executable
        spec = AgentSpec(args.name or Path(args.workdir).name, Path(args.workdir).resolve(), cmd, args.action_tool,
                         {"NEBIUS_MODEL": args.agent_model})
    audit = json.loads(Path(args.audit).read_text())
    run_dir = RUNS_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', spec.name)}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(base_url=TOKEN_FACTORY, api_key=key, max_retries=4, timeout=600)
    sandbox = None
    if args.sandbox:
        from src.sandbox import SandboxRunner

        sandbox = SandboxRunner(key, os.environ.get("PROJECT_ID") or os.environ.get("NEBIUS_PROJECT_ID"))
    fixer = Fixer(client, args.model, AgentAuditor(TOKEN_FACTORY, key, pairs=args.pairs), args.candidates,
                  args.rounds, sandbox=sandbox)
    where = "each in its own Token Factory Sandboxes branch" if sandbox else "locally"
    print(f"Asking {args.model} for {args.candidates} candidate fixes (up to {args.rounds} rounds), audited {where} ...")
    result = fixer.fix(spec, audit, args.file, run_dir)
    for rnd in result["rounds"]:
        for c in rnd:
            head = f"\nRound {c['round']}, candidate {c['candidate']} (written in {c['gen_seconds']}s)"
            if not c.get("audit"):
                print(f"{head}: rejected ({c['problem']})")
                continue
            print(f"{head}, {c['changed_lines']} lines changed:")
            for ch in c["changes"]:
                print(f"  - {ch}")
            if c.get("smoke_failed"):
                run = c["audit"]["scenarios"][0]
                print(f"  Failed the normal run: {run['verdict']}")
            else:
                print_report(c["audit"])
    if result.get("comparison"):
        cmp_ = result["comparison"]
        b = result["best"]
        print(f"\nBest: round {b['round']} candidate {b['candidate']}: {cmp_['grade_before']} ({cmp_['score_before']}) -> "
              f"{cmp_['grade_after']} ({cmp_['score_after']}). Fixed: {', '.join(cmp_['fixed']) or 'nothing'}. "
              f"Regressed: {', '.join(cmp_['regressed']) or 'nothing'}.")
        print(f"Patched file and diff: {run_dir}")
    else:
        print("\nNo candidate passed the normal run.")
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
