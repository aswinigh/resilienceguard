"""Fix loop tests: a stub 'Nemotron' returns code; the loop must validate, re-audit and pick the best."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent_audit import AgentAuditor, AgentSpec
from src.fixer import Fixer, build_messages, context_files, parse_reply
from tests.fake_model import start_fake_model

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StubLLM:
    """Returns the queued replies in order, shaped like an OpenAI chat completion."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        text = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500))


def _reply(code: str, changes=("Add a retry budget.",)) -> str:
    return "CHANGES:\n" + "".join(f"- {c}\n" for c in changes) + f"CODE:\n```python\n{code}```\n"


@pytest.fixture(scope="module")
def model_url():
    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k)
    server, url = start_fake_model()
    yield url
    server.shutdown()


def test_parse_reply_extracts_changes_and_code():
    changes, code = parse_reply(_reply("print('x')\n", ["Fix A.", "Fix B."]))
    assert changes == ["Fix A.", "Fix B."] and code == "print('x')\n"


def test_parse_reply_keeps_code_that_contains_backticks():
    """Regression: the cookbook agents strip Markdown fences with regexes containing ```."""
    code = 'import re\n\ndef extract(t):\n    t = re.sub(r"^```(?:json)?", "", t)\n    return re.sub(r"```$", "", t)\n'
    changes, parsed = parse_reply(_reply(code))
    assert parsed == code
    compile(parsed, "main.py", "exec")


def test_prompt_contains_source_evidence_and_playbook():
    audit = {"grade": "F", "score": 10, "action_tools": ["post_status_update"], "scenarios": [
        {"name": "Brief provider outage", "simulates": "4 x 500", "verdict": "Crashed after 3 calls", "passed": False}]}
    msgs = build_messages("SOURCE_MARKER", "main.py", audit, "INPUT_MARKER")
    text = msgs[1]["content"]
    assert "SOURCE_MARKER" in text and "Crashed after 3 calls" in text and "idempotent" in text
    assert "INPUT_MARKER" in text and "hash()" in text


def test_context_files_picks_small_data_files(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "alert.json").write_text('{"id": "inc_1"}')
    (tmp_path / "main.py").write_text("print(1)")
    (tmp_path / ".env").write_text("SECRET=1")
    ctx = context_files(tmp_path, "main.py")
    assert '"id": "inc_1"' in ctx and "SECRET" not in ctx and "print(1)" not in ctx


def _setup(model_url):
    spec = AgentSpec("brittle", FIXTURES, [sys.executable, "incident_agent_brittle.py"], ["post_status_update"],
                     timeout_s=60)
    auditor = AgentAuditor(model_url, "k", pairs=1)
    return spec, auditor, auditor.audit(spec)


def test_broken_first_draft_is_repaired_from_its_error_output(model_url, tmp_path):
    brittle_src = (FIXTURES / "incident_agent_brittle.py").read_text()
    fixed_src = (FIXTURES / "incident_agent_fixed.py").read_text()
    broken = fixed_src.replace("import json\n", "", 1)  # crashes the normal run with NameError
    spec, auditor, before = _setup(model_url)
    stub = StubLLM([_reply(broken, ["Retry budget."]), _reply(fixed_src, ["Restore the json import."])])
    result = Fixer(stub, auditor=auditor, candidates=1, rounds=3).fix(spec, before, "incident_agent_brittle.py", tmp_path)

    r1, r2 = result["rounds"][0][0], result["rounds"][1][0]
    assert r1["smoke_failed"] and "NameError" in r1["audit"]["scenarios"][0]["verdict"]
    repair_prompt = stub.calls[1]["messages"][1]["content"]
    assert "YOUR PREVIOUS FIX" in repair_prompt and "NameError" in repair_prompt
    assert r2["audit"]["grade"] == "A" and len(result["rounds"]) == 2  # stops once a candidate passes
    assert result["best"] == {"round": 2, "candidate": 1}
    assert result["comparison"]["score_after"] == 100 and "max_retries=6" in result["best_diff"]
    assert (FIXTURES / "incident_agent_brittle.py").read_text() == brittle_src  # original untouched
    assert (tmp_path / "best.diff").exists()


def test_syntax_error_is_reported_back_for_repair(model_url, tmp_path):
    fixed_src = (FIXTURES / "incident_agent_fixed.py").read_text()
    spec, auditor, before = _setup(model_url)
    stub = StubLLM([_reply("def broken(:\n"), _reply(fixed_src)])
    result = Fixer(stub, auditor=auditor, candidates=1, rounds=2).fix(spec, before, "incident_agent_brittle.py", tmp_path)
    assert "Syntax error" in result["rounds"][0][0]["problem"]
    assert "Syntax error" in stub.calls[1]["messages"][1]["content"]
    assert result["comparison"]["grade_after"] == "A"


def test_fixer_audits_each_candidate_in_a_sandbox_branch(tmp_path):
    class FakeSandbox:
        def __init__(self):
            self.calls = []

        def audit(self, spec, pairs, patched=None, label=None):
            self.calls.append(patched)
            good = "max_retries=6" in next(iter(patched.values()))
            return {"agent": label, "score": 100 if good else 10, "grade": "A" if good else "F", "baseline_ok": True,
                    "failed": [] if good else ["Brief provider outage"], "passed": [], "action_tools": [],
                    "scenarios": [], "sandbox": {"agent_checkpoint": "img-1", "result_image": f"img-{len(self.calls) + 1}"}}

    fixed_src = (FIXTURES / "incident_agent_fixed.py").read_text()
    brittle_src = (FIXTURES / "incident_agent_brittle.py").read_text()
    spec = AgentSpec("brittle", FIXTURES, [sys.executable, "incident_agent_brittle.py"], ["post_status_update"])
    before = {"grade": "F", "score": 10, "action_tools": ["post_status_update"], "scenarios": [], "failed": ["x"],
              "baseline_ok": True}
    sandbox = FakeSandbox()
    stub = StubLLM([_reply(brittle_src), _reply(fixed_src)])
    result = Fixer(stub, auditor=AgentAuditor("http://unused", "k", pairs=1), candidates=2, rounds=1,
                   sandbox=sandbox).fix(spec, before, "incident_agent_brittle.py", tmp_path)
    assert len(sandbox.calls) == 2 and all(list(c) == ["incident_agent_brittle.py"] for c in sandbox.calls)
    assert result["comparison"]["grade_after"] == "A"
    assert all(c.get("sandbox") for c in result["rounds"][0])
    assert not any(tmp_path.iterdir()) or not list(tmp_path.glob("round1_candidate*"))  # nothing run locally
