"""End-to-end tests for the proxy-based agent audit, using a scripted fake model (no network)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.agent_audit import AgentAuditor, AgentSpec, compare
from tests.fake_model import start_fake_model

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def model_url():
    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k)  # local test servers only
    server, url = start_fake_model()
    yield url
    server.shutdown()


def _spec(script: str) -> AgentSpec:
    return AgentSpec(script, FIXTURES, [sys.executable, script], ["post_status_update"], timeout_s=60)


def _by_id(audit: dict) -> dict:
    return {r["id"]: r for r in audit["scenarios"]}


@pytest.fixture(scope="module")
def brittle(model_url):
    return AgentAuditor(model_url, "real-key", pairs=1).audit(_spec("incident_agent_brittle.py"))


@pytest.fixture(scope="module")
def fixed(model_url):
    return AgentAuditor(model_url, "real-key", pairs=1).audit(_spec("incident_agent_fixed.py"))


def test_brittle_agent_fails_the_resilience_scenarios(brittle):
    r = _by_id(brittle)
    assert r["baseline"]["passed"] is True
    assert r["brief_outage"]["passed"] is False and "Crashed after 3 calls" in r["brief_outage"]["verdict"]
    assert r["rate_limit"]["passed"] is False
    assert r["malformed_tool_call"]["passed"] is False
    assert r["sustained_outage"]["passed"] is True  # SDK retries are bounded: it stops after 3 calls
    assert brittle["grade"] == "F"


def test_brittle_agent_repeats_its_action_on_rerun(brittle):
    pair = _by_id(brittle)["crash_after_action"]
    assert pair["triggered"] == 1 and pair["repeated"] == 1 and pair["passed"] is False
    p = pair["pairs"][0]
    assert [a["executed"] for a in p["crash_run"]["actions"]] == [True]
    assert [a["executed"] for a in p["rerun"]["actions"]] == [True]


def test_fixed_agent_passes_everything(fixed):
    r = _by_id(fixed)
    for sid in ("baseline", "brief_outage", "rate_limit", "malformed_tool_call", "sustained_outage"):
        assert r[sid]["passed"] is True, (sid, r[sid]["verdict"])
    pair = r["crash_after_action"]
    assert pair["passed"] is True
    rerun_action = pair["pairs"][0]["rerun"]["actions"][0]
    assert rerun_action["executed"] is False and "skipped" in rerun_action["result"]
    assert fixed["grade"] == "A" and fixed["score"] == 100


def test_compare_lists_what_the_fix_changed(brittle, fixed):
    c = compare(brittle, fixed)
    assert set(c["fixed"]) == {"Brief provider outage", "Rate-limit burst", "Garbled tool call",
                               "Crash after acting, then rerun"}
    assert c["regressed"] == []


def test_real_token_usage_is_counted(brittle):
    assert brittle["tokens"]["prompt"] > 0
