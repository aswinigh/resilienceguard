"""Unit tests for ResilienceAuditEngine and diagnostic reporting."""

from src.audit import ResilienceAuditEngine


def test_audit_engine_unprotected_500():
    engine = ResilienceAuditEngine(provider="nebius")
    res = engine.test_unprotected_500_failure()
    assert res["status"] == "VULNERABLE"
    assert res["loop_detected"] is True
    assert res["tokens_wasted"] > 0


def test_audit_engine_protected_500():
    engine = ResilienceAuditEngine(provider="nebius")
    res = engine.test_protected_500_failure(max_retries=3)
    assert res["protected"] is True
    assert res["circuit_open"] is True
    assert res["retries_count"] == 3
    assert res["loop_detected"] is False


def test_audit_engine_stream_drop():
    engine = ResilienceAuditEngine(provider="nebius")
    res = engine.test_stream_drop_failure()
    assert res["status"] == "PASS"
    assert res["stream_recovered"] is True


def test_audit_engine_tool_corruption():
    engine = ResilienceAuditEngine(provider="nebius")
    res = engine.test_tool_corruption_failure()
    assert res["status"] == "PASS"
    assert res["malformed_payload_detected"] is True


def test_full_audit_and_markdown_generation():
    engine = ResilienceAuditEngine(provider="nebius")
    audit = engine.run_full_audit()
    assert audit["unprotected"]["score"] < 50
    assert audit["protected"]["score"] > 90
    assert audit["provider"] == "NEBIUS"

    md = ResilienceAuditEngine.generate_markdown_report(audit)
    assert "# 🛡️ AI Agent Resilience Diagnostic Report" in md
    assert "Executive Scorecard" in md
    assert "Nebius" in md
