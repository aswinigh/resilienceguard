"""Studio server: saved runs, path safety, and live-run request checks (no live run is started)."""

from __future__ import annotations

import json
import threading

import httpx
import pytest

import app


@pytest.fixture(scope="module")
def base():
    server = app.make_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_page_and_saved_runs(base):
    assert "Break it. Fix it. Prove it." in httpx.get(base + "/").text
    runs = httpx.get(base + "/api/runs").json()
    assert runs and all(r["file"].endswith(".json") for r in runs)
    run = httpx.get(f"{base}/api/runs/{runs[0]['file']}").json()
    assert run["before"]["scenarios"] and run["meta"]["title"]


def test_saved_runs_contain_no_machine_paths_or_keys():
    for p in app.RUNS.glob("*.json"):
        text = p.read_text()
        assert "/Users/" not in text and "/home/" not in text
        assert "Bearer " not in text


def test_run_names_cannot_escape_the_runs_folder(base):
    assert httpx.get(base + "/api/runs/..%2F.env").status_code == 404
    assert httpx.get(base + "/api/runs/..%2Fapp.py").status_code == 404


def test_live_runs_only_from_the_studio_page(base):
    body = json.dumps({"token": "nope"})
    r = httpx.post(base + "/api/live", content=body,
                   headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
    assert r.status_code == 403
    r = httpx.post(base + "/api/live", content=body, headers={"Content-Type": "text/plain"})
    assert r.status_code == 403
    r = httpx.post(base + "/api/live", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "Fetch the agent first" in r.json()["error"]


def test_config_reports_limits(base):
    cfg = httpx.get(base + "/api/config").json()
    assert set(cfg) >= {"hosted", "fix_available", "sandbox_available", "jobs_left_today", "presets",
                        "drills", "audit_eta", "fix_eta"}
    assert len(cfg["drills"]) == 6
    assert "incident_response" in cfg["presets"]


def test_hosted_rules(monkeypatch, base):
    """Fix needs the code, fetched code never runs without Sandboxes, commands are validated."""
    monkeypatch.setattr(app, "HOSTED", True)
    monkeypatch.setattr(app, "ACCESS_CODE", "s3cret")
    app.INSPECTED["t1"] = {"label": "acme/bot", "url": "https://github.com/acme/bot", "workdir": "/tmp/x",
                           "command": "python main.py", "preset": None}
    hdr = {"Content-Type": "application/json"}
    post = lambda body: httpx.post(base + "/api/live", content=json.dumps(body), headers=hdr)  # noqa: E731
    fix = lambda body: httpx.post(base + "/api/fix", content=json.dumps(body), headers=hdr)  # noqa: E731
    assert fix({"job": "nope"}).status_code == 400  # the fix builds on a finished audit
    app.JOBS["a1"] = {"kind": "audit", "status": "done", "audit_path": "/tmp/none.json",
                      "req": {"info": app.INSPECTED["t1"]}}
    assert fix({"job": "a1", "code": "wrong"}).status_code == 403
    assert post({"token": "t1", "command": "rm -rf /"}).status_code == 400
    monkeypatch.setattr(app, "sandbox_available", lambda: False)
    assert post({"token": "t1"}).status_code == 503
    # hosted: any Host is fine but the Origin must match it
    r = httpx.post(base + "/api/live", content=json.dumps({"token": "t1"}),
                   headers={**hdr, "Origin": "https://evil.example"})
    assert r.status_code == 403
