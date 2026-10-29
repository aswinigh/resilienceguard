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
    assert "ResilienceGuard Studio" in httpx.get(base + "/").text
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
    body = json.dumps({"agent": "incident_response"})
    r = httpx.post(base + "/api/live", content=body,
                   headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
    assert r.status_code == 403
    r = httpx.post(base + "/api/live", content=body, headers={"Content-Type": "text/plain"})
    assert r.status_code == 403
    r = httpx.post(base + "/api/live", content=json.dumps({"agent": "nope"}),
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 400
