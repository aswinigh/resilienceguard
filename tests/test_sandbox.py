"""Sandbox runner logic against a fake Contree client (no network)."""

from __future__ import annotations

import json
import sys
import uuid as uuid_mod
from pathlib import Path

from src.agent_audit import AgentSpec
from src.sandbox import SandboxRunner, agent_files


class FakeResult:
    counter = 0

    def __init__(self, request, files_store):
        FakeResult.counter += 1
        self.request = request
        self.exit_code = 0
        self.stdout, self.stderr = "", ""
        # like the real SDK: uuid.UUID objects, not strings
        self.uuid = None if request["disposable"] else uuid_mod.UUID(int=FakeResult.counter)
        self.files_store = files_store

    def wait(self):
        return self

    def read(self, path):
        return self.files_store[path]


class FakeImage:
    def __init__(self, client, uuid):
        self.client, self.uuid = client, uuid

    def run(self, **kw):
        req = {"disposable": kw.get("disposable", True), **kw, "from": self.uuid}
        self.client.runs.append(req)
        return FakeResult(req, {"/out/audit.json": json.dumps(self.client.audit).encode()})


class FakeClient:
    def __init__(self, audit):
        self.runs, self.oci_calls, self.audit, self.tagged = [], 0, audit, False
        self.images = self

    def oci(self, ref):
        self.oci_calls += 1
        return FakeImage(self, f"oci:{ref}")

    def use(self, uuid, strict=True):
        if str(uuid).startswith("resilienceguard-runtime-") and not self.tagged:
            raise LookupError("no image with that tag")
        return FakeImage(self, uuid)


AUDIT = {"agent": "x", "score": 100, "grade": "A", "baseline_ok": True, "failed": [], "passed": [], "scenarios": []}


def _runner(tmp_path, client):
    return SandboxRunner("real-key", "proj", client_factory=lambda: client, cache_path=tmp_path / "cache.json")


def _agent(tmp_path) -> AgentSpec:
    d = tmp_path / "agent"
    (d / "data").mkdir(parents=True)
    (d / "main.py").write_text("print('hi')")
    (d / "data" / "alert.json").write_text('{"id": 1}')
    (d / ".env").write_text("SECRET=1")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.pyc").write_bytes(b"0")
    return AgentSpec("agent", d, [sys.executable, "main.py"], ["draft_status_update"], {"NEBIUS_MODEL": "nv/super"})


def test_agent_files_skip_secrets_and_caches(tmp_path):
    files = agent_files(_agent(tmp_path).workdir)
    assert sorted(files) == ["data/alert.json", "main.py"]


def test_runtime_image_is_built_once_and_cached(tmp_path):
    client = FakeClient(AUDIT)
    r = _runner(tmp_path, client)
    first = r.runtime_image(log=lambda m: None)
    second = _runner(tmp_path, client).runtime_image(log=lambda m: None)
    assert first == second and client.oci_calls == 1
    build = client.runs[0]
    assert "pip install" in build["shell"] and "/rg/src/proxy.py" in build["files"] and build["disposable"] is False


def test_audit_runs_in_a_branch_with_the_key_only_in_env(tmp_path):
    client = FakeClient(AUDIT)
    r = _runner(tmp_path, client)
    spec = _agent(tmp_path)
    audit = r.audit(spec, pairs=2, patched={"main.py": "print('fixed')"}, log=lambda m: None)
    run = client.runs[-1]
    assert run["env"] == {"RG_UPSTREAM_KEY": "real-key"}
    assert "real-key" not in run["shell"]
    assert "--action-tool draft_status_update" in run["shell"] and "--model nv/super" in run["shell"]
    assert "--cmd 'python main.py'" in run["shell"] and "--pairs 2" in run["shell"]
    assert run["files"] == {"/agent/main.py": b"print('fixed')"}  # only the patched file changes in the branch
    checkpoint = next(x for x in client.runs if x.get("files") and "/agent/main.py" in x["files"] and x["shell"] == "true")
    assert "/agent/.env" not in checkpoint["files"]
    assert audit["sandbox"]["agent_checkpoint"] == run["from"] and audit["sandbox"]["branched_files"] == ["main.py"]


def test_agent_checkpoint_is_shared_by_candidate_branches(tmp_path):
    client = FakeClient(AUDIT)
    r = _runner(tmp_path, client)
    spec = _agent(tmp_path)
    a = r.audit(spec, patched={"main.py": "a"}, log=lambda m: None)
    b = r.audit(spec, patched={"main.py": "b"}, log=lambda m: None)
    assert a["sandbox"]["agent_checkpoint"] == b["sandbox"]["agent_checkpoint"]
    assert sum(1 for x in client.runs if x["shell"] == "true") == 1


def test_ids_are_strings_and_a_lost_cache_finds_the_image_by_tag(tmp_path):
    client = FakeClient(AUDIT)
    r = _runner(tmp_path, client)
    uuid = r.runtime_image(log=lambda m: None)
    assert isinstance(uuid, str)
    json.loads((tmp_path / "cache.json").read_text())  # serialisable
    assert client.runs[0]["tag"].startswith("resilienceguard-runtime-")
    (tmp_path / "cache.json").unlink()
    client.tagged = True
    again = _runner(tmp_path, client).runtime_image(log=lambda m: None)
    assert client.oci_calls == 1 and again  # found by tag, no rebuild
