"""Run audits inside Nebius Token Factory Sandboxes (microVMs) instead of on this machine.

Follows Nebius's "Compare two fixes" recipe:

1. Runtime image (built once, cached by content hash): Python 3.12 + ResilienceGuard's proxy, audit and
   shim + the packages agents need. Saved as a checkpoint.
2. Agent checkpoint: the runtime + the agent's files, saved once per audit job.
3. Each audit runs inside its own microVM: proxy, agent and all six drills. Each candidate fix is its
   own branch from the agent checkpoint, with only the patched file changed; candidates run in parallel.

The Token Factory key reaches the microVM only as an environment variable of the audit process
(never in files, never preserved in images: preserve_env stays off), and the audit strips it
from the agent's environment, so the untrusted agent only ever sees a dummy key.

    python -m src.sandbox --probe          # check access, build the runtime image, test egress
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Callable

from src.agent_audit import PROJECT_ROOT, TOKEN_FACTORY, AgentSpec

BASE_IMAGE = "docker.io/library/python:3.12-slim"
RUNTIME_PACKAGES = ["openai>=1.40", "httpx>=0.27", "pydantic>=2.7", "langchain-openai>=0.3", "python-dotenv>=1.0"]
RG_FILES = ["src/__init__.py", "src/config/__init__.py", "src/config/parser.py", "src/guard.py", "src/proxy.py",
            "src/agent_audit.py", "src/rg_shim/sitecustomize.py"]
CACHE = PROJECT_ROOT / ".rg_runs" / "sandbox_runtime.json"
SKIP = [".git", ".venv", "venv", "node_modules", "__pycache__", ".*", "*.pyc", "effects.log", "ledger.json",
        ".action_ledger.json"]
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 30_000_000


def agent_files(workdir: Path) -> dict[str, bytes]:
    """The agent's folder as {relative path: bytes}, skipping VCS, envs, caches and secrets (.env)."""
    out, total = {}, 0
    for p in sorted(workdir.rglob("*")):
        rel = p.relative_to(workdir)
        if any(fnmatch.fnmatch(part, pat) for part in rel.parts for pat in SKIP) or not p.is_file():
            continue
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            continue
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"Agent folder is larger than {MAX_TOTAL_BYTES // 1_000_000} MB")
        out[str(rel)] = p.read_bytes()
    return out


def _runtime_key() -> str:
    h = hashlib.sha256(BASE_IMAGE.encode() + json.dumps(RUNTIME_PACKAGES).encode())
    for rel in RG_FILES:
        h.update(rel.encode() + (PROJECT_ROOT / rel).read_bytes())
    return h.hexdigest()[:16]


def _text(x: Any) -> str:
    return x.decode(errors="replace") if isinstance(x, bytes) else str(x or "")


class SandboxRunner:
    """Builds the runtime and agent checkpoints and runs audits in microVMs."""

    def __init__(self, api_key: str, project_id: str | None = None, upstream: str = TOKEN_FACTORY,
                 client_factory: Callable[[], Any] | None = None, cache_path: Path = CACHE) -> None:
        self.api_key = api_key
        self.upstream = upstream
        self.cache_path = cache_path
        if client_factory is None:
            os.environ.setdefault("NEBIUS_API_KEY", api_key)  # the SDK reads these two
            if project_id:
                os.environ.setdefault("NEBIUS_PROJECT_ID", project_id)
            from contree_sdk import ContreeSync

            client_factory = ContreeSync
        self.client_factory = client_factory
        self._agent_images: dict[str, str] = {}

    # ------------------------------------------------------------------ images

    def runtime_image(self, log: Callable[[str], None] = print) -> str:
        key = _runtime_key()
        tag = f"resilienceguard-runtime-{key}"
        cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        if key in cache:
            return cache[key]
        try:  # built before but not cached locally (another machine, or a lost cache): find it by tag
            found = self.client_factory().images.use(tag)
            if getattr(found, "uuid", None):
                return self._remember(cache, key, str(found.uuid))
        except Exception:  # noqa: BLE001 - not built yet
            pass
        log(f"Building the sandbox runtime image ({BASE_IMAGE} + ResilienceGuard + {len(RUNTIME_PACKAGES)} packages)...")
        t0 = time.monotonic()
        client = self.client_factory()
        base = client.images.oci(BASE_IMAGE)
        files = {f"/rg/{rel}": (PROJECT_ROOT / rel).read_bytes() for rel in RG_FILES}
        cmd = "pip install --no-cache-dir -q " + " ".join(shlex.quote(p) for p in RUNTIME_PACKAGES)
        res = base.run(shell=cmd, files=files, disposable=False, timeout=900, tag=tag).wait()
        if res.exit_code != 0 or not res.uuid:
            raise RuntimeError(f"Runtime image build failed (exit {res.exit_code}): {_text(res.stderr)[-800:]}")
        uuid = self._remember(cache, key, str(res.uuid))
        log(f"Runtime image {uuid} ready in {time.monotonic() - t0:.0f}s (cached for next time).")
        return uuid

    def _remember(self, cache: dict, key: str, uuid: str) -> str:
        cache[key] = uuid
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2))
        return uuid

    def agent_image(self, spec: AgentSpec, log: Callable[[str], None] = print) -> str:
        """Runtime + the agent's files, saved once: the checkpoint every drill and every fix branches from."""
        key = str(spec.workdir)
        if key in self._agent_images:
            return self._agent_images[key]
        client = self.client_factory()
        runtime = client.images.use(self.runtime_image(log), strict=True)
        files = {f"/agent/{rel}": data for rel, data in agent_files(spec.workdir).items()}
        # The agent's own dependencies are installed inside the microVM, never on this machine.
        setup = ("pip install --no-cache-dir -q -r /agent/requirements.txt"
                 if "/agent/requirements.txt" in files else "true")
        res = runtime.run(shell=setup, files=files, disposable=False, timeout=900).wait()
        if res.exit_code != 0 or not res.uuid:
            raise RuntimeError(f"Agent checkpoint failed: {_text(res.stderr)[-500:]}")
        uuid = str(res.uuid)  # the SDK returns uuid.UUID objects
        log(f"Agent checkpoint {uuid} ({len(files)} files).")
        self._agent_images[key] = uuid
        return uuid

    # ------------------------------------------------------------------ audit

    def audit_command(self, spec: AgentSpec, pairs: int) -> str:
        entry = " ".join(shlex.quote(c if i else "python") for i, c in enumerate(spec.command))
        parts = ["mkdir -p /out && cd /rg && python -m src.agent_audit", "--workdir /agent",
                 f"--cmd {shlex.quote(entry)}", f"--upstream {shlex.quote(self.upstream)}",
                 "--key-env RG_UPSTREAM_KEY", f"--pairs {int(pairs)}", "--out /out/audit.json"]
        parts += [f"--action-tool {shlex.quote(t)}" for t in spec.action_tools]
        parts += [f"--env {shlex.quote(f'{k}={v}')}" for k, v in spec.env.items() if k != "NEBIUS_MODEL"]
        if spec.env.get("NEBIUS_MODEL"):
            parts.append(f"--model {shlex.quote(spec.env['NEBIUS_MODEL'])}")
        return " ".join(parts)

    def audit(self, spec: AgentSpec, pairs: int = 3, patched: dict[str, str] | None = None,
              label: str | None = None, log: Callable[[str], None] = print) -> dict:
        """Run the full audit in a microVM branched from the agent checkpoint (optionally with patched files)."""
        t0 = time.monotonic()
        agent_uuid = self.agent_image(spec, log)
        client = self.client_factory()
        image = client.images.use(agent_uuid, strict=True)
        files = {f"/agent/{name}": code.encode() for name, code in (patched or {}).items()}
        res = image.run(shell=self.audit_command(spec, pairs), env={"RG_UPSTREAM_KEY": self.api_key},
                        files=files or None, disposable=False, timeout=1200).wait()
        if res.exit_code != 0 or not res.uuid:
            raise RuntimeError(f"Sandboxed audit failed (exit {res.exit_code}): {_text(res.stderr)[-1500:]}")
        audit = json.loads(_text(res.read("/out/audit.json")))
        audit["agent"] = label or spec.name
        audit["sandbox"] = {"runtime_image": self.runtime_image(log), "agent_checkpoint": agent_uuid,
                            "result_image": str(res.uuid), "seconds": round(time.monotonic() - t0, 1),
                            "branched_files": sorted(patched or {})}
        return audit


def _load_env() -> tuple[str | None, str | None]:
    env = PROJECT_ROOT / ".env"
    vals: dict[str, str] = {}
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    key = os.environ.get("API_KEY") or vals.get("API_KEY") or os.environ.get("NEBIUS_API_KEY")
    project = os.environ.get("PROJECT_ID") or vals.get("PROJECT_ID") or os.environ.get("NEBIUS_PROJECT_ID")
    return key, project


def probe() -> dict:
    """Check access, build (or reuse) the runtime image, test egress to Token Factory and branching."""
    key, project = _load_env()
    out: dict[str, Any] = {"key_present": bool(key), "project_present": bool(project)}
    runner = SandboxRunner(key, project)
    t0 = time.monotonic()
    out["runtime_image"] = runner.runtime_image()
    out["runtime_seconds"] = round(time.monotonic() - t0, 1)
    client = runner.client_factory()
    img = client.images.use(out["runtime_image"], strict=True)
    egress = ("python -c \"import httpx,os;r=httpx.get('%s/models',headers={'Authorization':'Bearer '+"
              "os.environ['K']},timeout=20);print(r.status_code)\"" % runner.upstream)
    t1 = time.monotonic()
    res = img.run(shell=egress, env={"K": key}, disposable=True, timeout=120).wait()
    out["token_factory_from_sandbox"] = {"exit": res.exit_code, "stdout": _text(res.stdout).strip()[-200:],
                                         "stderr": _text(res.stderr).strip()[-400:],
                                         "seconds": round(time.monotonic() - t1, 1)}
    t2 = time.monotonic()
    from concurrent.futures import ThreadPoolExecutor

    def branch(i: int) -> str:
        r = client.images.use(out["runtime_image"], strict=True).run(
            shell=f"python -c 'import sys;print(\"branch {i}\", sys.version.split()[0])'", disposable=True,
            timeout=60).wait()
        return _text(r.stdout).strip()

    with ThreadPoolExecutor(max_workers=2) as pool:
        out["parallel_branches"] = list(pool.map(branch, [1, 2]))
    out["branch_seconds"] = round(time.monotonic() - t2, 1)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Token Factory Sandboxes runner for ResilienceGuard")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    if args.probe:
        result = probe()
        (PROJECT_ROOT / "sandbox_probe.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        print("Wrote sandbox_probe.json")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
