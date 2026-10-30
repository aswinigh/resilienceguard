"""Fetch an agent from a GitHub link and work out how to audit it.

Accepts https://github.com/<owner>/<repo>, optionally with /tree/<ref>/<path> pointing at a subfolder.
Lists the repository with one GitHub API call and downloads only the files under the linked folder
(no git, no whole-repository archive), writing them safely (no absolute paths, no "..", no VCS or
environment folders), and returns the agent folder. Nothing is executed here: audits of fetched agents
only ever run inside Token Factory Sandboxes.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

URL_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
                    r"(?:/(?:tree|blob)/([^/]+)(?:/(.*?))?)?/?$")
ACTION_VERBS = re.compile(r"(send|post|publish|create|draft|refund|charge|pay|delete|remove|update|write|"
                          r"submit|book|cancel|transfer|notify|email|page|deploy|merge|close|approve|issue)", re.I)
TOOL_DEF = re.compile(r"@tool\b[^\n]*\n(?:\s*@[^\n]*\n)*\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")


@dataclass
class GitHubRef:
    owner: str
    repo: str
    ref: str = "HEAD"
    path: str = ""

    @property
    def label(self) -> str:
        return f"{self.owner}/{self.repo}" + (f"/{self.path}" if self.path else "")

    @property
    def url(self) -> str:
        base = f"https://github.com/{self.owner}/{self.repo}"
        return base + (f"/tree/{self.ref}/{self.path}" if self.path or self.ref != "HEAD" else "")


@dataclass
class AgentInfo:
    ref: GitHubRef
    workdir: Path
    files: int
    entry: str | None
    command: str | None
    tools: list[str] = field(default_factory=list)
    suggested_action_tools: list[str] = field(default_factory=list)
    has_requirements: bool = False


def parse_github_url(url: str) -> GitHubRef:
    m = URL_RE.match(url.strip())
    if not m:
        raise ValueError("Give a GitHub link like https://github.com/owner/repo or .../tree/main/path/to/agent")
    owner, repo, ref, path = m.groups()
    path = (path or "").strip("/")
    if any(part in ("..", "") for part in PurePosixPath(path).parts if path):
        raise ValueError("Invalid path in the link")
    return GitHubRef(owner, repo, ref or "HEAD", path)


MAX_FILES = 300
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 30_000_000
SKIP_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".github"}


def safe_target(dest: Path, rel: str) -> Path | None:
    """Where a repository path may be written under dest, or None if it would escape or is unwanted."""
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts or any(part in SKIP_PARTS for part in p.parts):
        return None
    target = dest.joinpath(*p.parts)
    if dest.resolve() not in target.resolve().parents:
        return None
    return target


def download_folder(ref: GitHubRef, dest: Path, client: httpx.Client | None = None) -> Path:
    """Download only the files under ref.path: one tree listing via the GitHub API, then raw file downloads.

    Works for a single agent folder inside a large repository (e.g. the Nebius cookbook) without
    downloading the whole repository.
    """
    client = client or httpx.Client(follow_redirects=True, timeout=60)
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    r = client.get(f"https://api.github.com/repos/{ref.owner}/{ref.repo}/git/trees/{ref.ref}?recursive=1",
                   headers=headers)
    if r.status_code == 404:
        raise ValueError("Repository or branch not found (it must be public)")
    if r.status_code in (403, 429):
        raise ValueError("GitHub's rate limit was reached; try again in a few minutes")
    r.raise_for_status()
    prefix = f"{ref.path}/" if ref.path else ""
    blobs = [e for e in r.json().get("tree", []) if e.get("type") == "blob" and e["path"].startswith(prefix)]
    if not blobs:
        raise ValueError(f"No files found at '{ref.path or '/'}' in that repository")
    dest.mkdir(parents=True, exist_ok=True)
    picked, total = [], 0
    for e in blobs:
        target = safe_target(dest, e["path"][len(prefix):])
        size = int(e.get("size") or 0)
        if target is None or size > MAX_FILE_BYTES:
            continue
        total += size
        if len(picked) >= MAX_FILES or total > MAX_TOTAL_BYTES:
            raise ValueError(f"The agent folder is too large (limit {MAX_FILES} files / {MAX_TOTAL_BYTES // 1_000_000} MB);"
                             " link to the agent's own folder")
        picked.append((e["path"], target))

    def get(item: tuple[str, Path]) -> None:
        path, target = item
        raw = client.get(f"https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/{ref.ref}/{quote(path)}")
        raw.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw.content[:MAX_FILE_BYTES])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get, picked))
    if not picked:
        raise ValueError("No usable files found in that folder")
    return dest


def inspect_agent(workdir: Path) -> tuple[str | None, list[str], list[str], bool, int]:
    """Entry file, @tool function names, the ones that look like real-world actions, requirements, file count."""
    py = sorted(p for p in workdir.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts)
    entry = None
    for name in ("main.py", "agent.py", "app.py", "run.py"):
        if (workdir / name).exists():
            entry = name
            break
    if entry is None:
        for p in py:
            if p.parent == workdir and "__main__" in p.read_text(errors="ignore"):
                entry = p.name
                break
    tools: list[str] = []
    for p in py:
        for name in TOOL_DEF.findall(p.read_text(errors="ignore")):
            if name not in tools:
                tools.append(name)
    actions = [t for t in tools if ACTION_VERBS.search(t)]
    files = sum(1 for p in workdir.rglob("*") if p.is_file())
    return entry, tools, actions, (workdir / "requirements.txt").exists(), files


def fetch_agent(url: str, dest: Path, client: httpx.Client | None = None) -> AgentInfo:
    ref = parse_github_url(url)
    workdir = download_folder(ref, dest, client)
    entry, tools, actions, reqs, files = inspect_agent(workdir)
    return AgentInfo(ref, workdir, files, entry, f"python {entry}" if entry else None, tools, actions, reqs)
