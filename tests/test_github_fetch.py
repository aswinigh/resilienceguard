"""GitHub agent fetching: URL parsing, folder-only download, path safety, entry and tool detection (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from src.github_fetch import download_folder, fetch_agent, inspect_agent, parse_github_url

REPO_FILES = {
    "README.md": b"top",
    "agents/a/main.py": (b"from langchain_core.tools import tool\n\n@tool\ndef lookup(q: str) -> str:\n    return q\n\n"
                         b"@tool\ndef post_status_update(s: str) -> str:\n    return s\n\nif __name__ == '__main__':\n    pass\n"),
    "agents/a/data/alert.json": b'{"id": "inc_1"}',
    "agents/a/requirements.txt": b"langchain-openai\n",
    "agents/a/.git/config": b"secret",
    "agents/a/__pycache__/x.pyc": b"0",
    "agents/a/../../evil.py": b"escape",
    "agents/b/main.py": b"other agent",
}


def fake_github(files=REPO_FILES, status=200):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "api.github.com":
            if status != 200:
                return httpx.Response(status)
            tree = [{"path": p, "type": "blob", "size": len(b)} for p, b in files.items()]
            return httpx.Response(200, content=json.dumps({"tree": tree}))
        path = str(request.url.path).split("/", 4)[-1]  # /owner/repo/ref/<path>
        from urllib.parse import unquote
        return httpx.Response(200, content=files[unquote(path)])

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_parse_repo_and_subfolder_links():
    r = parse_github_url("https://github.com/nebius/token-factory-cookbook/tree/main/agents/langchain/incident-response-agent")
    assert (r.owner, r.repo, r.ref, r.path) == ("nebius", "token-factory-cookbook", "main", "agents/langchain/incident-response-agent")
    r = parse_github_url("https://github.com/acme/bot.git")
    assert (r.repo, r.ref, r.path) == ("bot", "HEAD", "")
    for bad in ("http://github.com/a/b", "https://gitlab.com/a/b", "https://github.com/a", "https://github.com/a/b/tree/main/../x"):
        with pytest.raises(ValueError):
            parse_github_url(bad)


def test_downloads_only_the_folder_and_blocks_escapes(tmp_path):
    client, seen = fake_github()
    out = download_folder(parse_github_url("https://github.com/acme/repo/tree/main/agents/a"), tmp_path / "x", client)
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["data/alert.json", "main.py", "requirements.txt"]
    assert not (tmp_path / "evil.py").exists()
    assert not any("agents/b" in u or "README" in u for u in seen)  # nothing outside the folder was downloaded


def test_fetch_agent_detects_entry_and_action_tools(tmp_path):
    client, _ = fake_github()
    info = fetch_agent("https://github.com/acme/repo/tree/main/agents/a", tmp_path / "x", client)
    assert info.entry == "main.py" and info.command == "python main.py" and info.has_requirements
    assert info.tools == ["lookup", "post_status_update"] and info.suggested_action_tools == ["post_status_update"]


def test_helpful_errors(tmp_path):
    client, _ = fake_github(status=404)
    with pytest.raises(ValueError, match="not found"):
        download_folder(parse_github_url("https://github.com/acme/nope"), tmp_path / "a", client)
    client, _ = fake_github(status=403)
    with pytest.raises(ValueError, match="rate limit"):
        download_folder(parse_github_url("https://github.com/acme/repo"), tmp_path / "b", client)
    client, _ = fake_github()
    with pytest.raises(ValueError, match="No files"):
        download_folder(parse_github_url("https://github.com/acme/repo/tree/main/missing"), tmp_path / "c", client)


def test_inspect_handles_decorated_async_tools(tmp_path):
    (tmp_path / "agent.py").write_text("@tool(parse_docstring=True)\nasync def send_refund(o: str) -> str:\n    return o\n")
    entry, tools, actions, reqs, files = inspect_agent(tmp_path)
    assert entry == "agent.py" and tools == ["send_refund"] and actions == ["send_refund"] and not reqs
