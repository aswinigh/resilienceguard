"""Tests for the HTTP fault proxy, using the real OpenAI SDK against a fake upstream."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest

from src.config.parser import FaultProfile
from src.guard import ResilienceGuard
from src.proxy import ProxyState, start_in_thread

TOOLS = [{"type": "function", "function": {"name": "send_email", "parameters": {"type": "object", "properties": {}}}}]


class FakeUpstream(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: tool call if tools are sent, SSE when stream=true."""

    protocol_version = "HTTP/1.1"
    seen_auth: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstream.seen_auth.append(self.headers.get("Authorization"))
        usage = {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(5):
                chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                         "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]}
                self._chunk(f"data: {json.dumps(chunk)}\n\n")
            self._chunk("data: " + json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1,
                                                "model": "m", "choices": [], "usage": usage}) + "\n\n")
            self._chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        message = {"role": "assistant", "content": "hello from upstream"}
        if body.get("tools"):
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "send_email", "arguments": '{"to": "a@b.c"}'}}]}
        out = json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": body.get("model"),
                          "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                          "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _chunk(self, s: str):
        data = s.encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def upstream_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _proxy(upstream_url, faults=None, guard=None, key=None):
    state = ProxyState(upstream_url, FaultProfile(faults=faults or []), guard, key)
    server, base = start_in_thread(state)
    return state, server, base


def _client(base, retries=0):
    return openai.OpenAI(base_url=base, api_key="agent-key", max_retries=retries)


MSG = [{"role": "user", "content": "hi"}]


def test_passthrough_records_real_usage(upstream_url):
    state, server, base = _proxy(upstream_url)
    resp = _client(base).chat.completions.create(model="nemotron", messages=MSG)
    assert resp.choices[0].message.content == "hello from upstream"
    state.wait_idle()
    ev = state.events[-1]
    assert ev["forwarded"] and ev["outcome"] == "ok" and ev["usage"]["prompt_tokens"] == 42
    server.shutdown()


def test_injected_500_is_retried_by_the_sdk_and_every_attempt_is_counted(upstream_url):
    state, server, base = _proxy(upstream_url, [{"action": "error", "status_code": 500}])
    with pytest.raises(openai.InternalServerError):
        _client(base, retries=2).chat.completions.create(model="m", messages=MSG)
    state.wait_idle()
    chat = [e for e in state.events if e["kind"] == "chat"]
    assert len(chat) == 3  # the SDK's own 2 retries are visible at the proxy
    assert all(e["outcome"] == "injected_error" and not e["forwarded"] for e in chat)
    server.shutdown()


def test_guard_blocks_calls_after_threshold(upstream_url):
    guard = ResilienceGuard(failure_threshold=3, halt_after_blocked=None)
    state, server, base = _proxy(upstream_url, [{"action": "error", "status_code": 503}], guard=guard)
    client = _client(base)
    for _ in range(6):
        with pytest.raises(openai.APIStatusError):
            client.chat.completions.create(model="m", messages=MSG)
    state.wait_idle()
    outcomes = [e["outcome"] for e in state.events]
    assert outcomes == ["injected_error"] * 3 + ["circuit_open"] * 3
    assert state.summary()["blocked"] == 3
    server.shutdown()


def test_kill_switch_stops_every_call_until_revived(upstream_url):
    state, server, base = _proxy(upstream_url)
    ctl = base[:-3]
    httpx.post(f"{ctl}/rg/kill", json={"reason": "operator stop"})
    with pytest.raises(openai.PermissionDeniedError):
        _client(base).chat.completions.create(model="m", messages=MSG)
    httpx.post(f"{ctl}/rg/revive", json={})
    assert _client(base).chat.completions.create(model="m", messages=MSG).choices[0].message.content
    server.shutdown()


def test_stream_passthrough_and_usage(upstream_url):
    state, server, base = _proxy(upstream_url)
    text = "".join(c.choices[0].delta.content or "" for c in
                   _client(base).chat.completions.create(model="m", messages=MSG, stream=True) if c.choices)
    assert text == "t0 t1 t2 t3 t4 "
    state.wait_idle()
    assert state.events[-1]["usage"]["completion_tokens"] == 7
    server.shutdown()


def test_stream_drop_breaks_the_client_mid_stream(upstream_url):
    state, server, base = _proxy(upstream_url, [{"action": "stream_drop", "stream_drop_after_chunks": 2}])
    got = []
    with pytest.raises(Exception):
        for c in _client(base).chat.completions.create(model="m", messages=MSG, stream=True):
            got.append(c)
    assert len(got) == 2
    state.wait_idle()
    assert state.events[-1]["outcome"] == "stream_dropped"
    server.shutdown()


def test_corrupt_truncates_real_tool_call_arguments(upstream_url):
    state, server, base = _proxy(upstream_url, [{"action": "corrupt", "max_occurrences": 1}])
    resp = _client(base).chat.completions.create(model="m", messages=MSG, tools=TOOLS)
    args = resp.choices[0].message.tool_calls[0].function.arguments
    with pytest.raises(json.JSONDecodeError):
        json.loads(args)
    state.wait_idle()
    assert state.events[-1]["outcome"] == "corrupted"
    assert state.events[-1]["tool_calls"][0]["name"] == "send_email"
    # next call is clean
    resp = _client(base).chat.completions.create(model="m", messages=MSG, tools=TOOLS)
    assert json.loads(resp.choices[0].message.tool_calls[0].function.arguments) == {"to": "a@b.c"}
    server.shutdown()


def test_delay_then_forward(upstream_url):
    state, server, base = _proxy(upstream_url, [{"action": "delay", "delay_seconds": 0.3, "max_occurrences": 1}])
    t = time.monotonic()
    _client(base).chat.completions.create(model="m", messages=MSG)
    assert time.monotonic() - t >= 0.3
    state.wait_idle()
    assert state.events[-1]["forwarded"] and state.events[-1]["delay_s"] == 0.3
    server.shutdown()


def test_429_with_retry_after(upstream_url):
    state, server, base = _proxy(upstream_url, [{"action": "error", "status_code": 429, "max_occurrences": 1}])
    with pytest.raises(openai.RateLimitError):
        _client(base).chat.completions.create(model="m", messages=MSG)
    assert _client(base).chat.completions.create(model="m", messages=MSG)
    server.shutdown()


def test_proxy_can_inject_the_upstream_key(upstream_url):
    FakeUpstream.seen_auth.clear()
    state, server, base = _proxy(upstream_url, key="real-secret")
    _client(base).chat.completions.create(model="m", messages=MSG)
    assert FakeUpstream.seen_auth[-1] == "Bearer real-secret"
    server.shutdown()


def test_mock_success_without_upstream():
    state, server, base = _proxy("http://127.0.0.1:9/v1", [{"action": "mock_success", "response_content": "canned"}])
    assert _client(base).chat.completions.create(model="m", messages=MSG).choices[0].message.content == "canned"
    server.shutdown()


def test_fault_waits_until_the_model_requests_a_given_tool(upstream_url):
    """Models a crash right after a side-effecting tool: calls succeed until send_email is requested."""
    rule = {"action": "error", "status_code": 500, "after_tool_call": "send_email"}
    state, server, base = _proxy(upstream_url, [rule])
    client = _client(base)
    assert client.chat.completions.create(model="m", messages=MSG).choices[0].message.content  # no tools yet
    resp = client.chat.completions.create(model="m", messages=MSG, tools=TOOLS)  # model asks for send_email
    assert resp.choices[0].message.tool_calls[0].function.name == "send_email"
    with pytest.raises(openai.InternalServerError):  # the next call (after the action) fails
        client.chat.completions.create(model="m", messages=MSG, tools=TOOLS)
    state.wait_idle()
    assert [e["outcome"] for e in state.events] == ["ok", "ok", "injected_error"]
    server.shutdown()


def test_fault_after_n_calls(upstream_url):
    state, server, base = _proxy(upstream_url, [{"action": "error", "status_code": 503, "after_calls": 2}])
    client = _client(base)
    client.chat.completions.create(model="m", messages=MSG)
    client.chat.completions.create(model="m", messages=MSG)
    with pytest.raises(openai.APIStatusError):
        client.chat.completions.create(model="m", messages=MSG)
    server.shutdown()
