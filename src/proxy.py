"""ResilienceGuard proxy: framework-agnostic fault injection in front of Token Factory.

Point any agent's OpenAI-compatible base URL at this proxy instead of Token Factory. It works
for CrewAI, Agno, LangChain/LangGraph, Google ADK, Pydantic AI, NeMo Agent Toolkit and
anything else that speaks the chat completions API, in any language.

For each ``POST .../chat/completions`` it:

1. checks the kill switch and the circuit breaker (ResilienceGuard),
2. applies the first matching fault rule (error status, delay, mid-stream drop,
   corrupted tool-call arguments, or a canned healthy reply), otherwise
3. forwards the request to the real upstream and streams the reply back,

and records one event per call: fault applied, status, latency, real token usage from the
upstream reply, and the tool calls the model asked for.

Other paths (``/models`` etc.) are forwarded untouched. Control endpoints live under ``/rg/``.

    python -m src.proxy --port 8787 --profile fault_profile.json --guard
    export OPENAI_BASE_URL=http://localhost:8787/v1      # then run your agent unchanged
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import httpx
import openai

from src.config.parser import FaultProfile, FaultRule
from src.guard import CircuitOpenError, ResilienceGuard, RunawayHaltError

DEFAULT_UPSTREAM = os.environ.get("RG_UPSTREAM", "https://api.tokenfactory.nebius.com/v1")
_HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-authorization", "proxy-authenticate", "host", "content-length",
               "content-encoding", "accept-encoding"}


def _status_error(status: int, message: str) -> openai.APIStatusError:
    """An openai error object for an HTTP status, so the guard can classify it."""
    resp = httpx.Response(status, request=httpx.Request("POST", "http://upstream/chat/completions"))
    return openai.APIStatusError(message, response=resp, body=None)


def _error_body(message: str, err_type: str, code: str | None = None) -> bytes:
    return json.dumps({"error": {"message": message, "type": err_type, "code": code}}).encode()


def _tool_calls_from_message(msg: dict) -> list[dict]:
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        out.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments")})
    return out


def _tool_results_from_request(req: dict) -> list[dict]:
    """Tool results the agent is sending back: evidence of which tools actually ran and what they returned."""
    out = []
    for m in req.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content)
            out.append({"tool_call_id": m.get("tool_call_id"), "content": content[:500]})
    return out


class ProxyState:
    """Fault profile, guard, kill switch and event log shared by all request threads."""

    def __init__(self, upstream: str = DEFAULT_UPSTREAM, profile: FaultProfile | None = None,
                 guard: ResilienceGuard | None = None, upstream_api_key: str | None = None,
                 events_path: str | Path | None = None, max_events: int = 10_000) -> None:
        self.upstream = upstream.rstrip("/")
        self.profile = profile or FaultProfile()
        self.guard = guard
        self.upstream_api_key = upstream_api_key
        self.events_path = Path(events_path) if events_path else None
        self.max_events = max_events
        self.killed: str | None = None
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self.calls_seen = 0  # chat requests received
        self.tools_seen: set[str] = set()  # tool names the model has asked for
        self._idle = threading.Condition(self._lock)
        self.http = httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0))

    def begin(self) -> None:
        with self._lock:
            self._in_flight += 1

    def end(self) -> None:
        with self._lock:
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until no chat request is in flight (its event is then recorded)."""
        with self._lock:
            return self._idle.wait_for(lambda: self._in_flight == 0, timeout)

    def record(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                del self.events[: len(self.events) - self.max_events]
            if self.events_path:
                with self.events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")

    def _match(self, exclude: set[str]) -> FaultRule | None:
        # Everything passing through the proxy is traffic under test, so rule targets are ignored.
        if not self.profile.enabled:
            return None
        for rule in self.profile.faults:
            if rule.action in exclude:
                continue
            if rule.after_calls is not None and self.calls_seen <= rule.after_calls:
                continue
            if rule.after_tool_call is not None and rule.after_tool_call not in self.tools_seen:
                continue
            if rule.max_occurrences is not None and rule.current_occurrences >= rule.max_occurrences:
                continue
            if rule.probability < 1.0 and (rule.probability <= 0 or random.random() > rule.probability):
                continue
            return rule
        return None

    def note_tool_calls(self, tool_calls: list[dict]) -> None:
        """Record tool names the model asked for (before the reply reaches the agent)."""
        with self._lock:
            self.tools_seen.update(tc["name"] for tc in tool_calls if tc.get("name"))

    def select_fault(self, stream: bool) -> tuple[FaultRule | None, float]:
        """Pick the rule for this call. Returns (rule, delay_seconds_to_apply_first)."""
        with self._lock:
            self.calls_seen += 1
            exclude = set() if stream else {"stream_drop"}
            rule = self._match(exclude)
            delay = 0.0
            if rule is not None and rule.action == "delay":
                rule.record_occurrence()
                delay = rule.get_delay()
                rule = self._match(exclude | {"delay"})
            elif rule is not None:
                delay = rule.get_delay()
            if rule is not None:
                rule.record_occurrence()
            return rule, delay

    def summary(self) -> dict:
        with self._lock:
            ev = list(self.events)
        chat = [e for e in ev if e["kind"] == "chat"]
        return {
            "upstream": self.upstream,
            "killed": self.killed,
            "calls": len(chat),
            "forwarded": sum(1 for e in chat if e["forwarded"]),
            "faults": sum(1 for e in chat if e["fault"] and e["fault"] != "mock_success"),
            "blocked": sum(1 for e in chat if e["outcome"] in ("circuit_open", "killed")),
            "prompt_tokens": sum((e.get("usage") or {}).get("prompt_tokens", 0) or 0 for e in chat),
            "completion_tokens": sum((e.get("usage") or {}).get("completion_tokens", 0) or 0 for e in chat),
            "guard": self.guard.snapshot() if self.guard else None,
        }


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: ProxyState  # set by make_server

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("RG_PROXY_VERBOSE"):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ plumbing

    def _send(self, status: int, body: bytes, content_type: str = "application/json",
              extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _upstream_headers(self) -> dict:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        if self.state.upstream_api_key:
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = f"Bearer {self.state.upstream_api_key}"
        return headers

    def _upstream_url(self) -> str:
        path = self.path
        if path.startswith("/v1/"):
            path = path[3:]
        return self.state.upstream + path

    def _start_chunked(self, status: int, headers: dict) -> None:
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in _HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def _end_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _abort(self) -> None:
        """Drop the connection mid-body, like a network failure."""
        self.close_connection = True
        try:
            self.wfile.flush()
            self.connection.shutdown(2)
        except OSError:
            pass

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:
        if self.path.startswith("/rg/"):
            return self._control_get()
        self._passthrough("GET", b"")

    def do_POST(self) -> None:
        body = self._read_body()
        if self.path.startswith("/rg/"):
            return self._control_post(body)
        if self.path.rstrip("/").endswith("/chat/completions"):
            return self._chat(body)
        self._passthrough("POST", body)

    def _control_get(self) -> None:
        if self.path.startswith("/rg/events"):
            self.state.wait_idle()
            with self.state._lock:
                payload = {"events": list(self.state.events)}
            return self._send(200, json.dumps(payload).encode())
        if self.path.startswith("/rg/status"):
            self.state.wait_idle()
            return self._send(200, json.dumps(self.state.summary()).encode())
        self._send(404, _error_body("unknown control endpoint", "not_found"))

    def _control_post(self, body: bytes) -> None:
        data = json.loads(body or b"{}")
        if self.path.startswith("/rg/kill"):
            self.state.killed = data.get("reason") or "kill switch activated"
        elif self.path.startswith("/rg/revive"):
            self.state.killed = None
        elif self.path.startswith("/rg/profile"):
            with self.state._lock:
                self.state.profile = FaultProfile.model_validate(data)
        elif self.path.startswith("/rg/reset"):
            with self.state._lock:
                self.state.events.clear()
                self.state.profile.reset()
                self.state.calls_seen = 0
                self.state.tools_seen.clear()
            if self.state.guard:
                self.state.guard.reset()
        else:
            return self._send(404, _error_body("unknown control endpoint", "not_found"))
        self._send(200, json.dumps(self.state.summary()).encode())

    def _passthrough(self, method: str, body: bytes) -> None:
        try:
            r = self.state.http.request(method, self._upstream_url(), headers=self._upstream_headers(),
                                        content=body)
        except httpx.HTTPError as e:
            return self._send(502, _error_body(f"upstream unreachable: {e}", "proxy_error"))
        ctype = r.headers.get("content-type", "application/json")
        self._send(r.status_code, r.content, ctype)

    # ------------------------------------------------------------------ chat

    def _chat(self, body: bytes) -> None:
        st = self.state
        t0 = time.monotonic()
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            req = {}
        stream = bool(req.get("stream"))
        event: dict[str, Any] = {
            "kind": "chat", "id": uuid.uuid4().hex[:12], "ts": time.time(), "model": req.get("model"),
            "stream": stream, "n_messages": len(req.get("messages") or []),
            "has_tools": bool(req.get("tools")), "fault": None, "forwarded": False,
            "status": None, "outcome": None, "usage": None, "tool_calls": [],
            "tool_results": _tool_results_from_request(req),
        }
        st.begin()
        try:
            self._chat_inner(req, body, stream, event)
        finally:
            event["latency_s"] = round(time.monotonic() - t0, 3)
            st.record(event)
            st.end()

    def _guard_fail(self, status: int, message: str) -> None:
        if self.state.guard:
            self.state.guard.record_failure(_status_error(status, message))

    def _guard_ok(self) -> None:
        if self.state.guard:
            self.state.guard.record_success()

    def _chat_inner(self, req: dict, body: bytes, stream: bool, event: dict) -> None:
        st = self.state
        if st.killed:
            event.update(status=403, outcome="killed")
            return self._send(403, _error_body(f"ResilienceGuard kill switch: {st.killed}",
                                               "permission_denied", "rg_killed"),
                              extra={"x-resilienceguard": "killed"})
        if st.guard:
            try:
                st.guard.before_call()
            except (CircuitOpenError, RunawayHaltError) as e:
                event.update(status=503, outcome="circuit_open")
                return self._send(503, _error_body(str(e), "service_unavailable", "rg_circuit_open"),
                                  extra={"x-resilienceguard": "circuit-open", "retry-after": "30"})

        rule, delay = st.select_fault(stream)
        if delay > 0:
            event["delay_s"] = round(delay, 3)
            time.sleep(delay)
        if rule is not None:
            event["fault"] = rule.action

        if rule is not None and rule.action == "error":
            status = rule.status_code or 500
            event.update(status=status, outcome="injected_error")
            self._guard_fail(status, rule.message)
            code = "rate_limit_exceeded" if status == 429 else "rg_injected"
            extra = {"retry-after": "1"} if status == 429 else None
            return self._send(status, _error_body(rule.message, "server_error" if status >= 500 else "rate_limit",
                                                  code), extra=extra)

        if rule is not None and rule.action == "mock_success":
            event.update(status=200, outcome="mock")
            self._guard_ok()
            return self._send_mock(rule, req, stream)

        # Forward to the real upstream.
        event["forwarded"] = True
        try:
            up = st.http.send(
                st.http.build_request("POST", self._upstream_url(), headers=self._upstream_headers(),
                                      content=body),
                stream=True,
            )
        except httpx.HTTPError as e:
            event.update(status=502, outcome="upstream_unreachable", error=str(e)[:300])
            self._guard_fail(502, str(e))
            return self._send(502, _error_body(f"upstream unreachable: {e}", "proxy_error"))

        try:
            event["status"] = up.status_code
            if up.status_code >= 400 or not stream:
                raw = up.read()
                if up.status_code >= 400:
                    event["outcome"] = "upstream_error"
                    event["error"] = raw[:300].decode("utf-8", "replace")
                    self._guard_fail(up.status_code, event["error"])
                    return self._send(up.status_code, raw, up.headers.get("content-type", "application/json"))
                raw = self._inspect_json(raw, event, rule)
                self._guard_ok()
                return self._send(200, raw, up.headers.get("content-type", "application/json"))
            self._relay_stream(up, event, rule)
        finally:
            up.close()

    def _inspect_json(self, raw: bytes, event: dict, rule: FaultRule | None) -> bytes:
        """Record usage/tool calls; apply the 'corrupt' fault to tool-call arguments."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            event["outcome"] = "ok_unparsed"
            return raw
        event["usage"] = data.get("usage")
        msg = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
        event["tool_calls"] = _tool_calls_from_message(msg)
        self.state.note_tool_calls(event["tool_calls"])
        event["outcome"] = "ok"
        if rule is not None and rule.action == "corrupt" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                args = (tc.get("function") or {}).get("arguments") or ""
                tc["function"]["arguments"] = args[: max(1, len(args) - 3)]  # chop the closing brace
            event["outcome"] = "corrupted"
            return json.dumps(data).encode()
        if rule is not None and rule.action == "corrupt":
            event["outcome"] = "ok_no_tool_calls_to_corrupt"
        return raw

    def _relay_stream(self, up: httpx.Response, event: dict, rule: FaultRule | None) -> None:
        drop_after = rule.stream_drop_after_chunks if rule is not None and rule.action == "stream_drop" else None
        self._start_chunked(up.status_code, dict(up.headers))
        sent, buf, tools = 0, b"", {}
        try:
            for piece in up.iter_raw():
                buf += piece
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    self._sniff_sse(ev, event, tools)
                    self._chunk(ev + b"\n\n")
                    sent += 1
                    if drop_after is not None and sent >= drop_after:
                        event["outcome"] = "stream_dropped"
                        event["chunks_sent"] = sent
                        self._guard_fail(502, "stream dropped")
                        return self._abort()
            if buf:
                self._chunk(buf)
            self.state.note_tool_calls(list(tools.values()))
            self._end_chunked()
            event["outcome"] = "ok"
            event["chunks_sent"] = sent
            event["tool_calls"] = list(tools.values())
            self._guard_ok()
        except (httpx.HTTPError, OSError) as e:
            event["outcome"] = "stream_error"
            event["error"] = str(e)[:300]
            self._guard_fail(502, str(e))
            self._abort()

    @staticmethod
    def _sniff_sse(ev: bytes, event: dict, tools: dict) -> None:
        for line in ev.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                event["usage"] = chunk["usage"]
            for choice in chunk.get("choices") or []:
                for tc in (choice.get("delta") or {}).get("tool_calls") or []:
                    slot = tools.setdefault(tc.get("index", 0), {"id": None, "name": None, "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    slot["arguments"] += fn.get("arguments") or ""

    def _send_mock(self, rule: FaultRule, req: dict, stream: bool) -> None:
        now = int(time.time())
        if not stream:
            body = {
                "id": f"rg-mock-{now}", "object": "chat.completion", "created": now,
                "model": req.get("model") or "rg-mock",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": rule.response_content}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            return self._send(200, json.dumps(body).encode())
        self._start_chunked(200, {"Content-Type": "text/event-stream"})
        for i, word in enumerate(rule.response_content.split(" ")):
            chunk = {"id": f"rg-mock-{now}", "object": "chat.completion.chunk", "created": now,
                     "model": req.get("model") or "rg-mock",
                     "choices": [{"index": 0, "delta": {"content": word + " ", **({"role": "assistant"} if i == 0 else {})},
                                  "finish_reason": None}]}
            self._chunk(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self._chunk(b"data: [DONE]\n\n")
        self._end_chunked()


def make_server(state: ProxyState, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    handler = type("BoundProxyHandler", (ProxyHandler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def start_in_thread(state: ProxyState, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start a proxy on a background thread; returns (server, base_url ending in /v1)."""
    server = make_server(state, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}/v1"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ResilienceGuard fault-injection proxy")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    ap.add_argument("--profile", help="fault profile JSON file (default: no faults, pure passthrough)")
    ap.add_argument("--guard", action="store_true", help="enable the circuit breaker")
    ap.add_argument("--events", help="append events as JSON lines to this file")
    ap.add_argument("--inject-key-from-env", metavar="VAR",
                    help="send this env var as the upstream API key (agents can then use a dummy key)")
    args = ap.parse_args(argv)

    profile = FaultProfile.model_validate(json.loads(Path(args.profile).read_text())) if args.profile else FaultProfile()
    guard = ResilienceGuard(halt_after_blocked=None) if args.guard else None
    key = os.environ.get(args.inject_key_from_env) if args.inject_key_from_env else None
    state = ProxyState(args.upstream, profile, guard, key, args.events)
    server = make_server(state, args.host, args.port)
    print(f"ResilienceGuard proxy on http://{args.host}:{args.port}/v1 -> {state.upstream}")
    print(f"  faults: {len(profile.faults)} rule(s)   guard: {'on' if guard else 'off'}   "
          f"status: http://{args.host}:{args.port}/rg/status")
    print(f"  point your agent at it:  export OPENAI_BASE_URL=http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n" + json.dumps(state.summary(), indent=2))


if __name__ == "__main__":
    main()
