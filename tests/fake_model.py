"""A scripted stand-in for a Token Factory model, for testing the audit without network access.

Behaviour, per request:
* If any assistant tool call in the history has arguments that are not valid JSON, reject the
  whole request with 400, as Token Factory does.
* If tools are offered and some have not been called yet in this conversation, call the next one
  (in the order offered), with arguments filled from its schema.
* Otherwise return a final JSON answer.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeModel(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = req.get("messages") or []
        called = []
        for m in messages:
            for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                args = (tc.get("function") or {}).get("arguments") or ""
                try:
                    json.loads(args)
                except json.JSONDecodeError as e:
                    return self._send(400, {"error": {"type": "BadRequestError", "message": str(e), "code": "400"}})
                called.append(tc["function"]["name"])
        usage = {"prompt_tokens": 100 + 20 * len(messages), "completion_tokens": 30, "total_tokens": 0}
        for t in req.get("tools") or []:
            fn = t["function"]
            if fn["name"] not in called:
                props = (fn.get("parameters") or {}).get("properties") or {}
                args = {k: ("INC-42" if "id" in k else f"{k} value") for k in props}
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call_{fn['name']}_{len(messages)}", "type": "function",
                    "function": {"name": fn["name"], "arguments": json.dumps(args)}}]}
                return self._send(200, {"id": "f", "object": "chat.completion", "created": 1, "model": req.get("model"),
                                        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                                        "usage": usage})
        msg = {"role": "assistant", "content": json.dumps({"status": "resolved", "tools_used": called})}
        return self._send(200, {"id": "f", "object": "chat.completion", "created": 1, "model": req.get("model"),
                                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage})


def start_fake_model() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"
