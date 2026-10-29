"""One-shot check: can we reach Token Factory, which Nemotron models exist, and does the proxy relay a real call?

Run from the project root:   .venv/bin/python scripts/check_tokenfactory.py
Reads API_KEY from .env. Writes results (no secrets) to rg_check.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import openai  # noqa: E402

from src.proxy import ProxyState, start_in_thread  # noqa: E402


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    load_env()
    key = os.environ.get("API_KEY") or os.environ.get("NEBIUS_API_KEY")
    out: dict = {"key_present": bool(key), "endpoints": {}}
    if not key:
        print("No API_KEY in .env"); return

    base = None
    for url in ("https://api.tokenfactory.nebius.com/v1", "https://api.studio.nebius.ai/v1"):
        try:
            r = httpx.get(f"{url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
            ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
            out["endpoints"][url] = {"status": r.status_code, "n_models": len(ids),
                                     "nvidia_models": [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]}
            if r.status_code == 200 and base is None:
                base = url
        except Exception as e:  # noqa: BLE001
            out["endpoints"][url] = {"error": f"{type(e).__name__}: {e}"[:200]}
        print(url, "->", out["endpoints"][url])

    if base:
        models = out["endpoints"][base]["nvidia_models"]
        model = next((m for m in models if "nemotron" in m.lower()), models[0] if models else None)
        out["chosen_base"], out["chosen_model"] = base, model
        if model:
            state = ProxyState(base, upstream_api_key=key)
            server, proxy_url = start_in_thread(state)
            try:
                client = openai.OpenAI(base_url=proxy_url, api_key="dummy-key-proxy-injects-real-one", max_retries=0)
                resp = client.chat.completions.create(
                    model=model, max_tokens=40,
                    messages=[{"role": "user", "content": "Reply with exactly: ResilienceGuard proxy OK"}])
                out["chat_reply"] = resp.choices[0].message.content
                tool_resp = client.chat.completions.create(
                    model=model, max_tokens=200,
                    messages=[{"role": "user", "content": "Email bob@example.com saying hi. Use the tool."}],
                    tools=[{"type": "function", "function": {
                        "name": "send_email", "description": "Send an email",
                        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                                       "required": ["to", "body"]}}}])
                tcs = tool_resp.choices[0].message.tool_calls or []
                out["tool_calling"] = [{"name": t.function.name, "arguments": t.function.arguments} for t in tcs]
            except Exception as e:  # noqa: BLE001
                out["chat_error"] = f"{type(e).__name__}: {e}"[:400]
            out["proxy_events"] = state.events
            server.shutdown()

    (ROOT / "rg_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "proxy_events"}, indent=2))
    print("\nWrote rg_check.json")


if __name__ == "__main__":
    main()
