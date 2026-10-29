"""ResilienceGuard launcher shim: reroute Token Factory traffic through the proxy, no agent changes.

Python imports ``sitecustomize`` at startup when its directory is on PYTHONPATH. If RG_PROXY_URL
is set, requests made with httpx (used by the OpenAI SDK, LangChain, ChatNebius and LiteLLM) to a
Token Factory host are sent to the proxy instead. Everything else is untouched.

    RG_PROXY_URL=http://127.0.0.1:8787 PYTHONPATH=src/rg_shim python agent.py
"""

import os

_proxy = os.environ.get("RG_PROXY_URL", "").rstrip("/")
_hosts = {"api.tokenfactory.nebius.com", "api.studio.nebius.ai"} | {
    h.strip() for h in os.environ.get("RG_INTERCEPT_HOSTS", "").split(",") if h.strip()
}


def _patch(module_name: str) -> None:
    """Patch Client.send / AsyncClient.send of an httpx-compatible module (httpx, httpx2)."""
    import importlib

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return
    target = mod.URL(_proxy)

    def reroute(request):
        if request.url.host in _hosts:
            request.url = request.url.copy_with(scheme=target.scheme, host=target.host, port=target.port)
            request.headers["Host"] = f"{target.host}:{target.port}" if target.port else target.host
        return request

    orig_send, orig_asend = mod.Client.send, mod.AsyncClient.send

    def send(self, request, *args, **kwargs):
        return orig_send(self, reroute(request), *args, **kwargs)

    async def asend(self, request, *args, **kwargs):
        return await orig_asend(self, reroute(request), *args, **kwargs)

    mod.Client.send, mod.AsyncClient.send = send, asend


if _proxy:
    # The OpenAI SDK 3.x uses httpx2; LangChain, LiteLLM and older SDKs use httpx.
    for _name in ("httpx", "httpx2"):
        try:
            _patch(_name)
        except Exception as exc:  # never break the agent because of the shim
            import sys

            print(f"[resilienceguard shim] {_name} not patched: {exc}", file=sys.stderr)
