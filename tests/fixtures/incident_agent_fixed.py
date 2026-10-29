"""The brittle stand-in with the three fixes ResilienceGuard looks for.

1. Retry budget: max_retries=6 with the SDK's backoff (honours Retry-After), and a timeout.
2. Malformed tool calls are answered with an error message instead of being sent back verbatim.
3. Idempotency: post_status_update checks a ledger keyed by incident_id before acting.
"""

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

INCIDENT = {"incident_id": "INC-42", "service": "checkout-api"}
LEDGER = Path(".action_ledger.json")


@tool
def search_logs(service: str) -> str:
    """Search recent logs for a service."""
    return "p95 latency 1900ms after payment-routing flag rollout"


@tool
def post_status_update(incident_id: str, summary: str) -> str:
    """Post a customer-facing status update for an incident."""
    done = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    key = f"status_update:{incident_id}"
    if key in done:
        return f"skipped: status update for {incident_id} already posted (idempotency key {key})"
    with open("effects.log", "a") as f:
        f.write(json.dumps({"tool": "post_status_update", "incident_id": incident_id}) + "\n")
    done[key] = True
    LEDGER.write_text(json.dumps(done))
    return f"posted update for {incident_id}"


TOOLS = [search_logs, post_status_update]


def main() -> None:
    llm = ChatOpenAI(api_key=os.environ["NEBIUS_API_KEY"], base_url="https://api.tokenfactory.nebius.com/v1/",
                     model=os.getenv("NEBIUS_MODEL", "m"), max_retries=6, timeout=60).bind_tools(TOOLS)
    tools = {t.name: t for t in TOOLS}
    messages = [HumanMessage(content=f"Handle this incident: {json.dumps(INCIDENT)}")]
    for _ in range(8):
        response = llm.invoke(messages)
        if isinstance(response, AIMessage) and response.invalid_tool_calls:
            bad = ", ".join(c.get("name") or "?" for c in response.invalid_tool_calls)
            messages.append(AIMessage(content=response.content or "", tool_calls=response.tool_calls))
            messages.append(HumanMessage(content=f"Your call to {bad} had invalid JSON arguments. Call it again with valid JSON."))
            continue
        messages.append(response)
        if not isinstance(response, AIMessage) or not response.tool_calls:
            print(response.content)
            return
        for call in response.tool_calls:
            messages.append(ToolMessage(content=tools[call["name"]].invoke(call["args"]), tool_call_id=call["id"]))
    raise RuntimeError("did not converge")


if __name__ == "__main__":
    main()
