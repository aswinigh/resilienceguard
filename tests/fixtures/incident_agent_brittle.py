"""Stand-in incident agent with the cookbook agents' structure (written for ResilienceGuard tests).

Hardcoded Token Factory URL, SDK-default retries, tool loop without error handling, and an
action tool (post_status_update) that records each execution to effects.log in the working dir.
"""

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

INCIDENT = {"incident_id": "INC-42", "service": "checkout-api"}


@tool
def search_logs(service: str) -> str:
    """Search recent logs for a service."""
    return "p95 latency 1900ms after payment-routing flag rollout"


@tool
def post_status_update(incident_id: str, summary: str) -> str:
    """Post a customer-facing status update for an incident."""
    with open("effects.log", "a") as f:
        f.write(json.dumps({"tool": "post_status_update", "incident_id": incident_id}) + "\n")
    return f"posted update for {incident_id}"


TOOLS = [search_logs, post_status_update]


def main() -> None:
    llm = ChatOpenAI(api_key=os.environ["NEBIUS_API_KEY"], base_url="https://api.tokenfactory.nebius.com/v1/",
                     model=os.getenv("NEBIUS_MODEL", "m")).bind_tools(TOOLS)
    tools = {t.name: t for t in TOOLS}
    messages = [HumanMessage(content=f"Handle this incident: {json.dumps(INCIDENT)}")]
    for _ in range(6):
        response = llm.invoke(messages)
        messages.append(response)
        if not isinstance(response, AIMessage) or not response.tool_calls:
            if not (response.content or "").strip():  # same nudge as the cookbook agents
                messages.append(HumanMessage(content="Return the final answer now."))
                continue
            print(response.content)
            return
        for call in response.tool_calls:
            messages.append(ToolMessage(content=tools[call["name"]].invoke(call["args"]), tool_call_id=call["id"]))
    raise RuntimeError("did not converge")


if __name__ == "__main__":
    main()
