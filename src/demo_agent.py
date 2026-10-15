"""Demo LangGraph Agent with circuit-breaking resilience and tool execution.

Provides dummy tools (web_search, send_email), LangGraph agent execution graph,
and circuit-breaking error handling against APIConnectionError and simulated 500 errors.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from functools import wraps
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import openai

logger = logging.getLogger(__name__)


class CircuitBreakerTrippedError(Exception):
    """Raised when the agent circuit breaker trips due to consecutive API failures exceeding max retries."""

    pass


# ---------------------------------------------------------------------------
# Tool Definitions & Callable Wrappers
# ---------------------------------------------------------------------------

def make_callable_tool(fn: Callable[..., str]) -> Any:
    """Wrap a tool function so it can be called directly or invoked as a LangChain tool."""
    structured = tool(fn)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        return fn(*args, **kwargs)

    wrapper.invoke = structured.invoke
    wrapper.ainvoke = structured.ainvoke
    wrapper.name = structured.name
    wrapper.description = structured.description
    wrapper.args_schema = structured.args_schema
    wrapper.tool = structured
    return wrapper


@make_callable_tool
def web_search(query: str) -> str:
    """Tool returning web search results using Tavily if API key is present, otherwise simulated search results."""
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            import httpx

            resp = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 3},
                timeout=10.0,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                snippets = [
                    f"- {r.get('title', 'Result')}: {r.get('content', '')}"
                    for r in results
                ]
                if snippets:
                    return "\n".join(snippets)
        except Exception as e:
            logger.warning("Tavily search call failed, falling back to simulation: %s", e)

    return f"Simulated search results for query: '{query}'"


@make_callable_tool
def send_email(to: str, subject: str, body: str) -> str:
    """dummy tool simulating sending an email."""
    return f"Simulated email sent to '{to}' with subject '{subject}' and body: '{body}'"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "dummy tool returning simulated search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "dummy tool simulating sending an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line of the email.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Body content of the email.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

TOOL_MAP: Dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "send_email": send_email,
}


# ---------------------------------------------------------------------------
# Agent State & Message Conversion
# ---------------------------------------------------------------------------

class AgentState(dict):
    """LangGraph agent state dictionary."""

    messages: Annotated[List[BaseMessage], add_messages]
    retry_count: int
    circuit_open: bool
    status: str  # "idle", "running", "retrying", "completed", "circuit_breaker_tripped", "error"
    error_message: Optional[str]


def message_to_openai_dict(msg: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain BaseMessage to an OpenAI chat message dictionary."""
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": str(msg.content)}
    elif isinstance(msg, AIMessage):
        d: Dict[str, Any] = {"role": "assistant", "content": str(msg.content or "")}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else str(tc["args"]),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
        return d
    elif isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": str(msg.content),
        }
    elif isinstance(msg, SystemMessage):
        return {"role": "system", "content": str(msg.content)}
    else:
        return {"role": "user", "content": str(msg.content)}


def _parse_choice_response(choice: Any) -> tuple[str, list[dict[str, Any]]]:
    """Extract content and tool calls safely from OpenAI Choice object or dict."""
    if isinstance(choice, dict):
        msg = choice.get("message", {})
    else:
        msg = getattr(choice, "message", None)

    if isinstance(msg, dict):
        content = msg.get("content") or ""
        raw_tools = msg.get("tool_calls") or []
    else:
        content = getattr(msg, "content", None) or ""
        raw_tools = getattr(msg, "tool_calls", None) or []

    parsed_tools: list[dict[str, Any]] = []
    for tc in raw_tools:
        if isinstance(tc, dict):
            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            fn_dict = tc.get("function", {})
            if isinstance(fn_dict, dict):
                fn_name = fn_dict.get("name", "")
                fn_args = fn_dict.get("arguments", "{}")
            else:
                fn_name = getattr(fn_dict, "name", "")
                fn_args = getattr(fn_dict, "arguments", "{}")
        else:
            tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
            fn = getattr(tc, "function", None)
            if fn is not None:
                fn_name = getattr(fn, "name", "")
                fn_args = getattr(fn, "arguments", "{}")
            else:
                fn_name = ""
                fn_args = "{}"

        if isinstance(fn_args, str):
            try:
                fn_args_parsed = json.loads(fn_args) if fn_args else {}
            except Exception:
                fn_args_parsed = {"raw": fn_args}
        elif isinstance(fn_args, dict):
            fn_args_parsed = fn_args
        else:
            fn_args_parsed = {}

        parsed_tools.append({
            "name": fn_name,
            "args": fn_args_parsed,
            "id": tc_id,
        })

    return content, parsed_tools


# ---------------------------------------------------------------------------
# Graph Nodes & Routing
# ---------------------------------------------------------------------------

def step_llm(
    state: AgentState,
    client: openai.OpenAI,
    max_retries: int = 3,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Execute LLM call with APIConnectionError handling and circuit breaker logic."""
    raw_messages = state.get("messages", [])
    openai_messages = [message_to_openai_dict(m) for m in raw_messages]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=TOOLS_SCHEMA,
        )
    except (openai.APIConnectionError, openai.OpenAIError, Exception) as err:
        current_retries = state.get("retry_count", 0) + 1
        logger.warning(
            "OpenAI API call failed (%s/%s retries): %s",
            current_retries,
            max_retries,
            err,
        )
        if current_retries >= max_retries:
            logger.error(
                "Circuit breaker tripped after %s retries: %s",
                current_retries,
                err,
            )
            return {
                "retry_count": current_retries,
                "circuit_open": True,
                "status": "circuit_breaker_tripped",
                "error_message": str(err),
            }
        else:
            return {
                "retry_count": current_retries,
                "circuit_open": False,
                "status": "retrying",
                "error_message": str(err),
            }

    # Successful call: reset retry count and parse response
    if not hasattr(response, "choices") or not response.choices:
        return {
            "messages": [AIMessage(content="")],
            "retry_count": 0,
            "circuit_open": False,
            "status": "completed",
            "error_message": None,
        }

    choice = response.choices[0]
    content, parsed_tool_calls = _parse_choice_response(choice)

    ai_msg = AIMessage(
        content=content,
        tool_calls=parsed_tool_calls if parsed_tool_calls else [],
    )

    return {
        "messages": [ai_msg],
        "retry_count": 0,
        "circuit_open": False,
        "status": "running" if parsed_tool_calls else "completed",
        "error_message": None,
    }


def execute_tools(state: AgentState) -> Dict[str, Any]:
    """Execute requested tool calls and produce ToolMessage responses."""
    messages = state.get("messages", [])
    if not messages:
        return {"status": "completed"}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []

    tool_messages: List[ToolMessage] = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"

        if name in TOOL_MAP:
            try:
                fn = TOOL_MAP[name]
                if isinstance(args, dict):
                    tool_output = fn(**args)
                else:
                    tool_output = fn(args)
            except Exception as e:
                tool_output = f"Error executing tool '{name}': {e}"
        else:
            tool_output = f"Error: Tool '{name}' not found."

        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc_id))

    return {
        "messages": tool_messages,
        "status": "running",
    }


def circuit_breaker_node(state: AgentState) -> Dict[str, Any]:
    """Circuit breaker terminal node executed when max retries are exceeded."""
    return {
        "status": "circuit_breaker_tripped",
        "circuit_open": True,
    }


def route_agent_output(state: AgentState) -> Literal["tools", "agent", "circuit_breaker", "__end__"]:
    """Determine the next step in the agent workflow."""
    if state.get("circuit_open"):
        return "circuit_breaker"

    if state.get("status") == "retrying":
        return "agent"

    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            return "tools"

    return "__end__"


# ---------------------------------------------------------------------------
# Graph Construction & Runner
# ---------------------------------------------------------------------------

def create_demo_agent_graph(
    client: Optional[openai.OpenAI] = None,
    max_retries: int = 3,
    model: str = "gpt-4o",
    provider: Optional[str] = None,
) -> Any:
    """Build and compile the LangGraph StateGraph for the resilient demo agent."""
    if client is None:
        if provider:
            from src.providers import get_sponsor_client

            client, resolved_model = get_sponsor_client(
                provider=provider, model=model if model != "gpt-4o" else None
            )
            model = resolved_model
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "mock-api-key")
            client = openai.OpenAI(api_key=api_key)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        return step_llm(state, client=client, max_retries=max_retries, model=model)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", execute_tools)
    workflow.add_node("circuit_breaker", circuit_breaker_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_agent_output,
        {
            "tools": "tools",
            "agent": "agent",
            "circuit_breaker": "circuit_breaker",
            "__end__": END,
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("circuit_breaker", END)

    return workflow.compile()


def run_demo_agent(
    prompt: Union[str, List[BaseMessage]],
    client: Optional[openai.OpenAI] = None,
    max_retries: int = 3,
    model: str = "gpt-4o",
    raise_on_circuit_break: bool = False,
    provider: Optional[str] = None,
) -> AgentState:
    """Initialize and run the demo agent graph with circuit breaker protection.

    Args:
        prompt: Initial user instruction string or list of BaseMessages.
        client: Optional OpenAI client instance.
        max_retries: Max consecutive retries before tripping circuit breaker.
        model: Model name for chat completions.
        raise_on_circuit_break: If True, raises CircuitBreakerTrippedError when circuit breaker trips.
        provider: Optional sponsor provider name ('nebius', 'nvidia', 'openai').

    Returns:
        The final AgentState after execution.
    """
    if isinstance(prompt, str):
        initial_messages = [HumanMessage(content=prompt)]
    else:
        initial_messages = list(prompt)

    initial_state: AgentState = {
        "messages": initial_messages,
        "retry_count": 0,
        "circuit_open": False,
        "status": "running",
        "error_message": None,
    }

    graph = create_demo_agent_graph(
        client=client, max_retries=max_retries, model=model, provider=provider
    )
    final_state = graph.invoke(initial_state)

    if final_state.get("circuit_open") and raise_on_circuit_break:
        raise CircuitBreakerTrippedError(
            f"Circuit breaker tripped after {final_state.get('retry_count')} retries: "
            f"{final_state.get('error_message')}"
        )

    return final_state
