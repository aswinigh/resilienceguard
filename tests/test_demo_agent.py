"""Unit tests for demo LangGraph agent and circuit breaker resilience."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import openai
import pytest

from src.demo_agent import (
    CircuitBreakerTrippedError,
    create_demo_agent_graph,
    message_to_openai_dict,
    run_demo_agent,
    send_email,
    web_search,
)


def _make_mock_response(content: str | None = None, tool_calls: list[dict] | None = None) -> MagicMock:
    """Helper to create a mock OpenAI chat completion response."""
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()

    message.content = content
    if tool_calls is not None:
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc.get("id", "call_1")
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = (
                json.dumps(tc["args"]) if isinstance(tc["args"], dict) else str(tc["args"])
            )
            mock_tcs.append(mock_tc)
        message.tool_calls = mock_tcs
    else:
        message.tool_calls = None

    choice.message = message
    response.choices = [choice]
    return response


def _make_api_connection_error(message: str = "Connection refused") -> openai.APIConnectionError:
    """Create a realistic openai.APIConnectionError."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APIConnectionError(message=message, request=req)


def _make_internal_server_error(message: str = "Simulated 500: Internal Server Error") -> openai.InternalServerError:
    """Create a realistic openai.InternalServerError (HTTP 500)."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(status_code=500, request=req)
    return openai.InternalServerError(
        message=message,
        response=resp,
        body={"error": {"message": message, "type": "server_error", "code": 500}},
    )


# ---------------------------------------------------------------------------
# Tool Tests
# ---------------------------------------------------------------------------

def test_web_search_direct_and_invoke() -> None:
    """Verify web_search tool works when called directly and via invoke."""
    direct_res = web_search("AI agent resilience")
    assert "Simulated search results" in direct_res
    assert "AI agent resilience" in direct_res

    invoke_res = web_search.invoke({"query": "AI agent resilience"})
    assert invoke_res == direct_res
    assert web_search.name == "web_search"
    assert "simulated search" in web_search.description.lower()


def test_send_email_direct_and_invoke() -> None:
    """Verify send_email tool works when called directly and via invoke."""
    direct_res = send_email("alice@example.com", "Test Subject", "Test Body")
    assert "Simulated email sent" in direct_res
    assert "alice@example.com" in direct_res
    assert "Test Subject" in direct_res

    invoke_res = send_email.invoke({
        "to": "alice@example.com",
        "subject": "Test Subject",
        "body": "Test Body",
    })
    assert invoke_res == direct_res
    assert send_email.name == "send_email"
    assert "sending an email" in send_email.description.lower()


# ---------------------------------------------------------------------------
# Message Conversion Tests
# ---------------------------------------------------------------------------

def test_message_to_openai_dict() -> None:
    """Verify conversion of LangChain messages to OpenAI schema."""
    human = HumanMessage(content="Hello world")
    assert message_to_openai_dict(human) == {"role": "user", "content": "Hello world"}

    ai = AIMessage(
        content="Searching...",
        tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "call_99"}],
    )
    converted_ai = message_to_openai_dict(ai)
    assert converted_ai["role"] == "assistant"
    assert converted_ai["content"] == "Searching..."
    assert len(converted_ai["tool_calls"]) == 1
    assert converted_ai["tool_calls"][0]["id"] == "call_99"
    assert json.loads(converted_ai["tool_calls"][0]["function"]["arguments"]) == {"query": "test"}

    tool_msg = ToolMessage(content="Search result", tool_call_id="call_99")
    assert message_to_openai_dict(tool_msg) == {
        "role": "tool",
        "tool_call_id": "call_99",
        "content": "Search result",
    }

    system = SystemMessage(content="System instruction")
    assert message_to_openai_dict(system) == {"role": "system", "content": "System instruction"}


# ---------------------------------------------------------------------------
# Agent Normal Execution Tests
# ---------------------------------------------------------------------------

def test_normal_agent_execution_without_tools() -> None:
    """Verify standard agent execution when LLM responds directly."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(
        content="Resilience in distributed systems is vital."
    )

    state = run_demo_agent("Explain resilience", client=mock_client)

    assert state["status"] == "completed"
    assert state["circuit_open"] is False
    assert state["retry_count"] == 0
    assert state["error_message"] is None
    assert len(state["messages"]) == 2
    assert state["messages"][0].content == "Explain resilience"
    assert state["messages"][1].content == "Resilience in distributed systems is vital."
    mock_client.chat.completions.create.assert_called_once()


def test_normal_agent_execution_with_web_search() -> None:
    """Verify tool execution loop when LLM requests web_search."""
    mock_client = MagicMock()
    # Step 1: LLM calls web_search
    # Step 2: LLM provides final answer based on search
    mock_client.chat.completions.create.side_effect = [
        _make_mock_response(
            content=None,
            tool_calls=[{"name": "web_search", "args": {"query": "circuit breaker pattern"}, "id": "call_1"}],
        ),
        _make_mock_response(
            content="Based on search results, the circuit breaker pattern prevents catastrophic failures."
        ),
    ]

    state = run_demo_agent("Search about circuit breakers", client=mock_client)

    assert state["status"] == "completed"
    assert state["circuit_open"] is False
    assert state["retry_count"] == 0
    assert mock_client.chat.completions.create.call_count == 2

    # Verify message sequence: Human -> AI (tool_call) -> ToolMessage -> AI (final)
    messages = state["messages"]
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], AIMessage)

    assert messages[2].tool_call_id == "call_1"
    assert "Simulated search results for query: 'circuit breaker pattern'" in messages[2].content
    assert "circuit breaker pattern prevents catastrophic failures" in messages[3].content


def test_normal_agent_execution_with_send_email() -> None:
    """Verify tool execution loop when LLM requests send_email."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_mock_response(
            content=None,
            tool_calls=[{
                "name": "send_email",
                "args": {"to": "ops@example.com", "subject": "Alert", "body": "High error rate detected"},
                "id": "call_email_1",
            }],
        ),
        _make_mock_response(content="Notification email has been sent successfully."),
    ]

    state = run_demo_agent("Email ops about the alert", client=mock_client)

    assert state["status"] == "completed"
    assert state["circuit_open"] is False
    assert mock_client.chat.completions.create.call_count == 2

    tool_msg = state["messages"][2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_email_1"
    assert "ops@example.com" in tool_msg.content
    assert "Alert" in tool_msg.content


# ---------------------------------------------------------------------------
# Circuit Breaker & Resilience Tests
# ---------------------------------------------------------------------------

def test_circuit_breaker_trips_on_api_connection_error() -> None:
    """Verify agent retries up to max_retries on APIConnectionError and trips circuit breaker."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_connection_error(
        "Network connection failed: endpoint unreachable"
    )

    max_retries = 3
    state = run_demo_agent("Perform resilient task", client=mock_client, max_retries=max_retries)

    # 1. Verify it retried exactly up to max_retries (3 times)
    assert mock_client.chat.completions.create.call_count == max_retries

    # 2. Verify circuit breaker opened and status is set correctly
    assert state["circuit_open"] is True
    assert state["status"] == "circuit_breaker_tripped"
    assert state["retry_count"] == max_retries
    assert "Network connection failed" in (state["error_message"] or "")

    # 3. Crucially prove execution halted instead of looping infinitely
    # (If it had looped, call_count would exceed 3 or recursion limit would fail)


def test_circuit_breaker_trips_on_simulated_500_error() -> None:
    """Primary test case: prove agent hits simulated 500 error and circuit-breaks instead of looping infinitely."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_internal_server_error(
        "Simulated 500: Triton model server crashed"
    )

    max_retries = 3
    state = run_demo_agent("Test agent against simulated 500", client=mock_client, max_retries=max_retries)

    # Proves the agent halts at max_retries and circuit-breaks
    assert mock_client.chat.completions.create.call_count == max_retries
    assert state["circuit_open"] is True
    assert state["status"] == "circuit_breaker_tripped"
    assert state["retry_count"] == max_retries
    assert "Simulated 500" in (state["error_message"] or "")


def test_circuit_breaker_tripped_error_raised() -> None:
    """Verify CircuitBreakerTrippedError is raised when raise_on_circuit_break is True."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_connection_error("API unreachable")

    with pytest.raises(CircuitBreakerTrippedError) as exc_info:
        run_demo_agent(
            "Test raise",
            client=mock_client,
            max_retries=2,
            raise_on_circuit_break=True,
        )

    assert "Circuit breaker tripped after 2 retries" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 2


def test_transient_failure_recovery() -> None:
    """Verify agent recovers when an API error is followed by a successful response."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_api_connection_error("Temporary network blip"),
        _make_mock_response(content="Recovered after blip!"),
    ]

    state = run_demo_agent("Recover test", client=mock_client, max_retries=3)

    assert mock_client.chat.completions.create.call_count == 2
    assert state["circuit_open"] is False
    assert state["status"] == "completed"
    assert state["retry_count"] == 0
    assert state["messages"][-1].content == "Recovered after blip!"


def test_custom_max_retries_threshold() -> None:
    """Verify circuit breaker trips with custom max_retries setting."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_connection_error("Immediate failure")

    # Set max_retries = 1: Should fail immediately on the 1st attempt
    state = run_demo_agent("Single attempt", client=mock_client, max_retries=1)

    assert mock_client.chat.completions.create.call_count == 1
    assert state["circuit_open"] is True
    assert state["status"] == "circuit_breaker_tripped"
    assert state["retry_count"] == 1


def test_create_demo_agent_graph_default_compilation() -> None:
    """Verify create_demo_agent_graph constructs a runnable compiled graph."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(content="Graph compiled OK")

    graph = create_demo_agent_graph(client=mock_client, max_retries=3)
    assert hasattr(graph, "invoke")

    res = graph.invoke({
        "messages": [HumanMessage(content="Test compile")],
        "retry_count": 0,
        "circuit_open": False,
        "status": "idle",
        "error_message": None,
    })
    assert res["status"] == "completed"
    assert res["circuit_open"] is False


# ---------------------------------------------------------------------------
# End-to-End Interceptor Integration Tests
# ---------------------------------------------------------------------------

def test_e2e_circuit_breaking_with_live_interceptor() -> None:
    """End-to-end test verifying demo_agent circuit-breaks under OpenAIFaultInterceptor fault injection."""
    from src.config.parser import FaultProfile, FaultRule
    from src.interceptor import intercept_openai

    rule = FaultRule(
        error_type="APIConnectionError",
        message="Simulated 500: Fault injected by interceptor",
        probability=1.0,
    )
    profile = FaultProfile(faults=[rule])

    with intercept_openai(profile) as interceptor:
        client = openai.OpenAI(api_key="mock-api-key")
        max_retries = 3
        state = run_demo_agent("E2E fault injection test", client=client, max_retries=max_retries)

        # Proves agent retried exactly 3 times, hit simulated error, and halted cleanly
        assert interceptor.get_injected_count() == max_retries
        assert state["circuit_open"] is True
        assert state["status"] == "circuit_breaker_tripped"
        assert state["retry_count"] == max_retries
        assert "Simulated 500: Fault injected by interceptor" in (state["error_message"] or "")

