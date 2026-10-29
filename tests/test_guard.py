"""Tests for ResilienceGuard (src/guard.py) driven by the fault interceptor."""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest
from openai.resources.chat.completions import AsyncCompletions, Completions

from src.guard import (
    CircuitOpenError,
    CircuitState,
    ResilienceGuard,
    RunawayHaltError,
    is_transient_error,
)
from src.interceptor import intercept_openai

MSG = [{"role": "user", "content": "hi"}]
OUTAGE = {"faults": [{"target": "*", "action": "error"}]}


def _client() -> openai.OpenAI:
    return openai.OpenAI(api_key="test", base_url="https://api.studio.nebius.ai/v1")


def _call(client) -> object:
    return client.chat.completions.create(model="m", messages=MSG)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_circuit_opens_after_threshold_and_blocks_without_reaching_upstream():
    guard = ResilienceGuard(failure_threshold=3, halt_after_blocked=None)
    client = guard.wrap(_client())
    with intercept_openai(OUTAGE) as inj:
        for _ in range(3):
            with pytest.raises(openai.APIConnectionError):
                _call(client)
        assert guard.state is CircuitState.OPEN
        for _ in range(5):
            with pytest.raises(CircuitOpenError):
                _call(client)
    assert inj.get_injected_count() == 3  # blocked calls never reached the (faulted) upstream
    snap = guard.snapshot()
    assert snap["upstream_attempts"] == 3 and snap["blocked"] == 5 and snap["trips"] == 1


def test_naive_except_exception_loop_is_halted():
    """The vulnerable-agent pattern: retry forever on any Exception."""
    guard = ResilienceGuard(failure_threshold=3, halt_after_blocked=3)
    client = _client()
    attempts = 0
    with intercept_openai(OUTAGE) as inj, guard.protect():
        with pytest.raises(RunawayHaltError):
            while True:
                attempts += 1
                try:
                    _call(client)
                except Exception:
                    if attempts > 1000:
                        pytest.fail("guard did not halt the loop")
    assert inj.get_injected_count() == 3
    assert attempts == 6  # 3 upstream failures + 3 blocked calls
    assert guard.halted
    # stays halted until reset
    with pytest.raises(RunawayHaltError):
        guard.call(lambda: None)
    guard.reset()
    assert guard.call(lambda: "ok") == "ok"


def test_runaway_halt_is_not_an_exception_subclass():
    assert not issubclass(RunawayHaltError, Exception)
    assert issubclass(CircuitOpenError, openai.APIConnectionError)


def test_transient_failures_below_threshold_recover():
    profile = {"faults": [
        {"target": "*", "action": "error", "max_occurrences": 2},
        {"target": "*", "action": "mock_success", "response_content": "recovered"},
    ]}
    guard = ResilienceGuard(failure_threshold=3)
    client = guard.wrap(_client())
    with intercept_openai(profile):
        for _ in range(2):
            with pytest.raises(openai.APIConnectionError):
                _call(client)
        resp = _call(client)
    assert resp.choices[0].message.content == "recovered"
    assert guard.state is CircuitState.CLOSED
    assert guard.snapshot()["consecutive_failures"] == 0


def test_half_open_probe_success_closes_and_failure_reopens():
    clock = FakeClock()
    guard = ResilienceGuard(failure_threshold=2, recovery_timeout=10, halt_after_blocked=None, clock=clock)
    calls = {"fail": True}

    def upstream():
        if calls["fail"]:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        return "ok"

    for _ in range(2):
        with pytest.raises(openai.APIConnectionError):
            guard.call(upstream)
    assert guard.state is CircuitState.OPEN
    clock.now = 10
    assert guard.state is CircuitState.HALF_OPEN
    with pytest.raises(openai.APIConnectionError):  # probe fails -> reopen immediately
        guard.call(upstream)
    assert guard.state is CircuitState.OPEN
    clock.now = 25
    calls["fail"] = False
    assert guard.call(upstream) == "ok"  # probe succeeds -> closed
    assert guard.state is CircuitState.CLOSED


def test_non_transient_errors_do_not_trip():
    guard = ResilienceGuard(failure_threshold=1)
    bad_request = openai.BadRequestError(
        message="bad", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None
    )
    assert not is_transient_error(bad_request)

    def upstream():
        raise bad_request

    for _ in range(5):
        with pytest.raises(openai.BadRequestError):
            guard.call(upstream)
    assert guard.state is CircuitState.CLOSED


def test_max_calls_budget_halts():
    guard = ResilienceGuard(max_calls=2)
    guard.call(lambda: 1)
    guard.call(lambda: 2)
    with pytest.raises(RunawayHaltError):
        guard.call(lambda: 3)


def test_stream_drop_counts_as_failure():
    profile = {"faults": [{"target": "*", "action": "stream_drop", "stream_drop_after_chunks": 2}]}
    guard = ResilienceGuard(failure_threshold=2, halt_after_blocked=None)
    client = guard.wrap(_client())
    with intercept_openai(profile):
        for _ in range(2):
            stream = client.chat.completions.create(model="m", messages=MSG, stream=True)
            with pytest.raises(openai.APIConnectionError):
                for _chunk in stream:
                    pass
        assert guard.state is CircuitState.OPEN


def test_async_client_is_guarded():
    async def scenario():
        guard = ResilienceGuard(failure_threshold=2, halt_after_blocked=None)
        client = guard.wrap(openai.AsyncOpenAI(api_key="t", base_url="https://api.studio.nebius.ai/v1"))
        with intercept_openai(OUTAGE) as inj:
            for _ in range(2):
                with pytest.raises(openai.APIConnectionError):
                    await client.chat.completions.create(model="m", messages=MSG)
            with pytest.raises(CircuitOpenError):
                await client.chat.completions.create(model="m", messages=MSG)
        return inj.get_injected_count()

    assert asyncio.run(scenario()) == 2


def test_protect_restores_sdk_methods():
    orig, aorig = Completions.create, AsyncCompletions.create
    with ResilienceGuard().protect():
        assert Completions.create is not orig
        assert AsyncCompletions.create is not aorig
    assert Completions.create is orig
    assert AsyncCompletions.create is aorig


def test_wrapped_client_passes_other_attributes_through():
    raw = _client()
    wrapped = ResilienceGuard().wrap(raw)
    assert wrapped.base_url == raw.base_url
    assert wrapped.chat.completions.with_raw_response is not None
