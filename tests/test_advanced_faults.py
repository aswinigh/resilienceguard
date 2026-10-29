"""Tests for advanced fault injection modalities:
- Provider target filtering (Nebius vs NVIDIA vs OpenAI)
- Streaming drop (mid-stream SSE termination)
- Latency jitter
- Malformed schema / tool corruption
- Tavily tool integration
"""

import pytest
import openai
from src.config.parser import FaultProfile, FaultRule
from src.interceptor.openai_patch import OpenAIFaultInterceptor, intercept_openai


def get_sponsor_client(provider: str):
    """Plain OpenAI clients pointed at each provider's base URL (no network calls are made)."""
    urls = {"nebius": "https://api.tokenfactory.nebius.com/v1", "nvidia": "https://integrate.api.nvidia.com/v1",
            "openai": None}
    kwargs = {"api_key": "test-key", **({"base_url": urls[provider]} if urls[provider] else {})}
    return openai.OpenAI(**kwargs), "nvidia/nemotron-3-super-120b-a12b"


def test_target_provider_filtering_nebius():
    """Verify that a rule targeting 'nebius' only triggers for Nebius clients."""
    rule = FaultRule(
        target="nebius",
        action="error",
        error_type="APIConnectionError",
        message="Simulated Nebius GPU failure",
        endpoints=["chat.completions.create"],
    )
    profile = FaultProfile(faults=[rule])

    nebius_client, model = get_sponsor_client("nebius")
    openai_client, _ = get_sponsor_client("openai")

    with intercept_openai(profile):
        # Nebius client should hit the fault
        with pytest.raises(openai.APIConnectionError, match="Simulated Nebius GPU failure"):
            nebius_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
            )

        # Standard OpenAI client shouldn't match rule target, but since we are offline,
        # it will not raise our simulated Nebius fault (it attempts real or mocked call).
        # We verify that our interceptor does NOT match the nebius rule for openai client:
        assert not rule.matches_target(str(openai_client.base_url))


def test_target_provider_filtering_nvidia():
    """Verify that a rule targeting 'nvidia' matches NVIDIA NIM base_url."""
    rule = FaultRule(target="nvidia", action="error")
    nvidia_client, _ = get_sponsor_client("nvidia")
    assert rule.matches_target(str(nvidia_client.base_url))
    assert not rule.matches_target("https://api.openai.com/v1/")


def test_streaming_drop_fault():
    """Verify that action='stream_drop' yields initial chunks and then raises APIConnectionError."""
    rule = FaultRule(
        target="nebius",
        action="stream_drop",
        error_type="APIConnectionError",
        message="Simulated stream dropout mid-generation",
        stream_drop_after_chunks=3,
    )
    profile = FaultProfile(faults=[rule])

    client, model = get_sponsor_client("nebius")

    with intercept_openai(profile):
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "stream test"}],
            stream=True,
        )

        received_chunks = []
        with pytest.raises(openai.APIConnectionError, match="Simulated stream dropout mid-generation"):
            for chunk in stream:
                received_chunks.append(chunk)

        assert len(received_chunks) == 3


@pytest.mark.anyio
async def test_async_streaming_drop_fault():
    """Verify that async stream drop yields initial chunks and raises error."""
    rule = FaultRule(
        target="nebius",
        action="stream_drop",
        error_type="APIConnectionError",
        message="Simulated async stream dropout",
        stream_drop_after_chunks=2,
    )
    profile = FaultProfile(faults=[rule])

    async_client = openai.AsyncOpenAI(api_key="mock-key", base_url="https://api.studio.nebius.ai/v1")

    with intercept_openai(profile):
        stream = await async_client.chat.completions.create(
            model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            messages=[{"role": "user", "content": "async stream test"}],
            stream=True,
        )

        received = []
        with pytest.raises(openai.APIConnectionError, match="Simulated async stream dropout"):
            async for chunk in stream:
                received.append(chunk)

        assert len(received) == 2


def test_corrupted_response_fault():
    """Verify that action='corrupt' returns a response with malformed JSON arguments."""
    rule = FaultRule(
        target="nebius",
        action="corrupt",
        corrupt_type="malformed_json",
    )
    profile = FaultProfile(faults=[rule])

    client, model = get_sponsor_client("nebius")

    with intercept_openai(profile):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "corrupt test"}],
        )

        tool_calls = response.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        args = tool_calls[0].function.arguments
        # Verify arguments string is malformed JSON (missing closing brace)
        assert args.startswith('{"query":')
        assert not args.endswith("}")


def test_latency_jitter_delay():
    """Verify that get_delay returns values within jitter range."""
    rule = FaultRule(
        action="delay",
        delay_min_seconds=0.1,
        delay_max_seconds=0.5,
    )
    delays = [rule.get_delay() for _ in range(20)]
    for d in delays:
        assert 0.1 <= d <= 0.5
