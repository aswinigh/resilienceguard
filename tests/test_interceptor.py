"""Unit tests for the OpenAI Fault Injection Interceptor."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from openai.resources.chat.completions import AsyncCompletions, Completions

from src.config.parser import FaultProfile, FaultRule
from src.interceptor import (
    OpenAIFaultInterceptor,
    intercept_openai,
    patch_openai,
    unpatch_openai,
)


def _dummy_completion_response():
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "mock response"
    resp.choices = [choice]
    return resp


class TestOpenAIFaultInterceptor:
    """Test suite for OpenAIFaultInterceptor."""

    def test_sync_raises_connection_error(self):
        """Test that patched OpenAI client raises APIConnectionError by default."""
        client = openai.OpenAI(api_key="fake-test-key")

        with intercept_openai() as interceptor:
            with pytest.raises(openai.APIConnectionError) as exc_info:
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                )

            assert "Simulated 500" in str(exc_info.value)
            assert interceptor.get_injected_count() == 1

    def test_async_raises_connection_error(self):
        """Test that patched AsyncOpenAI client raises APIConnectionError."""
        async def _run():
            client = openai.AsyncOpenAI(api_key="fake-test-key")
            with intercept_openai() as interceptor:
                with pytest.raises(openai.APIConnectionError) as exc_info:
                    await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

                assert "Simulated 500" in str(exc_info.value)
                assert interceptor.get_injected_count() == 1

        asyncio.run(_run())

    def test_max_occurrences_limits_faults(self):
        """Test that max_occurrences limits the number of faults injected before passthrough."""
        rule = FaultRule(
            target="openai",
            endpoints=["chat.completions.create"],
            max_occurrences=2,
            error_type="APIConnectionError",
        )
        profile = FaultProfile(faults=[rule])

        mock_resp = _dummy_completion_response()
        with patch.object(Completions, "create", return_value=mock_resp) as mock_create:
            interceptor = OpenAIFaultInterceptor(profile)
            interceptor.patch()

            try:
                client = openai.OpenAI(api_key="fake-test-key")

                # Occurrence 1: should fail
                with pytest.raises(openai.APIConnectionError):
                    client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "1"}],
                    )

                # Occurrence 2: should fail
                with pytest.raises(openai.APIConnectionError):
                    client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "2"}],
                    )

                # Occurrence 3: max_occurrences reached, should pass through to original
                result = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "3"}],
                )

                assert result == mock_resp
                assert mock_create.call_count == 1
                assert interceptor.get_injected_count() == 2
            finally:
                interceptor.unpatch()

    def test_async_max_occurrences_limits_faults(self):
        """Test that max_occurrences limits faults for async calls."""
        async def _run():
            rule = FaultRule(
                target="openai",
                endpoints=["chat.completions.create"],
                max_occurrences=1,
                error_type="APIConnectionError",
            )
            profile = FaultProfile(faults=[rule])

            mock_resp = _dummy_completion_response()
            with patch.object(
                AsyncCompletions, "create", new_callable=AsyncMock, return_value=mock_resp
            ) as mock_acreate:
                interceptor = OpenAIFaultInterceptor(profile)
                interceptor.patch()

                try:
                    client = openai.AsyncOpenAI(api_key="fake-test-key")

                    # Occurrence 1: should fail
                    with pytest.raises(openai.APIConnectionError):
                        await client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": "1"}],
                        )

                    # Occurrence 2: max_occurrences reached, should pass through
                    result = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "2"}],
                    )

                    assert result == mock_resp
                    assert mock_acreate.call_count == 1
                    assert interceptor.get_injected_count() == 1
                finally:
                    interceptor.unpatch()

        asyncio.run(_run())

    def test_context_manager_unpatches_cleanly(self):
        """Test that context manager cleans up and restores original methods."""
        orig_create = Completions.create
        orig_acreate = AsyncCompletions.create

        with intercept_openai() as interceptor:
            assert interceptor.is_patched
            assert Completions.create is not orig_create
            assert AsyncCompletions.create is not orig_acreate

        # After exiting context manager
        assert not interceptor.is_patched
        assert Completions.create is orig_create
        assert AsyncCompletions.create is orig_acreate

    def test_patch_and_unpatch_convenience_functions(self):
        """Test patch_openai and unpatch_openai convenience functions."""
        orig_create = Completions.create
        orig_acreate = AsyncCompletions.create

        interceptor = patch_openai()
        assert interceptor.is_patched
        assert Completions.create is not orig_create
        assert AsyncCompletions.create is not orig_acreate

        unpatch_openai()
        assert not interceptor.is_patched
        assert Completions.create is orig_create
        assert AsyncCompletions.create is orig_acreate

    def test_custom_error_types(self):
        """Test internal server error and rate limit error injection."""
        client = openai.OpenAI(api_key="fake-test-key")

        # Internal Server Error (500)
        profile_500 = FaultProfile(
            faults=[
                FaultRule(
                    error_type="InternalServerError",
                    status_code=500,
                    message="Custom 500 error",
                )
            ]
        )
        with intercept_openai(profile_500):
            with pytest.raises(openai.InternalServerError) as exc_info:
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                )
            assert exc_info.value.status_code == 500
            assert "Custom 500 error" in str(exc_info.value)

        # Rate Limit Error (429)
        profile_429 = FaultProfile(
            faults=[
                FaultRule(
                    error_type="RateLimitError",
                    status_code=429,
                    message="Rate limit hit",
                )
            ]
        )
        with intercept_openai(profile_429):
            with pytest.raises(openai.RateLimitError) as exc_info:
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                )
            assert exc_info.value.status_code == 429
            assert "Rate limit hit" in str(exc_info.value)

        # Unknown Error falls back to APIConnectionError
        profile_fallback = FaultProfile(
            faults=[
                FaultRule(
                    error_type="SomeUnknownError",
                    message="Fallback connection error",
                )
            ]
        )
        with intercept_openai(profile_fallback):
            with pytest.raises(openai.APIConnectionError):
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                )

    def test_delay_seconds(self):
        """Test that configured delay introduces the expected wait time."""
        profile = FaultProfile(
            faults=[
                FaultRule(
                    error_type="APIConnectionError",
                    delay_seconds=0.08,
                )
            ]
        )
        client = openai.OpenAI(api_key="fake-test-key")

        with intercept_openai(profile):
            start = time.perf_counter()
            with pytest.raises(openai.APIConnectionError):
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}],
                )
            elapsed = time.perf_counter() - start
            assert elapsed >= 0.07

    def test_reset_stats(self):
        """Test that reset_stats resets injected count and rule occurrences."""
        rule = FaultRule(max_occurrences=1)
        profile = FaultProfile(faults=[rule])
        client = openai.OpenAI(api_key="fake-test-key")

        mock_resp = _dummy_completion_response()
        with patch.object(Completions, "create", return_value=mock_resp):
            with intercept_openai(profile) as interceptor:
                # 1st call triggers fault
                with pytest.raises(openai.APIConnectionError):
                    client.chat.completions.create(model="gpt-4o", messages=[])
                assert interceptor.get_injected_count() == 1
                assert rule.current_occurrences == 1

                # 2nd call passes through
                client.chat.completions.create(model="gpt-4o", messages=[])
                assert interceptor.get_injected_count() == 1

                # Reset stats
                interceptor.reset_stats()
                assert interceptor.get_injected_count() == 0
                assert rule.current_occurrences == 0

                # 3rd call triggers fault again after reset
                with pytest.raises(openai.APIConnectionError):
                    client.chat.completions.create(model="gpt-4o", messages=[])
                assert interceptor.get_injected_count() == 1
                assert rule.current_occurrences == 1

    def test_dict_and_file_profile_inputs(self, tmp_path: Path):
        """Test instantiating interceptor with dict and file path inputs."""
        # Dict with faults list
        dict_profile = {
            "version": "1.0",
            "enabled": True,
            "faults": [{"error_type": "InternalServerError", "status_code": 500}],
        }
        interceptor1 = OpenAIFaultInterceptor(dict_profile)
        assert len(interceptor1.profile.faults) == 1
        assert interceptor1.profile.faults[0].error_type == "InternalServerError"

        # Dict with single fault rule definition
        single_rule_dict = {"error_type": "RateLimitError", "status_code": 429}
        interceptor2 = OpenAIFaultInterceptor(single_rule_dict)
        assert len(interceptor2.profile.faults) == 1
        assert interceptor2.profile.faults[0].error_type == "RateLimitError"

        # File path
        file_path = tmp_path / "test_profile.json"
        file_path.write_text(
            '{"version": "1.0", "enabled": true, "faults": [{"error_type": "APIConnectionError"}]}',
            encoding="utf-8",
        )
        interceptor3 = OpenAIFaultInterceptor(file_path)
        assert len(interceptor3.profile.faults) == 1
        assert interceptor3.profile.faults[0].error_type == "APIConnectionError"

    def test_disabled_profile_passes_through(self):
        """Test that disabled profile delegates calls to original method without fault."""
        profile = FaultProfile(
            enabled=False,
            faults=[FaultRule(error_type="APIConnectionError")],
        )
        mock_resp = _dummy_completion_response()

        with patch.object(Completions, "create", return_value=mock_resp) as mock_create:
            with intercept_openai(profile) as interceptor:
                client = openai.OpenAI(api_key="fake-test-key")
                res = client.chat.completions.create(model="gpt-4o", messages=[])
                assert res == mock_resp
                assert interceptor.get_injected_count() == 0
                assert mock_create.call_count == 1
