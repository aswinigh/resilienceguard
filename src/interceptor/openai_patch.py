"""OpenAI Python SDK Fault Interceptor.

Monkey-patches `openai.resources.chat.completions.Completions.create` and
`openai.resources.chat.completions.AsyncCompletions.create` to inject configured
faults (exceptions, delays) for AI resilience testing.
"""

from __future__ import annotations

import asyncio
import functools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import httpx
import openai
from openai.resources.chat.completions import AsyncCompletions, Completions

from src.config.parser import (
    FaultConfigParser,
    FaultProfile,
    FaultRule,
    load_fault_profile,
)


def _build_openai_error(fault: FaultRule) -> Exception:
    """Instantiate appropriate OpenAI SDK exception based on the fault rule."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    err_type = fault.error_type or "APIConnectionError"
    message = fault.message or "Simulated 500: API connection error"
    status_code = fault.status_code if fault.status_code is not None else 500

    if err_type == "APIConnectionError":
        return openai.APIConnectionError(request=req, message=message)
    elif err_type == "InternalServerError":
        resp = httpx.Response(status_code, request=req)
        return openai.InternalServerError(message=message, response=resp, body=None)
    elif err_type == "RateLimitError":
        resp_code = status_code if status_code != 500 else 429
        resp = httpx.Response(resp_code, request=req)
        return openai.RateLimitError(message=message, response=resp, body=None)
    elif err_type == "APITimeoutError":
        return openai.APITimeoutError(request=req)

    # General OpenAI error class lookup
    err_cls = getattr(openai, err_type, None)
    if err_cls is not None and isinstance(err_cls, type) and issubclass(err_cls, Exception):
        if issubclass(err_cls, openai.APIStatusError):
            resp = httpx.Response(status_code, request=req)
            return err_cls(message=message, response=resp, body=None)
        if issubclass(err_cls, openai.APIConnectionError):
            return err_cls(request=req, message=message)
        try:
            return err_cls(message=message)
        except TypeError:
            try:
                return err_cls(request=req)
            except TypeError:
                pass

    # Default fallback as specified in requirements
    return openai.APIConnectionError(request=req, message=message)


def _resolve_profile(
    profile: FaultProfile | FaultRule | dict[str, Any] | str | Path | None,
) -> FaultProfile:
    """Normalize input profile into a FaultProfile instance."""
    if isinstance(profile, FaultProfile):
        return profile
    if isinstance(profile, FaultRule):
        return FaultProfile(faults=[profile])
    if isinstance(profile, (str, Path)):
        return FaultConfigParser.parse_file(profile)
    if isinstance(profile, dict):
        if "rules" in profile and "faults" not in profile:
            data = dict(profile)
            data["faults"] = data.pop("rules")
            return FaultConfigParser.parse_dict(data)
        if "faults" not in profile and any(
            k in profile
            for k in (
                "error_type",
                "status_code",
                "message",
                "probability",
                "action",
                "delay_seconds",
                "target",
                "max_occurrences",
            )
        ):
            return FaultProfile(faults=[FaultRule.model_validate(profile)])
        return FaultConfigParser.parse_dict(profile)
    if profile is None:
        loaded = load_fault_profile(None)
        if not loaded.faults:
            # Default to a standard FaultRule targeting chat.completions.create
            return FaultProfile(faults=[FaultRule()])
        return loaded
    raise TypeError(f"Unsupported profile type: {type(profile).__name__}")


def _extract_target_url(completions_self: Any) -> str | None:
    """Extract the base URL from the underlying client instance if available."""
    client = getattr(completions_self, "_client", None)
    if client is not None:
        base_url = getattr(client, "base_url", None)
        if base_url is not None:
            return str(base_url)
    return None


def _build_corrupted_response(fault: FaultRule) -> Any:
    """Create a synthetic ChatCompletion response with malformed/corrupted tool call arguments."""
    from openai.types.chat import ChatCompletion

    corrupt_type = getattr(fault, "corrupt_type", "malformed_json")
    if corrupt_type == "malformed_json":
        args = '{"query": "corrupted_query_missing_brace'
    elif corrupt_type == "invalid_type":
        args = '{"query": 123456}'
    else:
        args = "invalid_non_json_string"

    return ChatCompletion(
        id=f"fault-corrupt-{int(time.time())}",
        choices=[{
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_corrupted",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": args,
                    },
                }],
            },
        }],
        created=int(time.time()),
        model="fault-injector-corrupt",
        object="chat.completion",
    )


def _estimate_prompt_tokens(kwargs: dict[str, Any]) -> int:
    """Rough request size in tokens (~4 chars/token over messages + tool schemas)."""
    import json

    payload = {"messages": kwargs.get("messages", []), "tools": kwargs.get("tools", [])}
    try:
        return max(1, len(json.dumps(payload, default=str)) // 4)
    except Exception:
        return 1


def _build_mock_response(fault: FaultRule, stream: bool) -> Any:
    """Canned successful completion standing in for a healthy upstream."""
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    now = int(time.time())
    if stream:
        def _gen() -> Iterator[Any]:
            for i, word in enumerate(fault.response_content.split(" ")):
                yield ChatCompletionChunk(
                    id=f"mock-{now}",
                    choices=[{
                        "delta": {"content": word + " ", "role": "assistant" if i == 0 else None},
                        "finish_reason": None,
                        "index": 0,
                    }],
                    created=now,
                    model="fault-injector-mock",
                    object="chat.completion.chunk",
                )
        return _gen()
    return ChatCompletion(
        id=f"mock-{now}",
        choices=[{
            "finish_reason": "stop",
            "index": 0,
            "message": {"role": "assistant", "content": fault.response_content},
        }],
        created=now,
        model="fault-injector-mock",
        object="chat.completion",
    )


def _wrap_stream_drop(orig_stream: Any, fault: FaultRule) -> Iterator[Any]:
    """Wrap a synchronous stream generator and drop/raise mid-stream after N chunks."""
    from openai.types.chat import ChatCompletionChunk

    threshold = getattr(fault, "stream_drop_after_chunks", 2)
    count = 0
    if orig_stream is not None:
        try:
            for chunk in orig_stream:
                yield chunk
                count += 1
                if count >= threshold:
                    raise _build_openai_error(fault)
            return
        except Exception as e:
            if isinstance(e, openai.OpenAIError):
                raise

    # Synthetic stream drop fallback for offline/mock clients
    for i in range(threshold):
        yield ChatCompletionChunk(
            id=f"chunk-{i}",
            choices=[{
                "delta": {
                    "content": f" token_{i}",
                    "role": "assistant" if i == 0 else None,
                },
                "finish_reason": None,
                "index": 0,
            }],
            created=int(time.time()),
            model="fault-injector-stream",
            object="chat.completion.chunk",
        )
    raise _build_openai_error(fault)


async def _wrap_async_stream_drop(orig_stream: Any, fault: FaultRule) -> Any:
    """Wrap an asynchronous stream generator and drop/raise mid-stream after N chunks."""
    from openai.types.chat import ChatCompletionChunk

    threshold = getattr(fault, "stream_drop_after_chunks", 2)
    count = 0
    if orig_stream is not None:
        try:
            async for chunk in orig_stream:
                yield chunk
                count += 1
                if count >= threshold:
                    raise _build_openai_error(fault)
            return
        except Exception as e:
            if isinstance(e, openai.OpenAIError):
                raise

    for i in range(threshold):
        yield ChatCompletionChunk(
            id=f"chunk-{i}",
            choices=[{
                "delta": {
                    "content": f" token_{i}",
                    "role": "assistant" if i == 0 else None,
                },
                "finish_reason": None,
                "index": 0,
            }],
            created=int(time.time()),
            model="fault-injector-stream",
            object="chat.completion.chunk",
        )
    raise _build_openai_error(fault)


async def _async_iter(items: Any) -> Any:
    for item in items:
        yield item


class OpenAIFaultInterceptor:
    """Intercepts OpenAI chat completions to inject simulated faults."""

    def __init__(
        self,
        profile: FaultProfile | FaultRule | dict[str, Any] | str | Path | None = None,
        on_event: Any = None,
    ) -> None:
        self.profile: FaultProfile = _resolve_profile(profile)
        self.on_event = on_event  # optional callback(event: str, info: dict)
        self._injected_count: int = 0
        self._mock_count: int = 0
        self._is_patched: bool = False
        self._original_create: Any = None
        self._original_acreate: Any = None

    @property
    def is_patched(self) -> bool:
        """Whether the interceptor is currently active."""
        return self._is_patched

    def patch(self) -> OpenAIFaultInterceptor:
        """Patch OpenAI SDK methods to intercept calls."""
        if self._is_patched:
            return self

        self._original_create = Completions.create
        self._original_acreate = AsyncCompletions.create

        interceptor = self
        orig_create = self._original_create
        orig_acreate = self._original_acreate

        @functools.wraps(orig_create)
        def intercepted_create(completions_self: Any, *args: Any, **kwargs: Any) -> Any:
            target_url = _extract_target_url(completions_self)
            fault = interceptor._select(target_url, kwargs)
            if fault is not None and fault.action == "delay":
                delay = fault.get_delay()
                if delay > 0:
                    time.sleep(delay)
                fault = interceptor._select(target_url, kwargs, after_delay=True)
            elif fault is not None and fault.get_delay() > 0:
                time.sleep(fault.get_delay())  # optional delay before the fault itself
            if fault is None:
                return orig_create(completions_self, *args, **kwargs)
            return interceptor._apply(fault, kwargs)

        @functools.wraps(orig_acreate)
        async def intercepted_acreate(
            async_completions_self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            target_url = _extract_target_url(async_completions_self)
            fault = interceptor._select(target_url, kwargs)
            if fault is not None and fault.action == "delay":
                delay = fault.get_delay()
                if delay > 0:
                    await asyncio.sleep(delay)
                fault = interceptor._select(target_url, kwargs, after_delay=True)
            elif fault is not None and fault.get_delay() > 0:
                await asyncio.sleep(fault.get_delay())
            if fault is None:
                return await orig_acreate(async_completions_self, *args, **kwargs)
            result = interceptor._apply(fault, kwargs)
            if fault.action == "stream_drop":
                return _wrap_async_stream_drop(None, fault)
            if fault.action == "mock_success" and kwargs.get("stream"):
                return _async_iter(result)
            return result

        Completions.create = intercepted_create
        AsyncCompletions.create = intercepted_acreate
        self._is_patched = True
        return self

    def start(self) -> OpenAIFaultInterceptor:
        """Alias for patch()."""
        return self.patch()

    def unpatch(self) -> None:
        """Restore original OpenAI SDK methods."""
        if not self._is_patched:
            return

        if self._original_create is not None:
            Completions.create = self._original_create
        if self._original_acreate is not None:
            AsyncCompletions.create = self._original_acreate

        self._original_create = None
        self._original_acreate = None
        self._is_patched = False

    def stop(self) -> None:
        """Alias for unpatch()."""
        self.unpatch()

    def reset_stats(self) -> None:
        """Reset injected fault counts and rule occurrence counters."""
        self._injected_count = 0
        self._mock_count = 0
        if self.profile is not None:
            self.profile.reset()

    def get_injected_count(self) -> int:
        """Return the number of faults injected since instantiation or last reset."""
        return self._injected_count

    def get_mock_count(self) -> int:
        """Number of calls answered with a canned healthy response (action='mock_success')."""
        return self._mock_count

    def _emit(self, event: str, **info: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, info)
            except Exception:
                pass

    def _select(
        self, target_url: str | None, kwargs: dict[str, Any], after_delay: bool = False
    ) -> FaultRule | None:
        """Pick the rule for this call and record it.

        stream_drop rules only apply to streaming calls; after a delay rule fires, matching
        continues with the non-delay rules so latency can be combined with a response.
        """
        exclude: set[str] = set()
        if not kwargs.get("stream"):
            exclude.add("stream_drop")
        if after_delay:
            exclude.add("delay")
        fault = self._match_fault("chat.completions.create", target_or_url=target_url, exclude_actions=exclude)
        if fault is None:
            return None
        fault.record_occurrence()
        tokens = _estimate_prompt_tokens(kwargs)
        if fault.action == "mock_success":
            self._mock_count += 1
            self._emit("mock_response", prompt_tokens_est=tokens)
        else:
            self._injected_count += 1
            self._emit("fault", action=fault.action, prompt_tokens_est=tokens)
        return fault

    def _apply(self, fault: FaultRule, kwargs: dict[str, Any]) -> Any:
        """Produce the faulted result (or raise) for a non-delay rule (sync path)."""
        if fault.action == "mock_success":
            return _build_mock_response(fault, stream=bool(kwargs.get("stream")))
        if fault.action == "stream_drop":
            return _wrap_stream_drop(None, fault)
        if fault.action == "corrupt":
            return _build_corrupted_response(fault)
        raise _build_openai_error(fault)

    def _match_fault(
        self,
        endpoint: str,
        target_or_url: str | None = None,
        exclude_actions: set[str] | frozenset[str] = frozenset(),
    ) -> FaultRule | None:
        """Find matching fault rule for endpoint and target."""
        if not self.profile or not self.profile.enabled:
            return None
        for ep in (endpoint, f"openai.{endpoint}", "chat.completions"):
            rule = self.profile.get_matching_fault(
                ep, target_or_url=target_or_url, exclude_actions=exclude_actions
            )
            if rule is not None:
                return rule
        return None

    def __enter__(self) -> OpenAIFaultInterceptor:
        self.patch()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unpatch()


_active_interceptor: OpenAIFaultInterceptor | None = None


def patch_openai(
    profile: FaultProfile | FaultRule | dict[str, Any] | str | Path | None = None,
) -> OpenAIFaultInterceptor:
    """Globally patch OpenAI SDK methods with the given fault profile."""
    global _active_interceptor
    if _active_interceptor is not None and _active_interceptor.is_patched:
        _active_interceptor.unpatch()
    _active_interceptor = OpenAIFaultInterceptor(profile)
    _active_interceptor.patch()
    return _active_interceptor


def unpatch_openai() -> None:
    """Globally unpatch OpenAI SDK methods."""
    global _active_interceptor
    if _active_interceptor is not None:
        _active_interceptor.unpatch()
        _active_interceptor = None


@contextmanager
def intercept_openai(
    profile: FaultProfile | FaultRule | dict[str, Any] | str | Path | None = None,
    on_event: Any = None,
) -> Iterator[OpenAIFaultInterceptor]:
    """Context manager for temporarily injecting faults into OpenAI SDK calls.

    This is the chaos-testing side: it *causes* failures. For protection see
    ``src.guard.ResilienceGuard``.
    """
    interceptor = OpenAIFaultInterceptor(profile, on_event=on_event)
    interceptor.patch()
    try:
        yield interceptor
    finally:
        interceptor.unpatch()
