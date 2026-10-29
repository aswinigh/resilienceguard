"""ResilienceGuard: circuit breaker and runaway-loop halt for OpenAI-compatible clients.

This is the *protection* half of the project (the fault interceptor is the testing half).
It sits around ``chat.completions.create`` and:

* counts consecutive transient upstream failures (connection errors, timeouts, 429, 5xx);
* opens the circuit after ``failure_threshold`` of them, after which calls fail fast with
  :class:`CircuitOpenError` without reaching the upstream API;
* lets one probe call through after ``recovery_timeout`` seconds (half-open) and closes the
  circuit again if it succeeds;
* raises :class:`RunawayHaltError` if the caller keeps calling while the circuit is open.
  That error derives from ``BaseException`` so a naive ``except Exception: retry`` loop
  cannot swallow it.

Two ways to use it::

    guard = ResilienceGuard()
    client = guard.wrap(OpenAI(...))          # protect one client

    with ResilienceGuard().protect():         # protect every OpenAI client in the process
        run_agent(prompt)
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Iterator

import httpx
import openai
from openai.resources.chat.completions import AsyncCompletions, Completions

__all__ = [
    "CircuitOpenError",
    "CircuitState",
    "ResilienceGuard",
    "RunawayHaltError",
    "is_transient_error",
]


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(openai.APIConnectionError):
    """Raised instead of calling upstream while the circuit is open.

    Subclasses ``openai.APIConnectionError`` so agents that already handle connection
    errors treat it like the outage it represents.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            request=httpx.Request("POST", "https://resilienceguard.local/circuit-open"),
        )


class RunawayHaltError(BaseException):
    """Raised when calls keep arriving while the circuit is open, or the call budget is spent.

    Deliberately a ``BaseException`` (like ``KeyboardInterrupt``): its job is to stop retry
    loops that catch ``Exception``. Catch it explicitly at the top of your agent if you want
    to handle it.
    """


def is_transient_error(err: BaseException) -> bool:
    """True for errors that indicate upstream trouble rather than a bad request."""
    if isinstance(err, CircuitOpenError):
        return False
    if isinstance(err, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
        return True
    if isinstance(err, openai.APIStatusError):
        return err.status_code >= 500 or err.status_code in (408, 409, 429)
    return False


EventHook = Callable[[str, dict], None]


class ResilienceGuard:
    """Circuit breaker + runaway halt for OpenAI-compatible chat completion calls."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        halt_after_blocked: int | None = 3,
        max_calls: int | None = None,
        on_event: EventHook | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            failure_threshold: consecutive transient failures that open the circuit.
            recovery_timeout: seconds the circuit stays open before allowing one probe call.
            halt_after_blocked: raise RunawayHaltError once this many calls have been blocked
                since the circuit opened. ``None`` disables the halt (fail fast forever).
            max_calls: optional hard budget of total calls; exceeding it halts.
            on_event: optional callback ``(event_name, info)`` for telemetry.
            clock: injectable time source (for tests).
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.halt_after_blocked = halt_after_blocked
        self.max_calls = max_calls
        self.on_event = on_event
        self._clock = clock
        self._lock = threading.RLock()
        self._patch_originals: tuple[Any, Any] | None = None
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Close the circuit, clear the halt and zero all counters."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = 0.0
            self._blocked_since_open = 0
            self._probe_in_flight = False
            self._halted = False
            self._halt_reason: str | None = None
            self.stats = {
                "calls": 0,
                "upstream_attempts": 0,
                "successes": 0,
                "failures": 0,
                "blocked": 0,
                "trips": 0,
            }

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._current_state()

    @property
    def halted(self) -> bool:
        return self._halted

    def snapshot(self) -> dict:
        """Current state and counters, JSON-serialisable."""
        with self._lock:
            return {
                **self.stats,
                "state": self._current_state().value,
                "consecutive_failures": self._consecutive_failures,
                "halted": self._halted,
                "halt_reason": self._halt_reason,
            }

    def _emit(self, event: str, **info: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, info)
            except Exception:  # telemetry must never break the call path
                pass

    def _current_state(self) -> CircuitState:
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._blocked_since_open = 0
            self._emit("half_open")
        return self._state

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        self._emit("halt", reason=reason)
        raise RunawayHaltError(reason)

    def _before_call(self) -> None:
        with self._lock:
            if self._halted:
                raise RunawayHaltError(self._halt_reason or "ResilienceGuard halted")
            if self.max_calls is not None and self.stats["calls"] >= self.max_calls:
                self._halt(f"Call budget of {self.max_calls} exhausted")
            self.stats["calls"] += 1

            state = self._current_state()
            if state is CircuitState.HALF_OPEN and self._probe_in_flight:
                state = CircuitState.OPEN  # only one probe at a time
            if state is CircuitState.OPEN:
                self.stats["blocked"] += 1
                self._blocked_since_open += 1
                self._emit("blocked", blocked_since_open=self._blocked_since_open)
                if (
                    self.halt_after_blocked is not None
                    and self._blocked_since_open >= self.halt_after_blocked
                ):
                    self._halt(
                        f"Runaway retry loop halted: {self._blocked_since_open} calls made "
                        f"while the circuit was open after {self.failure_threshold} "
                        "consecutive upstream failures"
                    )
                raise CircuitOpenError(
                    f"Circuit open after {self.failure_threshold} consecutive upstream "
                    "failures; call blocked without reaching the API"
                )
            if state is CircuitState.HALF_OPEN:
                self._probe_in_flight = True
            self.stats["upstream_attempts"] += 1
            self._emit("attempt", state=state.value)

    def _record_success(self) -> None:
        with self._lock:
            was = self._state
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._blocked_since_open = 0
            self._probe_in_flight = False
            self.stats["successes"] += 1
            self._emit("success")
            if was is not CircuitState.CLOSED:
                self._emit("closed")

    def _record_failure(self, err: BaseException) -> None:
        with self._lock:
            probe = self._probe_in_flight
            self._probe_in_flight = False
            if not is_transient_error(err):
                # Upstream answered (e.g. 400/401): not an outage, leave the circuit alone.
                self._emit("non_transient_error", error=type(err).__name__)
                return
            self._consecutive_failures += 1
            self.stats["failures"] += 1
            self._emit("failure", error=type(err).__name__, consecutive=self._consecutive_failures)
            if probe or self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._blocked_since_open = 0
                self.stats["trips"] += 1
                self._emit("trip", consecutive=self._consecutive_failures)

    def _release_probe(self) -> None:
        with self._lock:
            self._probe_in_flight = False

    # Public hooks for callers that are not Python SDK calls (e.g. the HTTP proxy).
    def before_call(self) -> None:
        """Raise CircuitOpenError / RunawayHaltError if this call must not go upstream."""
        self._before_call()

    def record_success(self) -> None:
        self._record_success()

    def record_failure(self, err: BaseException) -> None:
        self._record_failure(err)

    # ------------------------------------------------------------------ calls

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous completion call through the breaker."""
        self._before_call()
        try:
            result = fn(*args, **kwargs)
        except Exception as err:
            self._record_failure(err)
            raise
        except BaseException:
            self._release_probe()
            raise
        if kwargs.get("stream"):
            return self._guard_stream(result)
        self._record_success()
        return result

    async def acall(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run an async completion call through the breaker."""
        self._before_call()
        try:
            result = await fn(*args, **kwargs)
        except Exception as err:
            self._record_failure(err)
            raise
        except BaseException:
            self._release_probe()
            raise
        if kwargs.get("stream"):
            return self._guard_async_stream(result)
        self._record_success()
        return result

    def _guard_stream(self, stream: Any) -> Iterator[Any]:
        """Success is recorded when the stream finishes; a mid-stream drop is a failure."""
        failed = False
        try:
            for chunk in stream:
                yield chunk
        except Exception as err:
            failed = True
            self._record_failure(err)
            raise
        finally:
            if not failed:
                self._record_success()

    async def _guard_async_stream(self, stream: Any) -> Any:
        failed = False
        try:
            async for chunk in stream:
                yield chunk
        except Exception as err:
            failed = True
            self._record_failure(err)
            raise
        finally:
            if not failed:
                self._record_success()

    # ------------------------------------------------------------------ wrapping

    def wrap(self, client: Any) -> Any:
        """Return a proxy of ``client`` whose ``chat.completions.create`` is guarded.

        Everything else on the client passes through unchanged.
        """
        return _GuardedClient(client, self)

    def install(self) -> "ResilienceGuard":
        """Patch the OpenAI SDK so every client in this process goes through the guard."""
        if self._patch_originals is not None:
            return self
        orig, aorig = Completions.create, AsyncCompletions.create
        guard = self

        @functools.wraps(orig)
        def create(completions_self: Any, *args: Any, **kwargs: Any) -> Any:
            return guard.call(orig, completions_self, *args, **kwargs)

        @functools.wraps(aorig)
        async def acreate(completions_self: Any, *args: Any, **kwargs: Any) -> Any:
            return await guard.acall(aorig, completions_self, *args, **kwargs)

        Completions.create = create
        AsyncCompletions.create = acreate
        self._patch_originals = (orig, aorig)
        return self

    def uninstall(self) -> None:
        """Undo :meth:`install`."""
        if self._patch_originals is None:
            return
        Completions.create, AsyncCompletions.create = self._patch_originals
        self._patch_originals = None

    @contextmanager
    def protect(self) -> Iterator["ResilienceGuard"]:
        """Context manager form of :meth:`install` / :meth:`uninstall`."""
        self.install()
        try:
            yield self
        finally:
            self.uninstall()


class _GuardedCompletions:
    def __init__(self, inner: Any, guard: ResilienceGuard, is_async: bool) -> None:
        self._inner = inner
        self._guard = guard
        self._is_async = is_async

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_async:
            return self._guard.acall(self._inner.create, *args, **kwargs)
        return self._guard.call(self._inner.create, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _GuardedChat:
    def __init__(self, inner: Any, guard: ResilienceGuard, is_async: bool) -> None:
        self._inner = inner
        self.completions = _GuardedCompletions(inner.completions, guard, is_async)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _GuardedClient:
    def __init__(self, client: Any, guard: ResilienceGuard) -> None:
        self._client = client
        self.guard = guard
        self.chat = _GuardedChat(client.chat, guard, isinstance(client, openai.AsyncOpenAI))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
