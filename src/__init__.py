"""ResilienceGuard: fire drills for AI agents on Nebius Token Factory.

Main pieces:
    src.proxy        fault-injection proxy in front of Token Factory (any framework, any language)
    src.agent_audit  runs an unmodified agent through the proxy under each drill and scores it
    src.fixer        Nemotron writes fixes; each is re-audited until it passes
    src.guard        circuit breaker / kill switch (used by the proxy, usable in-process)
    src.interceptor  in-process fault injection for the OpenAI Python SDK (library mode)
"""

from src.guard import CircuitOpenError, ResilienceGuard, RunawayHaltError

__all__ = ["CircuitOpenError", "ResilienceGuard", "RunawayHaltError"]
