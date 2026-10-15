"""Fault Injection System Configuration Parser.

Provides data structures and parsing utilities for fault profiles and rules
used in SDK-level fault injection and resilience testing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FaultRule(BaseModel):
    """Configuration for an individual fault injection rule."""

    target: str = "openai"  # e.g., 'openai', 'nebius', 'nvidia', or '*'
    action: str = "error"  # e.g., 'error', 'delay', 'stream_drop', 'corrupt'
    error_type: str = "APIConnectionError"
    status_code: int = 500
    message: str = "Simulated 500: API connection error"
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    endpoints: list[str] = Field(default_factory=lambda: ["chat.completions.create"])
    max_occurrences: int | None = None
    delay_seconds: float = 0.0
    delay_min_seconds: float = 0.0
    delay_max_seconds: float = 0.0
    stream_drop_after_chunks: int = 2
    corrupt_type: str = "malformed_json"  # e.g., 'malformed_json', 'truncated'
    current_occurrences: int = 0

    @field_validator("endpoints", mode="before")
    @classmethod
    def _coerce_endpoints(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        return list(v)

    def matches_endpoint(self, endpoint: str) -> bool:
        """Check whether this rule applies to the specified endpoint."""
        if not self.endpoints or "*" in self.endpoints:
            return True
        return endpoint in self.endpoints

    def matches_target(self, target_or_url: str | None) -> bool:
        """Check whether this rule applies to the specified provider target or URL."""
        if not self.target or self.target in ("*", "all"):
            return True
        if target_or_url is None:
            return self.target in ("openai", "*", "all")

        target_lower = self.target.lower()
        url_lower = str(target_or_url).lower()

        if target_lower == url_lower:
            return True
        if "nebius" in target_lower and "nebius" in url_lower:
            return True
        if "nvidia" in target_lower and "nvidia" in url_lower:
            return True
        if "openai" in target_lower and ("openai" in url_lower or "api.openai.com" in url_lower):
            return True
        return False

    def get_delay(self) -> float:
        """Calculate effective delay considering static delay or min/max jitter range."""
        if self.delay_max_seconds > self.delay_min_seconds:
            return random.uniform(self.delay_min_seconds, self.delay_max_seconds)
        return self.delay_seconds

    def should_trigger(self, endpoint: str, target_or_url: str | None = None) -> bool:
        """Check if this fault rule should trigger for the given endpoint and target.

        Verifies endpoint match, target match, occurrence limits, and probability check.
        """
        if not self.matches_endpoint(endpoint):
            return False

        if not self.matches_target(target_or_url):
            return False

        if self.max_occurrences is not None and self.current_occurrences >= self.max_occurrences:
            return False

        if self.probability < 1.0:
            if self.probability <= 0.0 or random.random() > self.probability:
                return False

        return True

    def record_occurrence(self) -> None:
        """Increment occurrence counter."""
        self.current_occurrences += 1

    def reset(self) -> None:
        """Reset occurrence counter."""
        self.current_occurrences = 0


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaking resilience mechanism."""

    max_retries: int = 3
    failure_threshold: int = 3
    recovery_timeout: float = 30.0


class FaultProfile(BaseModel):
    """Top-level configuration profile defining active faults and resilience settings."""

    version: str = "1.0"
    enabled: bool = True
    faults: list[FaultRule] = Field(default_factory=list)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)

    def get_matching_fault(
        self, endpoint: str, target_or_url: str | None = None
    ) -> FaultRule | None:
        """Find the first matching fault rule for the given endpoint and target.

        Checks if profile is enabled, checks endpoints & target match, and checks probability.
        """
        if not self.enabled:
            return None

        for fault in self.faults:
            if fault.should_trigger(endpoint, target_or_url=target_or_url):
                return fault

        return None

    def reset(self) -> None:
        """Reset occurrence counters on all child fault rules."""
        for fault in self.faults:
            fault.reset()


class FaultConfigParser:
    """Parser and serializer for FaultProfile configurations."""

    @staticmethod
    def parse_dict(data: dict[str, Any]) -> FaultProfile:
        """Parse a FaultProfile instance from a dictionary."""
        return FaultProfile.model_validate(data)

    @staticmethod
    def parse_file(file_path: str | Path) -> FaultProfile:
        """Parse a FaultProfile instance from a JSON file."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fault profile file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FaultConfigParser.parse_dict(data)

    @staticmethod
    def save_to_file(profile: FaultProfile, file_path: str | Path) -> None:
        """Serialize and save a FaultProfile to a JSON file."""
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))


# Module-level aliases
parse_file = FaultConfigParser.parse_file
parse_dict = FaultConfigParser.parse_dict
save_to_file = FaultConfigParser.save_to_file


def load_fault_profile(path_or_dict: str | Path | dict[str, Any] | None = None) -> FaultProfile:
    """Convenience function to load a FaultProfile.

    - If path_or_dict is None, searches for 'fault_profile.json' in the current working directory
      or parent directories up to workspace root. Returns a default FaultProfile if not found.
    - If path_or_dict is a path/str, loads that file.
    - If path_or_dict is a dict, parses it directly.
    """
    if path_or_dict is None:
        # Check current working directory
        cwd_candidate = Path.cwd() / "fault_profile.json"
        if cwd_candidate.is_file():
            return FaultConfigParser.parse_file(cwd_candidate)

        # Check upward from module location
        current = Path(__file__).resolve().parent
        while current != current.parent:
            candidate = current / "fault_profile.json"
            if candidate.is_file():
                return FaultConfigParser.parse_file(candidate)
            current = current.parent

        return FaultProfile()

    if isinstance(path_or_dict, dict):
        return FaultConfigParser.parse_dict(path_or_dict)

    if isinstance(path_or_dict, (str, Path)):
        return FaultConfigParser.parse_file(path_or_dict)

    raise TypeError(f"Unsupported type for path_or_dict: {type(path_or_dict).__name__}")
