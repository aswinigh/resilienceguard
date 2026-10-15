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

    target: str = "openai"
    action: str = "error"  # e.g., 'error', 'delay', 'corrupt'
    error_type: str = "APIConnectionError"
    status_code: int = 500
    message: str = "Simulated 500: API connection error"
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    endpoints: list[str] = Field(default_factory=lambda: ["chat.completions.create"])
    max_occurrences: int | None = None
    delay_seconds: float = 0.0
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

    def should_trigger(self, endpoint: str) -> bool:
        """Check if this fault rule should trigger for the given endpoint.

        Verifies endpoint match, occurrence limits, and probability check.
        """
        if not self.matches_endpoint(endpoint):
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

    def get_matching_fault(self, endpoint: str) -> FaultRule | None:
        """Find the first matching fault rule for the given endpoint.

        Checks if profile is enabled, checks endpoints match, and checks probability.
        """
        if not self.enabled:
            return None

        for fault in self.faults:
            if fault.should_trigger(endpoint):
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
