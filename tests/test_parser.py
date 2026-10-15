"""Unit tests for the Fault Injection System configuration parser."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config.parser import (
    CircuitBreakerConfig,
    FaultConfigParser,
    FaultProfile,
    FaultRule,
    load_fault_profile,
    parse_dict,
    parse_file,
    save_to_file,
)


def test_default_models():
    """Verify default values of all configuration models."""
    rule = FaultRule()
    assert rule.target == "openai"
    assert rule.action == "error"
    assert rule.error_type == "APIConnectionError"
    assert rule.status_code == 500
    assert rule.message == "Simulated 500: API connection error"
    assert rule.probability == 1.0
    assert rule.endpoints == ["chat.completions.create"]
    assert rule.max_occurrences is None
    assert rule.delay_seconds == 0.0
    assert rule.current_occurrences == 0

    cb = CircuitBreakerConfig()
    assert cb.max_retries == 3
    assert cb.failure_threshold == 3
    assert cb.recovery_timeout == 30.0

    profile = FaultProfile()
    assert profile.version == "1.0"
    assert profile.enabled is True
    assert profile.faults == []
    assert profile.circuit_breaker.max_retries == 3


def test_parse_dict():
    """Verify parsing a configuration from a Python dictionary."""
    data = {
        "version": "1.0",
        "enabled": True,
        "faults": [
            {
                "target": "openai",
                "action": "error",
                "error_type": "RateLimitError",
                "status_code": 429,
                "message": "Rate limit exceeded",
                "probability": 0.5,
                "endpoints": ["chat.completions.create"],
                "max_occurrences": 5,
                "delay_seconds": 1.5,
            }
        ],
        "circuit_breaker": {
            "max_retries": 2,
            "failure_threshold": 2,
            "recovery_timeout": 15.0,
        },
    }
    profile = parse_dict(data)
    assert profile.version == "1.0"
    assert profile.enabled is True
    assert len(profile.faults) == 1
    assert profile.faults[0].error_type == "RateLimitError"
    assert profile.faults[0].status_code == 429
    assert profile.faults[0].probability == 0.5
    assert profile.faults[0].max_occurrences == 5
    assert profile.faults[0].delay_seconds == 1.5
    assert profile.circuit_breaker.max_retries == 2
    assert profile.circuit_breaker.failure_threshold == 2
    assert profile.circuit_breaker.recovery_timeout == 15.0


def test_endpoint_coercion():
    """Verify a single string endpoint is coerced into a list."""
    rule = FaultRule.model_validate({"endpoints": "chat.completions.create"})
    assert rule.endpoints == ["chat.completions.create"]


def test_load_default_fault_profile_file():
    """Verify loading the project's root fault_profile.json file."""
    profile = load_fault_profile()
    assert profile.version == "1.0"
    assert profile.enabled is True
    assert len(profile.faults) == 1

    fault = profile.faults[0]
    assert fault.target == "openai"
    assert fault.action == "error"
    assert fault.error_type == "APIConnectionError"
    assert fault.status_code == 500
    assert fault.message == "Simulated 500: OpenAI upstream APIConnectionError"
    assert fault.probability == 1.0
    assert fault.endpoints == ["chat.completions.create"]
    assert fault.max_occurrences == 3
    assert fault.delay_seconds == 0.0

    assert profile.circuit_breaker.max_retries == 3
    assert profile.circuit_breaker.failure_threshold == 3
    assert profile.circuit_breaker.recovery_timeout == 30.0


def test_load_fault_profile_from_path(tmp_path: Path):
    """Verify loading from explicit string and Path arguments."""
    sample_json = tmp_path / "custom_profile.json"
    sample_json.write_text(
        '{"version": "2.0", "enabled": true, "faults": [], "circuit_breaker": {"max_retries": 5}}',
        encoding="utf-8",
    )

    profile_from_str = load_fault_profile(str(sample_json))
    assert profile_from_str.version == "2.0"
    assert profile_from_str.circuit_breaker.max_retries == 5

    profile_from_path = load_fault_profile(sample_json)
    assert profile_from_path.version == "2.0"


def test_load_fault_profile_from_dict():
    """Verify load_fault_profile directly accepts a dict."""
    profile = load_fault_profile({"version": "1.1", "enabled": False})
    assert profile.version == "1.1"
    assert profile.enabled is False


def test_load_fault_profile_unsupported_type():
    """Verify passing an invalid type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported type"):
        load_fault_profile(12345)  # type: ignore[arg-type]


def test_matching_endpoint():
    """Verify matching fault rules by endpoint."""
    rule = FaultRule(
        endpoints=["chat.completions.create"],
        probability=1.0,
    )
    profile = FaultProfile(faults=[rule])

    # Matching endpoint
    matched = profile.get_matching_fault("chat.completions.create")
    assert matched is not None
    assert matched.error_type == "APIConnectionError"

    # Non-matching endpoint
    unmatched = profile.get_matching_fault("embeddings.create")
    assert unmatched is None


def test_wildcard_endpoint_matching():
    """Verify wildcard '*' matches any endpoint."""
    rule = FaultRule(endpoints=["*"], probability=1.0)
    profile = FaultProfile(faults=[rule])

    assert profile.get_matching_fault("chat.completions.create") is not None
    assert profile.get_matching_fault("embeddings.create") is not None
    assert profile.get_matching_fault("models.list") is not None


def test_profile_disabled_returns_none():
    """Verify no faults match if the entire profile is disabled."""
    rule = FaultRule(endpoints=["*"], probability=1.0)
    profile = FaultProfile(enabled=False, faults=[rule])

    assert profile.get_matching_fault("chat.completions.create") is None


def test_probability_filtering():
    """Verify rules with probability 0.0 never match."""
    zero_prob_rule = FaultRule(endpoints=["chat.completions.create"], probability=0.0)
    profile = FaultProfile(faults=[zero_prob_rule])
    assert profile.get_matching_fault("chat.completions.create") is None


def test_max_occurrences_limit_and_reset():
    """Verify fault rule respects max_occurrences limit and reset behavior."""
    rule = FaultRule(
        endpoints=["chat.completions.create"],
        probability=1.0,
        max_occurrences=2,
    )
    profile = FaultProfile(faults=[rule])

    # First match
    matched = profile.get_matching_fault("chat.completions.create")
    assert matched is not None
    matched.record_occurrence()
    assert matched.current_occurrences == 1

    # Second match
    matched = profile.get_matching_fault("chat.completions.create")
    assert matched is not None
    matched.record_occurrence()
    assert matched.current_occurrences == 2

    # Third match should be None because max_occurrences (2) is reached
    assert profile.get_matching_fault("chat.completions.create") is None

    # Reset profile
    profile.reset()
    assert rule.current_occurrences == 0
    assert profile.get_matching_fault("chat.completions.create") is not None


def test_save_and_reload(tmp_path: Path):
    """Verify saving a profile to file and reloading it."""
    out_file = tmp_path / "sub" / "saved_profile.json"
    profile = FaultProfile(
        version="1.5",
        enabled=True,
        faults=[
            FaultRule(
                target="openai",
                action="error",
                error_type="InternalServerError",
                status_code=500,
                endpoints=["chat.completions.create"],
            )
        ],
        circuit_breaker=CircuitBreakerConfig(max_retries=4),
    )

    save_to_file(profile, out_file)
    assert out_file.is_file()

    reloaded = parse_file(out_file)
    assert reloaded.version == "1.5"
    assert len(reloaded.faults) == 1
    assert reloaded.faults[0].error_type == "InternalServerError"
    assert reloaded.circuit_breaker.max_retries == 4


def test_parse_nonexistent_file_raises():
    """Verify parse_file raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        parse_file("non_existent_file_abc123.json")


def test_invalid_probability_validation():
    """Verify probability must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        FaultRule(probability=1.5)

    with pytest.raises(ValidationError):
        FaultRule(probability=-0.1)
