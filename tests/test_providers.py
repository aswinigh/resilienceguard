"""Unit tests for sponsor provider configurations (Nebius Token Factory, NVIDIA NIM, OpenAI)."""

import pytest
from src.providers import (
    NEBIUS_BASE_URL,
    NEBIUS_DEFAULT_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    get_sponsor_client,
)


def test_nebius_client_configuration():
    """Verify Nebius client default base_url, model, and key handling."""
    client, model = get_sponsor_client(provider="nebius", api_key="test-nebius-key")
    assert str(client.base_url).rstrip("/") == NEBIUS_BASE_URL.rstrip("/")
    assert model == NEBIUS_DEFAULT_MODEL
    assert client.api_key == "test-nebius-key"


def test_nvidia_client_configuration():
    """Verify NVIDIA NIM client default base_url, model, and key handling."""
    client, model = get_sponsor_client(provider="nvidia", api_key="test-nvidia-key")
    assert str(client.base_url).rstrip("/") == NVIDIA_BASE_URL.rstrip("/")
    assert model == NVIDIA_DEFAULT_MODEL
    assert client.api_key == "test-nvidia-key"


def test_openai_client_configuration():
    """Verify standard OpenAI client configuration."""
    client, model = get_sponsor_client(provider="openai", api_key="test-openai-key")
    assert model == OPENAI_DEFAULT_MODEL
    assert client.api_key == "test-openai-key"


def test_unsupported_provider_raises_value_error():
    """Verify ValueError when requesting an unknown provider."""
    with pytest.raises(ValueError, match="Unsupported provider 'anthropic'"):
        get_sponsor_client(provider="anthropic")


def test_custom_model_and_url_override():
    """Verify explicit override of model and base_url."""
    custom_url = "https://custom.endpoint.ai/v1"
    custom_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    client, model = get_sponsor_client(
        provider="nebius",
        base_url=custom_url,
        model=custom_model,
        api_key="override-key",
    )
    assert str(client.base_url).rstrip("/") == custom_url.rstrip("/")
    assert model == custom_model
    assert client.api_key == "override-key"
