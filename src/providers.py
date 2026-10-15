"""Sponsor LLM Provider Configuration for Nebius x NVIDIA Hackathon.

Provides standardized OpenAI-compatible client factories and default model configurations
for Nebius Token Factory, NVIDIA NIM, and OpenAI.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import openai

# Sponsor endpoint definitions
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
NEBIUS_DEFAULT_MODEL = os.environ.get(
    "NEBIUS_DEFAULT_MODEL", "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
)

NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_DEFAULT_MODEL = os.environ.get(
    "NVIDIA_DEFAULT_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"
)

OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o")

SUPPORTED_PROVIDERS = ("nebius", "nvidia", "openai")


def get_sponsor_client(
    provider: str = "nebius",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    **client_kwargs: Any,
) -> Tuple[openai.OpenAI, str]:
    """Instantiate an OpenAI-compatible client configured for the chosen sponsor/provider.

    Args:
        provider: One of 'nebius', 'nvidia', or 'openai'.
        api_key: Optional explicit API key override. If None, reads provider env var.
        base_url: Optional base_url override.
        model: Optional model override. If None, selects provider default model.
        **client_kwargs: Additional arguments passed to openai.OpenAI().

    Returns:
        Tuple of (configured_openai_client, selected_model_name).
    """
    provider_key = provider.lower().strip()

    if provider_key == "nebius":
        resolved_key = api_key or os.environ.get("NEBIUS_API_KEY") or "mock-nebius-key"
        resolved_url = base_url or NEBIUS_BASE_URL
        resolved_model = model or NEBIUS_DEFAULT_MODEL

    elif provider_key == "nvidia":
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY") or "mock-nvidia-key"
        resolved_url = base_url or NVIDIA_BASE_URL
        resolved_model = model or NVIDIA_DEFAULT_MODEL

    elif provider_key == "openai":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "mock-openai-key"
        resolved_url = base_url  # defaults to standard OpenAI endpoint when None
        resolved_model = model or OPENAI_DEFAULT_MODEL

    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose from {SUPPORTED_PROVIDERS}"
        )

    client_args: dict[str, Any] = {"api_key": resolved_key, **client_kwargs}
    if resolved_url:
        client_args["base_url"] = resolved_url

    client = openai.OpenAI(**client_args)
    return client, resolved_model
