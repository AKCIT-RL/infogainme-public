"""LLM configuration module to avoid circular imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class LLMConfig:
    """Configuration for `LLMAdapter`.

    Args:
        model: Model name (e.g., "gpt-4o-mini" or a vLLM-exposed model).
        api_key: API key to authenticate. If None, the OpenAI client will
            fallback to environment configuration.
        base_url: Optional custom base URL (e.g., vLLM OpenAI-compatible endpoint).
        timeout: Request timeout in seconds.
        max_tokens: Optional cap for tokens in the response.
        temperature: Sampling temperature.
        user: Optional user identifier for tracking/auditing.
        response_format: Optional response format (e.g., {"type": "json_object"}).
        extra: Extra provider-specific parameters to pass through.
    """

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    user: Optional[str] = None
    response_format: Optional[dict[str, Any]] = None
    use_reasoning: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


