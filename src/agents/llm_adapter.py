"""Lightweight OpenAI-compatible LLM adapter.

This module provides a minimal adapter around OpenAI (or OpenAI-compatible) chat
completions, maintaining an internal chat history. It is intentionally small
and dependency-light, following the project guidelines.

Public API (aligned with UML/IDEIA_ESCOPO):
  - class LLMAdapter
    - append_history(role: str, text: str) -> None
    - reset_history() -> None

No side effects occur on import. The OpenAI client is imported lazily at call time.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from .llm_config import LLMConfig
from ..utils.utils import llm_final_content

logger = logging.getLogger(__name__)

load_dotenv()

Role = Literal["system", "user", "assistant"]


class LLMAdapterError(RuntimeError):
    """Represents errors raised by the `LLMAdapter`."""


class ContextLengthExceededError(LLMAdapterError):
    """Raised when the provider rejects a request because the input is longer
    than the model's context window. Deterministic failure — never retried."""


def _is_context_length_error(exc: Exception) -> bool:
    """Detect provider-side context-overflow rejections.

    OpenAI-compatible servers (incl. vLLM) return HTTP 400 with a body whose
    message starts with "This model's maximum context length is..." or
    "context_length_exceeded". We treat both the status code and the message
    text as signals so this works across server implementations.
    """
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    msg = (str(exc) or "").lower()
    if status == 400 and ("context length" in msg or "context_length_exceeded" in msg
                          or "maximum context" in msg or "reduce the length" in msg):
        return True
    # Also catch when status isn't readable (e.g. wrapped exception): pure msg match.
    return ("context length exceeded" in msg or "maximum context length" in msg
            or "context_length_exceeded" in msg)


class LLMAdapter:
    """Minimal adapter for OpenAI-compatible chat completions.

    This adapter keeps an internal chat `history` and provides methods to append
    messages and reset the history. The `generate` method performs a chat
    completion request using the configured provider.

    Example:
        config = LLMConfig(model="gpt-4o-mini")
        llm = LLMAdapter(config)
        llm.append_history("system", "You are a helpful assistant.")
        llm.append_history("user", "Hello!")
        reply = llm.generate()
    """

    def __init__(
        self, 
        config: Optional[LLMConfig] = None,
        save_history: bool = True,
        save_reasoning: bool = False,
        ) -> None:
        self._config: LLMConfig = config
        self._history: list[dict[str, str]] = [] if save_history else None
        self._save_history = save_history
        self._save_reasoning = save_reasoning
        self._use_reasoning = self._config.use_reasoning
        # Separate storage for raw responses with reasoning (for export only)
        self._reasoning_history: list[dict[str, str]] = [] if save_reasoning else None

        assert self._config.api_key is not None if not self._config.base_url else True, "Both api_key and base_url cannot be None"

    # --- History management (Pythonic) ---
    def append_history(self, role: Role, text: str) -> None:
        """Append one message to the internal history.

        Args:
            role: One of {"system", "user", "assistant"}.
            text: Message content.

        Raises:
            ValueError: If `role` is invalid or `text` is empty.
        """
        assert self._save_history, "Trying to append history but save_history is False"
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {role}")
        if not text:
            raise ValueError("text must be non-empty")
        self._history.append({"role": role, "content": text})
        
        # Mirror system and user messages to reasoning_history for complete context
        if self._save_reasoning and role in ("system", "user"):
            self._reasoning_history.append({"role": role, "content": text})

    def reset_history(self) -> None:
        """Clear the internal chat history."""
        self._history.clear()

    def pop_last_if_assistant(self) -> bool:
        """Drop the last history entry if it is an assistant message."""
        if not self._save_history or not self._history:
            return False
        if self._history[-1].get("role") != "assistant":
            return False
        self._history.pop()
        if self._save_reasoning and self._reasoning_history and self._reasoning_history[-1].get("role") == "assistant":
            self._reasoning_history.pop()
        return True


    # --- Accessors ---
    @property
    def history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the internal history for read-only use."""
        return list(self._history)
    
    @property
    def reasoning_history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the reasoning history (raw responses with <think> tags).
        
        Returns:
            List of messages with raw content including reasoning tags.
            Empty list if save_reasoning=False.
        """
        return list(self._reasoning_history) if self._reasoning_history else []

    @property
    def config(self) -> LLMConfig:
        """Return the underlying configuration."""
        return self._config

    @staticmethod
    def see_messages(messages: list[dict[str, str]]) -> str:
        """Return the messages as a string pretty-printed."""
        lines = []
        for i, msg in enumerate(messages, 1):
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{i:2d}. {role}: {content}")
        
        return "\n".join(lines)
        
    def see_history(self) -> str:
        """Return the internal history as a string pretty-printed."""
        if not self._save_history or not self._history:
            return "No history available."
        
        return self.see_messages(self._history)


    # --- Core generation ---
    def generate(
        self,
        add_to_history: Optional[bool] = None,
        messages: Optional[list[dict[str, str]]] = None,
        *,
        stateless: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        bail_on_context_length: bool = False,
        with_raw: bool = False,
        # extra: Optional[dict[str, Any]] = None
    ) -> str | tuple[str, str]:
        """Call the provider to generate the next assistant message.

        Args:
            messages: If provided, use these messages instead of `self.history`.
            add_to_history: If True, append the assistant reply to `self.history`.
            stateless: If True, use only `messages` for the call but still save to history.
                      Useful for agents that need history export but stateless inference.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for max tokens.
            response_format: Optional override for response format (e.g., {"type": "json_object"}).
            extra: Optional dict of provider-specific parameters.

        Returns:
            Assistant reply text.

        Raises:
            LLMAdapterError: On client import or API errors.
            ValueError: If there are no messages to send.
        """
        if add_to_history is None:
            add_to_history = self._save_history

        if add_to_history and not self._save_history:
            raise ValueError("Trying to add to history but save_history is False")

        # Determine payload messages
        if stateless:
            # Stateless mode: use provided messages or raise error
            if messages is None:
                raise ValueError("Stateless mode requires explicit 'messages' parameter")
            payload_messages = messages
        else:
            # Normal mode: use messages if provided, otherwise use history
            payload_messages = messages if messages is not None else self.reasoning_history if self._use_reasoning else self._history
        
        if not payload_messages:
            raise ValueError("No messages to send. Provide `messages` or add history.")

        client_kwargs: dict[str, Any] = {}
        if self._config.api_key is not None:
            client_kwargs["api_key"] = self._config.api_key
        if self._config.base_url is not None:
            client_kwargs["base_url"] = self._config.base_url

        request_kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": payload_messages,
            "timeout": self._config.timeout,
        }
        
        # Only add temperature if explicitly set (allows model to use generation_config default)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if self._config.user is not None:
            request_kwargs["user"] = self._config.user
        if self._config.max_tokens is not None or max_tokens is not None:
            request_kwargs["max_tokens"] = self._config.max_tokens if max_tokens is None else max_tokens
        if response_format is not None or self._config.response_format is not None:
            request_kwargs["response_format"] = (
                response_format if response_format is not None
                else self._config.response_format
            )
        if self._config.extra:
            request_kwargs.update(self._config.extra)
        # if extra:
        #     print(f"Extra parameters: {extra}")
        #     for key, value in extra.items():
        #         request_kwargs[key] = value

        # Special handling for gpt-5 models
        if self._config.model.startswith("gpt-5"):
            if "max_tokens" in request_kwargs:
                # request_kwargs["max_completion_tokens"] = request_kwargs["max_tokens"]
                del request_kwargs["max_tokens"]
            if "temperature" in request_kwargs:
                del request_kwargs["temperature"]
        
        # Special handling for Gemini models via OpenAI API
        # Gemini requires at least one user message, not just system
        if self._config.model.startswith("gemini"):
            if payload_messages and len(payload_messages) == 1 and payload_messages[0].get("role") == "system":
                # Convert system-only to system + user pattern
                system_msg = payload_messages[0]["content"]
                request_kwargs["messages"] = [
                    {"role": "user", "content": f"{system_msg}\n\nSTARTING GAME!."}
                ]

        # Retry logic with exponential backoff (capped) + Retry-After for 429s
        max_retries = 50
        base_delay = 1.0   # seconds
        max_delay = 60.0   # cap so backoff never exceeds 1 minute

        # Single OpenAI client for the whole retry loop. The httpx connection
        # pool inside it must be drained explicitly via close() — otherwise
        # peer-half-closed sockets pile up in CLOSE_WAIT and exhaust FDs.
        client = OpenAI(**client_kwargs)
        try:
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(**request_kwargs)

                    # Build raw_content, prepending reasoning_content if present
                    # (vLLM with Qwen3 thinking returns reasoning in a separate field)
                    message = completion.choices[0].message
                    content = message.content or ""
                    extras = message.model_extra or {}
                    reasoning_content = extras.get("reasoning") or extras.get("reasoning_content")
                    if reasoning_content:
                        raw_content = f"<think>{reasoning_content}</think>{content}"
                    else:
                        raw_content = content
                    if not raw_content:
                        raise LLMAdapterError("Empty response from provider.")

                    # Clean content for return
                    final_content = llm_final_content(raw_content)
                    if not final_content:
                        raise LLMAdapterError("Response not in the expected format.")

                    break  # Success, exit retry loop
                except Exception as exc:
                    # Context-length errors are deterministic — retrying never
                    # helps and burns ~45 minutes of backoff per failed call.
                    # The caller opts in (e.g. judge_eval) so unrelated paths
                    # keep their existing best-effort retry behavior.
                    if bail_on_context_length and _is_context_length_error(exc):
                        raise ContextLengthExceededError(str(exc)) from exc

                    if attempt < max_retries - 1:
                        # Honour Retry-After header for 429 rate-limit responses
                        retry_after = None
                        try:
                            from openai import RateLimitError as _RLE
                            if isinstance(exc, _RLE):
                                retry_after = float(
                                    getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", 0)
                                    or getattr(exc, "retry_after", 0)
                                    or 0
                                ) or None
                        except Exception:
                            pass

                        if retry_after and retry_after > 0:
                            delay = retry_after
                        else:
                            delay = min(base_delay * (2 ** attempt), max_delay)

                        logger.warning("API error (attempt %d/%d): %s. Retrying in %.1fs...", attempt + 1, max_retries, exc, delay)
                        time.sleep(delay)
                    else:
                        # Final attempt failed
                        raise LLMAdapterError(f"Provider error after {max_retries} attempts: {exc}") from exc
        finally:
            try:
                client.close()
            except Exception:
                pass


        # Save to appropriate histories
        if add_to_history and final_content:
            # Main history: always cleaned (used as input for next turns)
            self._history.append({"role": "assistant", "content": final_content})
        
        if self._save_reasoning and raw_content:
            # Reasoning history: always raw (used for export/analysis only)
            self._reasoning_history.append({"role": "assistant", "content": raw_content})

        # with_raw: caller also wants the uncleaned output (incl. <think>…</think>
        # reasoning) for auditing — thread-safe (returned, not stored on self).
        if with_raw:
            return final_content, raw_content
        return final_content


__all__ = [
    "LLMAdapter",
    "LLMAdapterError",
]


if __name__ == "__main__":
    
    # Teste usando o modelo gpt-4o-mini através da API da OpenAI
    # adapter = LLMAdapter(LLMConfig(model="gpt-4o-mini"))

    # Teste através da API do vLLM
    adapter = LLMAdapter(
        LLMConfig(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # deve bater com o served_model_name
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",          # dummy
            temperature=0.2,
            max_tokens=128,
            timeout=60.0,
        )
    )

    adapter.append_history("system", "You are a helpful assistant.")
    adapter.append_history("user", "Explain how a geographic knowledge graph works.")
    resposta = adapter.generate()
    print(resposta)