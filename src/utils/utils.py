"""utilities functions
"""

from __future__ import annotations

import json
import re
from typing import Any


def llm_final_content(text: str) -> str:
    """Remove common artifacts from LLM outputs before JSON parsing.
    
    This function removes:
    - Thinking tags like <think>...</think> (closed or unclosed)
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Excessive whitespace
    
    Args:
        text: Raw LLM output text.
        
    Returns:
        Cleaned text ready for JSON parsing.
    """
    # Remove complete <think>...</think> tags (case-insensitive, multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove unclosed <think> tags (everything from <think> to end)
    text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove text before </think> tags
    text = re.sub(r'.*</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Strip excessive whitespace
    return text.strip()


def parse_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract and parse the first JSON object from an arbitrary string.

    This function first attempts to parse the entire text as JSON. If that
    fails, it searches for the first top-level ``{...}`` block and attempts to
    parse that segment as a JSON object.

    Args:
        text: The input text which may contain a JSON object.

    Returns:
        A dict representing the parsed JSON object, or ``None`` if parsing
        fails.
    """
    stripped = text.strip()

    # Fast path: entire content is JSON
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find first {...} block
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


__all__ = ["parse_first_json_object", "llm_final_content"]


