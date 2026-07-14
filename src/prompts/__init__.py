"""Prompt management utilities.

This module provides utilities to load and manage system prompts from markdown files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


_PROMPT_CACHE: Dict[str, str] = {}


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from a markdown file.
    
    Args:
        prompt_name: Name of the prompt file (without .md extension).
        
    Returns:
        The prompt content as a string.
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
        
    Note:
        Prompts are cached after first load for performance.
    """
    if prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]
    
    # Get the directory where this module is located
    prompts_dir = Path(__file__).parent
    prompt_file = prompts_dir / f"{prompt_name}.md"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    _PROMPT_CACHE[prompt_name] = content
    return content


def get_seeker_system_prompt(
    target_noun: str = "city",
    domain_description: str = "geographic (cities, countries, regions)",
    max_turns: int = 25,
    observability_mode: str = "PARTIALLY_OBSERVABLE",
    pool_description: str = "",
) -> str:
    """Get the SeekerAgent system prompt.

    Args:
        target_noun: Term for the target ("city" or "object").
        domain_description: Description of the domain for context.
        max_turns: Maximum number of turns allowed.
        observability_mode: "FULLY_OBSERVABLE" or "PARTIALLY_OBSERVABLE".
        pool_description: Optional extra context about how the candidate pool
            was constructed. Injected only into the Seeker prompt so the model
            can adjust its prior. Leave empty for baseline behaviour.
    """
    if observability_mode in ("FULLY_OBSERVABLE", "FO"):
        prompt_name = "seeker_system_fo"
    elif observability_mode in ("INITIALLY_OBSERVABLE", "IO"):
        prompt_name = "seeker_system_io"
    else:
        prompt_name = "seeker_system_po"
    content = load_prompt(prompt_name)
    prompt = (
        content
        .replace("{TARGET_NOUN}", target_noun)
        .replace("{DOMAIN_DESCRIPTION}", domain_description)
        .replace("{MAX_TURNS}", str(max_turns))
    )
    if pool_description:
        prompt += f"\n\n## Candidate Pool\n\n{pool_description}"
    return prompt


def get_oracle_system_prompt(
    target_noun: str = "city",
    domain_description: str = "geographic (cities, countries, regions)",
) -> str:
    """Get the OracleAgent system prompt.

    Args:
        target_noun: Term for the target ("city" or "object").
        domain_description: Description of the domain for context.
    """
    content = load_prompt("oracle_system")
    return content.replace("{TARGET_NOUN}", target_noun).replace(
        "{DOMAIN_DESCRIPTION}", domain_description
    )


def get_pruner_system_prompt(
    target_noun: str = "city",
) -> str:
    """Get the PrunerAgent system prompt.

    Args:
        target_noun: Noun for target in instructions ("city", "object", "disease").
    """
    content = load_prompt("pruner_system")
    return content.replace("{TARGET_NOUN}", target_noun)


def get_reasoning_synthesis_prompt() -> str:
    """Get the reasoning synthesis system prompt."""
    return load_prompt("reasoning_synthesis")


def get_belief_state_extraction_prompt() -> str:
    """Get the belief-state extraction system prompt."""
    return load_prompt("belief_state_extraction")


def clear_cache() -> None:
    """Clear the prompt cache. Useful for testing or reloading prompts."""
    _PROMPT_CACHE.clear()


__all__ = [
    "load_prompt",
    "get_seeker_system_prompt", 
    "get_oracle_system_prompt",
    "get_pruner_system_prompt",
    "get_reasoning_synthesis_prompt",
    "get_belief_state_extraction_prompt",
    "clear_cache",
]


