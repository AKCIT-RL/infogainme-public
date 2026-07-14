"""Reasoning synthesis utilities for SeekerAgent traces."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.llm_adapter import LLMAdapter
from ..agents.llm_config import LLMConfig
from ..prompts import get_reasoning_synthesis_prompt
from ..utils import parse_first_json_object, ClaryLogger

logger = ClaryLogger.get_logger(__name__)

# Pre-compiled regex patterns for better performance.
# `\s*` em vez de `\n` literal: gemma-4 emite "<think>texto</think>resposta"
# sem quebras de linha (visto em job 19798). Qwen/Olmo/Phi seguem usando
# "<think>\ntexto\n</think>" — `\s*` cobre 0+ whitespace, então é backwards-
# compatible. O `.strip()` no group(1) remove qualquer whitespace residual.
_THINK_PATTERN = re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL)
_ORACLE_PATTERN = re.compile(r'\[Oracle\]\s*-\s*(.*?)(?:\n|$)')


def load_seeker_conversation(file_path: Path) -> Dict[str, Any]:
    """Load seeker conversation data from JSON file.
    
    Args:
        file_path: Path to the seeker.json file.
        
    Returns:
        Dictionary containing the conversation data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Seeker conversation file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_reasoning_from_message(message: Dict[str, str]) -> Optional[str]:
    """Extract reasoning content from a message with <think> tags.
    
    Args:
        message: Message dictionary with 'role' and 'content' keys.
        
    Returns:
        Extracted reasoning content or None if no <think> tags found.
    """
    if message.get("role") != "assistant":
        return None
    
    content = message.get("content", "")
    
    # Look for <think> tags and extract content
    think_match = _THINK_PATTERN.search(content)
    if think_match:
        return think_match.group(1).strip()
    
    # If no tags found, check if this is a reasoning message (long content, not a simple question)
    # Reasoning messages are typically longer and contain analysis
    if len(content) > 100 and not content.startswith("Is the target"):
        return content.strip()
    
    return None


def synthesize_reasoning_trace(
    reasoning_text: str,
    llm_adapter: LLMAdapter
) -> Dict[str, Any]:
    """Synthesize a single reasoning trace using LLM.
    
    Args:
        reasoning_text: The raw reasoning text to synthesize.
        llm_adapter: LLM adapter to use for synthesis.
        
    Returns:
        Dictionary containing synthesized reasoning trace.
        
    Raises:
        ValueError: If synthesis fails.
    """
    # Prepare the synthesis prompt
    synthesis_prompt = get_reasoning_synthesis_prompt()
    
    user_content = f"<think>\n{reasoning_text}\n</think>"

    # Create messages for LLM call
    messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Generate synthesis using stateless call. `raw_content` (full model output,
    # incl. <think>…</think> reasoning) is saved by default as `raw_response`
    # for auditing / model comparison — on success AND on failure. The JSON is
    # parsed from the CLEANED content so reasoning braces don't break parsing.
    final_content = ""
    raw_content = ""
    try:
        final_content, raw_content = llm_adapter.generate(
            messages=messages,
            stateless=True,
            add_to_history=False,
            with_raw=True,
        )

        # Parse JSON from the cleaned content (not raw — reasoning may contain {})
        parsed = parse_first_json_object(final_content)
        if not parsed:
            raise ValueError(f"Invalid JSON response: {final_content}")

        # Validate required fields
        required_fields = ["summary", "questions_considered", "decision_rationale"]
        for field in required_fields:
            if field not in parsed:
                parsed[field] = f"Missing field: {field}"

        parsed["raw_response"] = raw_content
        return parsed

    except Exception as e:
        # Fallback: create basic structure if synthesis fails
        return {
            "summary": "Synthesis failed",
            "questions_considered": [],
            "decision_rationale": f"Error: {str(e)}",
            "raw_response": raw_content or final_content,
        }


def extract_oracle_answer(message_content: str) -> str:
    """Extract oracle answer from message content.
    
    Args:
        message_content: The content of the user message.
        
    Returns:
        The oracle answer text.
    """
    # Look for [Oracle] - pattern
    oracle_match = _ORACLE_PATTERN.search(message_content)
    if oracle_match:
        return oracle_match.group(1).strip()
    
    return message_content.strip()


def create_turn_based_traces(
    seeker_data: Dict[str, Any],
    llm_adapter: LLMAdapter,
    turn_workers: int = 4,
) -> List[Dict[str, Any]]:
    """Create turn-based traces similar to seeker.json structure.

    Turns within a conversation are synthesized in parallel (up to
    ``turn_workers`` concurrent LLM calls).

    Args:
        seeker_data: The loaded seeker conversation data.
        llm_adapter: LLM adapter for synthesis (stateless calls, thread-safe).
        turn_workers: Max parallel LLM calls per conversation.

    Returns:
        List of turn-based traces, ordered by turn index.
    """
    reasoning_history = seeker_data.get("reasoning_history", [])
    history = seeker_data.get("history", [])

    # Pre-filter messages for better performance
    reasoning_msgs = [
        msg for msg in reasoning_history
        if msg.get("role") == "assistant" and extract_reasoning_from_message(msg)
    ]

    assistant_msgs = [
        (i, msg) for i, msg in enumerate(history)
        if msg.get("role") == "assistant" and not msg.get("content", "").startswith("# SeekerAgent")
    ]

    # Collect all turns that need synthesis
    to_synthesize = []
    for i, (hist_idx, assistant_msg) in enumerate(assistant_msgs):
        question = assistant_msg.get("content", "").strip()
        oracle_answer = _find_oracle_answer(history, hist_idx)
        reasoning_text = None
        if i < len(reasoning_msgs):
            reasoning_text = extract_reasoning_from_message(reasoning_msgs[i])
        if reasoning_text:
            to_synthesize.append((i, reasoning_text, question, oracle_answer))

    logger.info("Processing %d reasoning traces for synthesis", len(to_synthesize))

    if not to_synthesize:
        return []

    # Synthesize all turns in parallel, preserving insertion order
    turns_dict: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(turn_workers, len(to_synthesize))) as executor:
        future_to_meta = {
            executor.submit(synthesize_reasoning_trace, reasoning_text, llm_adapter): (i, reasoning_text, question, oracle_answer)
            for i, reasoning_text, question, oracle_answer in to_synthesize
        }
        for future in as_completed(future_to_meta):
            i, reasoning_text, question, oracle_answer = future_to_meta[future]
            try:
                reasoning_trace = future.result()
                turns_dict[i] = {
                    "turn_index": i + 1,
                    "original_reasoning": reasoning_text,
                    "reasoning_trace": reasoning_trace,
                    "question": question,
                    "oracle_answer": oracle_answer,
                }
            except Exception as e:
                logger.warning(
                    "Failed to synthesize reasoning for question '%s': %s",
                    question[:50] + "..." if len(question) > 50 else question,
                    e,
                )

    # Restore turn order
    return [turns_dict[i] for i, *_ in to_synthesize if i in turns_dict]


def _find_oracle_answer(history: List[Dict[str, Any]], start_idx: int) -> str:
    """Find oracle answer starting from the given index.
    
    Args:
        history: List of conversation messages.
        start_idx: Starting index to search from.
        
    Returns:
        Oracle answer text or "Unknown" if not found.
    """
    for i in range(start_idx + 1, len(history)):
        msg = history[i]
        if msg.get("role") == "user" and "[Oracle]" in msg.get("content", ""):
            return extract_oracle_answer(msg.get("content", ""))
    return "Unknown"


def synthesize_conversation(
    seeker_path: Path,
    llm_config: LLMConfig,
    turn_workers: int = 4,
) -> Dict[str, Any]:
    """Synthesize all turns for one conversation and return the data dict.

    Unlike ``create_seeker_traces_file``, this does not write to disk —
    the caller decides where to persist the result.
    """
    seeker_data = load_seeker_conversation(seeker_path)
    llm_adapter = LLMAdapter(llm_config, save_history=False)
    turns = create_turn_based_traces(seeker_data, llm_adapter, turn_workers=turn_workers)
    return {
        "seeker_path": str(seeker_path),
        "agent_type": seeker_data.get("agent_type", "seeker"),
        "config": seeker_data.get("config", {}),
        "observability_mode": seeker_data.get("observability_mode"),
        "total_turns": len(turns),
        "turns": turns,
    }


def create_seeker_traces_file(
    input_path: Path,
    output_path: Path,
    llm_config: LLMConfig,
    turn_workers: int = 4,
) -> None:
    """Create seeker_traces.json from seeker.json file.

    Args:
        input_path: Path to input seeker.json file.
        output_path: Path where seeker_traces.json will be saved.
        llm_config: LLM configuration for synthesis.
        turn_workers: Max parallel LLM calls per conversation.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If processing fails.
    """
    # Load seeker conversation data
    seeker_data = load_seeker_conversation(input_path)

    # Create LLM adapter for synthesis
    llm_adapter = LLMAdapter(llm_config, save_history=False)

    # Create turn-based traces (turns parallelised internally)
    turns = create_turn_based_traces(seeker_data, llm_adapter, turn_workers=turn_workers)
    
    # Create output data structure similar to seeker.json
    output_data = {
        "agent_type": seeker_data.get("agent_type", "seeker"),
        "config": seeker_data.get("config", {}),
        "observability_mode": seeker_data.get("observability_mode"),
        "total_messages": seeker_data.get("total_messages"),
        "total_turns": len(turns),
        "history": turns
    }
    
    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
