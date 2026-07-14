"""Extract and score the belief state a Seeker maintains inside its reasoning.

This module supports the experiment that tests whether Chain-of-Thought acts as a
scaffold for *belief-state tracking* in the InfoGainMe game. For each turn of a CoT
conversation it parses the Seeker's ``<think>`` block with an extractor LLM
(Gemma-4-31B-IT in the paper) to recover the belief state the Seeker is holding:

  * ``constraints`` -- predicates it treats as established,
  * ``kept_candidates`` / ``excluded_candidates`` -- candidates it considers viable
    or ruled out,
  * ``believed_count`` -- the remaining-count it states, if any,
  * ``explicit_tracking`` -- whether it tracks the candidate set at all.

Because the Pruner records the *true* active set ``candidates_snapshot`` (= Omega_t)
at every turn, the extracted belief can be scored against ground truth: do the kept
candidates stay in Omega_t, are excluded ones really gone, and -- the fatal error --
is the true target ever ruled out while still active.

The module is read-only with respect to conversation files; it only calls the
extractor LLM and computes metrics.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.agents.llm_adapter import LLMAdapter
from src.agents.llm_config import LLMConfig
from src.prompts import get_belief_state_extraction_prompt
from src.utils.utils import llm_final_content, parse_first_json_object

logger = logging.getLogger(__name__)

_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

# The belief JSON is small; bound generation so a single call can't run away and
# trip the request timeout (the extractor occasionally over-generates otherwise).
_EXTRACT_MAX_TOKENS = 1024

# Belief-state JSON fields and their default empty values.
_BELIEF_FIELDS: dict[str, Any] = {
    "constraints": list,
    "kept_candidates": list,
    "excluded_candidates": list,
    "believed_count": lambda: None,
    "explicit_tracking": lambda: False,
}


# --------------------------------------------------------------------------- #
# Reading conversation artifacts
# --------------------------------------------------------------------------- #
def extract_think(content: str) -> str:
    """Return the text inside the first ``<think>`` block, or the raw content."""
    match = _THINK_PATTERN.search(content)
    if match:
        return match.group(1).strip()
    return content.strip()


def load_seeker_thinking(conversation_dir: Path) -> list[str]:
    """Return the per-turn reasoning text from ``seeker.json``.

    Args:
        conversation_dir: Directory holding ``seeker.json``.

    Returns:
        One think-block string per assistant turn, in order.
    """
    seeker = json.loads((conversation_dir / "seeker.json").read_text(encoding="utf-8"))
    entries = seeker.get("reasoning_history") or []
    return [extract_think(e.get("content", "")) for e in entries if e.get("role") == "assistant"]


def load_turns(conversation_dir: Path) -> list[dict[str, Any]]:
    """Load ``turns.jsonl`` (one record per game turn)."""
    text = (conversation_dir / "turns.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def resolve_target_label(conversation_dir: Path, turns: list[dict[str, Any]]) -> Optional[str]:
    """Best-effort recovery of the hidden target's label.

    Prefers ``metadata.json``; falls back to the final singleton active set.
    """
    meta_path = conversation_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # `target` is usually a nested dict {"id", "label", "attrs"}.
        target = meta.get("target")
        if isinstance(target, dict) and isinstance(target.get("label"), str):
            return target["label"]
        for key in ("target_label", "target", "target_name"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    if turns:
        last = turns[-1].get("candidates_snapshot") or []
        if len(last) == 1:
            return last[0]
    return None


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #
def _qa_history_text(turns: list[dict[str, Any]], up_to: int) -> str:
    """Render the question/answer history for turns strictly before ``up_to``."""
    lines = []
    for turn in turns[:up_to]:
        question = (turn.get("question") or {}).get("text", "")
        answer = (turn.get("answer") or {}).get("text", "")
        lines.append(f"Q{turn.get('turn_index', '?')}: {question} -> {answer}")
    return "\n".join(lines) if lines else "(no questions asked yet)"


def normalize_belief(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Coerce a parsed extractor response into the canonical belief schema."""
    belief: dict[str, Any] = {}
    raw = raw or {}
    for field, default in _BELIEF_FIELDS.items():
        value = raw.get(field, None)
        if field in ("constraints", "kept_candidates", "excluded_candidates"):
            belief[field] = [str(x) for x in value] if isinstance(value, list) else default()
        elif field == "believed_count":
            belief[field] = value if isinstance(value, int) else None
        else:  # explicit_tracking
            belief[field] = bool(value)
    return belief


def extract_belief_state(
    adapter: LLMAdapter,
    turns: list[dict[str, Any]],
    turn_index_zero_based: int,
    thinking_text: str,
) -> tuple[dict[str, Any], str, str]:
    """Call the extractor LLM to recover the belief state for one turn.

    Args:
        adapter: An ``LLMAdapter`` wrapping the extractor model (stateless use).
        turns: All turn records (for Q&A history).
        turn_index_zero_based: Index of the turn whose reasoning is ``thinking_text``.
        thinking_text: The Seeker's ``<think>`` content for this turn.

    Returns:
        A tuple ``(belief, user_prompt, raw_response)`` where ``belief`` is the
        normalized dict (always all five fields present), ``user_prompt`` is the
        exact user message sent to the extractor, and ``raw_response`` is the
        extractor's raw text before JSON parsing -- both kept for auditability.
    """
    history = _qa_history_text(turns, turn_index_zero_based)
    user = f"Question/answer history so far:\n{history}\n\nCurrent-turn reasoning:\n{thinking_text}"
    messages = [
        {"role": "system", "content": get_belief_state_extraction_prompt()},
        {"role": "user", "content": user},
    ]
    final = adapter.generate(
        messages=messages,
        stateless=True,
        add_to_history=False,
        temperature=0.0,
        max_tokens=_EXTRACT_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    raw_response = final if isinstance(final, str) else final[0]
    parsed = parse_first_json_object(llm_final_content(raw_response))
    return normalize_belief(parsed), user, raw_response


# --------------------------------------------------------------------------- #
# Scoring against ground truth Omega_t
# --------------------------------------------------------------------------- #
def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def match_to_pool(names: list[str], pool_index: dict[str, str]) -> set[str]:
    """Map extracted candidate names to canonical pool labels.

    Args:
        names: Names as written by the extractor.
        pool_index: Mapping of normalized label -> canonical label for Omega_0.

    Returns:
        Set of canonical labels that matched a pool member.
    """
    matched = set()
    for name in names:
        canonical = pool_index.get(_norm(name))
        if canonical is not None:
            matched.add(canonical)
    return matched


def compute_turn_metrics(
    belief: dict[str, Any],
    omega_t: set[str],
    target_label: Optional[str],
    pool_index: dict[str, str],
) -> dict[str, Any]:
    """Score a single-turn belief against the true active set Omega_t.

    Returns a dict of metrics; rate metrics are ``None`` when undefined (e.g. no
    named candidates of that polarity), so aggregation can ignore them cleanly.
    """
    kept = match_to_pool(belief["kept_candidates"], pool_index)
    excluded = match_to_pool(belief["excluded_candidates"], pool_index)
    n_named = len(kept | excluded)

    kept_precision = len(kept & omega_t) / len(kept) if kept else None
    zombie_kept_rate = len(kept - omega_t) / len(kept) if kept else None
    # Note: metric definitions are authoritatively recomputed in
    # analyze_belief_states.turn_metrics from the stored belief + omega_labels.
    excluded_correct_rate = len(excluded - omega_t) / len(excluded) if excluded else None

    target_in_active = target_label in omega_t if target_label else None
    fatal_target_excluded = (
        bool(target_label and target_in_active and target_label in excluded)
    )

    believed_count = belief["believed_count"]
    count_abs_error = abs(believed_count - len(omega_t)) if isinstance(believed_count, int) else None

    return {
        "omega_size": len(omega_t),
        "explicit_tracking": belief["explicit_tracking"],
        "n_constraints": len(belief["constraints"]),
        "n_named": n_named,
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "kept_precision": kept_precision,
        "zombie_kept_rate": zombie_kept_rate,
        "excluded_correct_rate": excluded_correct_rate,
        "believed_count": believed_count,
        "count_abs_error": count_abs_error,
        "fatal_target_excluded": fatal_target_excluded,
    }


def _mean(values: list[Any]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def summarize_conversation(turn_metrics: list[dict[str, Any]], ig_per_turn: Optional[float]) -> dict[str, Any]:
    """Aggregate per-turn metrics into a per-conversation summary."""
    return {
        "n_turns": len(turn_metrics),
        "ig_per_turn": ig_per_turn,
        "explicit_tracking_rate": _mean([t["explicit_tracking"] for t in turn_metrics]),
        "mean_kept_precision": _mean([t["kept_precision"] for t in turn_metrics]),
        "mean_zombie_kept_rate": _mean([t["zombie_kept_rate"] for t in turn_metrics]),
        "mean_excluded_correct_rate": _mean([t["excluded_correct_rate"] for t in turn_metrics]),
        "mean_count_abs_error": _mean([t["count_abs_error"] for t in turn_metrics]),
        "mean_n_named": _mean([t["n_named"] for t in turn_metrics]),
        "mean_n_constraints": _mean([t["n_constraints"] for t in turn_metrics]),
        "any_fatal_target_excluded": any(t["fatal_target_excluded"] for t in turn_metrics),
        "n_fatal_turns": sum(1 for t in turn_metrics if t["fatal_target_excluded"]),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_conversation(
    conversation_dir: Path,
    extractor_config: LLMConfig,
    max_turns: Optional[int] = None,
    save_io: bool = True,
) -> dict[str, Any]:
    """Extract and score belief states for every turn of one conversation.

    Args:
        conversation_dir: Directory with ``seeker.json`` and ``turns.jsonl``.
        extractor_config: Config for the extractor LLM.
        max_turns: If set, only the first ``max_turns`` turns are processed.
        save_io: If True, store the extractor's exact input (user prompt) and raw
            output per turn, plus the system prompt and model id once per record,
            for auditability/reproducibility.

    Returns:
        A record with per-turn beliefs+metrics and a conversation summary.
    """
    turns = load_turns(conversation_dir)
    thinking = load_seeker_thinking(conversation_dir)
    if not turns or not thinking:
        raise ValueError(f"No turns/thinking in {conversation_dir}")

    pool_labels = list(turns[0].get("candidates_snapshot") or [])
    pool_index = {_norm(lbl): lbl for lbl in pool_labels}
    target_label = resolve_target_label(conversation_dir, turns)

    n = min(len(turns), len(thinking))
    if max_turns is not None:
        n = min(n, max_turns)

    adapter = LLMAdapter(extractor_config, save_history=False)
    per_turn: list[dict[str, Any]] = []
    for i in range(n):
        omega_labels = list(turns[i].get("candidates_snapshot") or [])
        omega_t = set(omega_labels)
        belief, user_prompt, raw_response = extract_belief_state(adapter, turns, i, thinking[i])
        metrics = compute_turn_metrics(belief, omega_t, target_label, pool_index)
        # Constraints accumulate from prior answered questions; i = #questions asked
        # before this turn, so this lets the analyzer score belief completeness for
        # abstract reasoners that track constraints instead of naming candidates.
        metrics["n_questions_asked"] = i
        turn_record: dict[str, Any] = {
            "turn_index": turns[i].get("turn_index", i + 1),
            "info_gain": turns[i].get("info_gain"),
            "belief": belief,
            "metrics": metrics,
            # True active set Omega_t (labels) so each record is self-contained.
            "omega_labels": omega_labels,
        }
        if save_io:
            # Raw CoT for this turn as its own field, plus the exact extractor I/O:
            # user prompt (history + <think>) and raw output.
            turn_record["thinking"] = thinking[i]
            turn_record["llm_input"] = user_prompt
            turn_record["llm_output"] = raw_response
        per_turn.append(turn_record)

    ig_values = [t.get("info_gain") for t in turns[:n]]
    ig_per_turn = _mean(ig_values)
    record: dict[str, Any] = {
        "conversation_dir": str(conversation_dir),
        "target_label": target_label,
        "pool_size": len(pool_labels),
        "turns": per_turn,
        "summary": summarize_conversation([t["metrics"] for t in per_turn], ig_per_turn),
    }
    if save_io:
        # System prompt is constant across turns; store it once per record.
        record["extractor_model"] = extractor_config.model
        record["extractor_system_prompt"] = get_belief_state_extraction_prompt()
    return record
