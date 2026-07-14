"""Judge-model evaluation of Oracle / Pruner answers.

For each recorded turn in ``oracle.json`` / ``pruner.json`` we re-send the exact
same context the original agent saw to a larger "judge" model, then compare its
verdict to what the original agent produced.

How messages are reconstructed:

- ``oracle.json`` stores a canonical ``[system, user_1, assistant_1, ...,
  user_N, assistant_N]`` sequence. For turn T the judge sees
  ``history[:asst_T_index]`` — system + every prior user/assistant pair + the
  current user question.
- ``pruner.json`` stores stateless call pairs; the judge sees
  ``[system, user_T]`` per turn.

``turns.jsonl`` provides per-turn metadata only (question text, answer text,
target label). ``oracle.json`` / ``pruner.json`` hold the exact bytes the
agent saw, so we slice those directly.

``reasoning_history`` (raw output with ``<think>``/Harmony blocks) is copied
next to each turn so readers can inspect what the original agent *thought*
alongside its final verdict vs the judge's.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import ValidationError

from ..agents.llm_adapter import LLMAdapter, ContextLengthExceededError
from ..agents.llm_config import LLMConfig
from ..data_types import OracleResponse, PrunerResponse
from ..utils.utils import llm_final_content, parse_first_json_object

logger = logging.getLogger(__name__)

Kind = Literal["oracle", "pruner"]
OUTPUT_FILENAME = {"oracle": "oracle_judge_eval.json", "pruner": "pruner_judge_eval.json"}


_ORACLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "OracleResponse", "schema": OracleResponse.model_json_schema(), "strict": True},
}
_PRUNER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "PrunerResponse", "schema": PrunerResponse.model_json_schema(), "strict": True},
}


# ---------------------------------------------------------------------------
# Conversation loader
# ---------------------------------------------------------------------------


@dataclass
class ConversationContext:
    conv_dir: Path
    target_id: str
    target_label: str
    experiment: str
    seeker_model: str
    oracle_model: str
    pruner_model: str
    turns: list[dict[str, Any]]
    oracle_history: Optional[list[dict[str, str]]] = None
    oracle_reasoning_history: Optional[list[dict[str, str]]] = None
    pruner_history: Optional[list[dict[str, str]]] = None
    pruner_reasoning_history: Optional[list[dict[str, str]]] = None


def load_conversation(conv_dir: Path) -> ConversationContext:
    meta = json.loads((conv_dir / "metadata.json").read_text())
    target = meta.get("target", {})
    models = meta.get("config", {}).get("models", {})
    with (conv_dir / "turns.jsonl").open() as f:
        turns = [json.loads(line) for line in f if line.strip()]

    def _maybe_load(name: str) -> tuple[Optional[list], Optional[list]]:
        path = conv_dir / name
        if not path.exists():
            return None, None
        data = json.loads(path.read_text())
        return data.get("history"), data.get("reasoning_history")

    oh, orh = _maybe_load("oracle.json")
    ph, prh = _maybe_load("pruner.json")
    return ConversationContext(
        conv_dir=conv_dir,
        target_id=target.get("id", ""),
        target_label=target.get("label", ""),
        experiment=meta.get("config", {}).get("experiment_name", conv_dir.parent.parent.name),
        seeker_model=models.get("seeker", ""),
        oracle_model=models.get("oracle", ""),
        pruner_model=models.get("pruner", ""),
        turns=turns,
        oracle_history=oh,
        oracle_reasoning_history=orh,
        pruner_history=ph,
        pruner_reasoning_history=prh,
    )


# ---------------------------------------------------------------------------
# Assistant-reply parsing (Qwen/Gemma original output)
# ---------------------------------------------------------------------------


def parse_oracle_reply(raw: str) -> dict[str, Any]:
    cleaned = llm_final_content(raw)
    try:
        p = OracleResponse.model_validate_json(cleaned)
        return {"answer": p.answer, "rationale": p.rationale, "game_over": p.game_over}
    except ValidationError:
        obj = parse_first_json_object(cleaned) or {}
        return {"answer": obj.get("answer", ""), "rationale": obj.get("rationale", ""),
                "game_over": bool(obj.get("game_over", False))}


def parse_pruner_reply(raw: str) -> dict[str, Any]:
    cleaned = llm_final_content(raw)
    try:
        p = PrunerResponse.model_validate_json(cleaned)
        return {"keep_labels": list(p.keep_labels), "rationale": p.rationale}
    except ValidationError:
        obj = parse_first_json_object(cleaned) or {}
        keep = obj.get("keep_labels") or []
        return {"keep_labels": [str(l) for l in keep if isinstance(l, (str, int, float))],
                "rationale": obj.get("rationale", "")}


def _parse_active_from_user(user_content: str) -> set[str]:
    try:
        lines = user_content.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("Active candidates"))
    except StopIteration:
        return set()
    buf: list[str] = []
    for l in lines[start + 1:]:
        if l.startswith("TURN:") or not l.strip():
            break
        buf.append(l)
    return {c.strip() for c in " ".join(buf).split(",") if c.strip()}


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------


def build_judge_adapter(
    model: str,
    base_url: Optional[str],
    api_key: str,
    temperature: Optional[float] = None,
    timeout: float = 300.0,
    extra: Optional[dict[str, Any]] = None,
) -> LLMAdapter:
    """Return a carrier adapter. ``_call_judge`` builds a fresh per-call adapter
    from its config so ``reasoning_history`` is never shared across threads."""
    cfg = LLMConfig(model=model, api_key=api_key, base_url=base_url,
                    timeout=timeout, temperature=temperature,
                    extra=extra or {})
    return LLMAdapter(cfg, save_history=False, save_reasoning=False)


def _call_judge(
    adapter: LLMAdapter,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
) -> tuple[str, Optional[str]]:
    """One judge call with its own adapter — thread-safe under parallel turns.

    LLMAdapter mutates ``_reasoning_history`` on each ``generate()``; sharing
    one adapter across workers causes a race where reading ``[-1]`` after
    generate() can return another thread's reasoning. Building a fresh adapter
    per call is cheap (no persistent OpenAI client) and makes each call
    effectively local.
    """
    worker = LLMAdapter(adapter.config, save_history=False, save_reasoning=True)
    cleaned = worker.generate(
        messages=messages, stateless=True,
        response_format=response_format, add_to_history=False,
        # Judge inputs can exceed the judge model's context window. The retry
        # loop never recovers from that — fail fast so the caller marks the
        # turn skipped instead of burning ~45 min of useless backoff.
        bail_on_context_length=True,
    )
    raw = worker.reasoning_history[-1]["content"] if worker.reasoning_history else None
    return cleaned, raw


# ---------------------------------------------------------------------------
# Per-turn evaluation
# ---------------------------------------------------------------------------


def _reasoning_at(reasoning_history: Optional[list[dict[str, str]]], idx: int) -> Optional[str]:
    """``reasoning_history`` mirrors ``history`` 1:1 when both are saved."""
    if reasoning_history is None or idx >= len(reasoning_history):
        return None
    return reasoning_history[idx].get("content")


def _build_preceding_user_map(history: list[dict[str, str]]) -> dict[int, int]:
    """For each assistant index, the index of the immediately preceding user
    message (ignoring system). Single pass → O(N)."""
    out: dict[int, int] = {}
    last_user = None
    for i, m in enumerate(history):
        role = m.get("role")
        if role == "user":
            last_user = i
        elif role == "assistant" and last_user is not None:
            out[i] = last_user
            last_user = None  # each user pairs with exactly one assistant
    return out


def _assistant_indices(history: list[dict[str, str]]) -> list[int]:
    return [i for i, m in enumerate(history) if m.get("role") == "assistant"]


def _judge_oracle_turn(
    turn_idx: int,
    turn: dict[str, Any],
    history: list[dict[str, str]],
    reasoning_history: Optional[list[dict[str, str]]],
    asst_idx: int,
    user_idx: Optional[int],
    adapter: LLMAdapter,
) -> dict[str, Any]:
    if user_idx is None:
        return {"turn_index": turn.get("turn_index", turn_idx + 1),
                "question": turn.get("question", {}).get("text", ""),
                "skipped": "no preceding user message"}

    qwen = parse_oracle_reply(history[asst_idx].get("content", ""))
    qwen_raw = _reasoning_at(reasoning_history, asst_idx)
    messages = history[: user_idx + 1]

    try:
        cleaned, judge_raw = _call_judge(adapter, messages, _ORACLE_RESPONSE_FORMAT)
    except ContextLengthExceededError as exc:
        return {"turn_index": turn.get("turn_index", turn_idx + 1),
                "question": turn.get("question", {}).get("text", ""),
                "qwen_answer": qwen["answer"], "qwen_rationale": qwen["rationale"],
                "qwen_raw": qwen_raw,
                "skipped": f"context_length_exceeded: {exc}"}
    except Exception as exc:
        logger.warning("Judge oracle call failed (turn %d): %s", turn_idx + 1, exc)
        return {"turn_index": turn.get("turn_index", turn_idx + 1),
                "question": turn.get("question", {}).get("text", ""),
                "qwen_answer": qwen["answer"], "qwen_rationale": qwen["rationale"],
                "qwen_raw": qwen_raw, "judge_error": str(exc), "answer_match": None}

    judge = parse_oracle_reply(cleaned)
    return {
        "turn_index": turn.get("turn_index", turn_idx + 1),
        "question": turn.get("question", {}).get("text", ""),
        "qwen_answer": qwen["answer"], "qwen_rationale": qwen["rationale"],
        "qwen_game_over": qwen["game_over"], "qwen_raw": qwen_raw,
        "judge_answer": judge["answer"], "judge_rationale": judge["rationale"],
        "judge_game_over": judge["game_over"], "judge_raw": judge_raw,
        "answer_match": qwen["answer"] == judge["answer"],
        "game_over_match": qwen["game_over"] == judge["game_over"],
    }


def _judge_pruner_turn(
    turn_idx: int,
    turn: dict[str, Any],
    history: list[dict[str, str]],
    reasoning_history: Optional[list[dict[str, str]]],
    asst_idx: int,
    user_idx: Optional[int],
    system_msg: Optional[dict[str, str]],
    target_label: str,
    adapter: LLMAdapter,
) -> dict[str, Any]:
    if user_idx is None or system_msg is None:
        return {"turn_index": turn.get("turn_index", turn_idx + 1),
                "question": turn.get("question", {}).get("text", ""),
                "skipped": "missing system or user message"}

    qwen = parse_pruner_reply(history[asst_idx].get("content", ""))
    qwen_raw = _reasoning_at(reasoning_history, asst_idx)
    user_msg = history[user_idx]
    active = _parse_active_from_user(user_msg.get("content", ""))
    messages = [system_msg, user_msg]

    try:
        cleaned, judge_raw = _call_judge(adapter, messages, _PRUNER_RESPONSE_FORMAT)
    except ContextLengthExceededError as exc:
        return {"turn_index": turn.get("turn_index", turn_idx + 1),
                "question": turn.get("question", {}).get("text", ""),
                "answer": turn.get("answer", {}).get("text", ""),
                "qwen_keep_labels_count": len(qwen["keep_labels"]),
                "skipped": f"context_length_exceeded: {exc}"}
    except Exception as exc:
        logger.warning("Judge pruner call failed (turn %d): %s", turn_idx + 1, exc)
        return {"turn_index": turn.get("turn_index", turn_idx + 1),
                "question": turn.get("question", {}).get("text", ""),
                "answer": turn.get("answer", {}).get("text", ""),
                "qwen_keep_labels_count": len(qwen["keep_labels"]),
                "judge_error": str(exc)}

    judge = parse_pruner_reply(cleaned)
    qk = set(qwen["keep_labels"]) & active if active else set(qwen["keep_labels"])
    jk = set(judge["keep_labels"]) & active if active else set(judge["keep_labels"])
    inter, union = qk & jk, qk | jk
    return {
        "turn_index": turn.get("turn_index", turn_idx + 1),
        "question": turn.get("question", {}).get("text", ""),
        "answer": turn.get("answer", {}).get("text", ""),
        "active_count": len(active),
        "qwen_keep_labels_count": len(qk),
        "judge_keep_labels_count": len(jk),
        "intersection": len(inter),
        "union": len(union),
        "jaccard": len(inter) / len(union) if union else 1.0,
        "precision_qwen": len(inter) / len(qk) if qk else 0.0,
        "recall_qwen": len(inter) / len(jk) if jk else 0.0,
        "qwen_kept_target": (target_label in qk) if target_label else None,
        "judge_kept_target": (target_label in jk) if target_label else None,
        "qwen_rationale": qwen["rationale"],
        "judge_rationale": judge["rationale"],
        "qwen_raw": qwen_raw,
        "judge_raw": judge_raw,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _oracle_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [t for t in turns if "judge_error" not in t and "skipped" not in t]
    matches = sum(1 for t in ok if t.get("answer_match") is True)
    conf = {"YY": 0, "NN": 0, "YN": 0, "NY": 0, "other": 0}
    for t in ok:
        qa, ja = t.get("qwen_answer", ""), t.get("judge_answer", "")
        if not qa or not ja:
            conf["other"] += 1
            continue
        key = ("Y" if qa.startswith("Yes") else "N") + ("Y" if ja.startswith("Yes") else "N")
        conf[key] += 1
    return {
        "n_turns": len(turns),
        "n_ok": len(ok),
        "n_matches": matches,
        "agreement": matches / len(ok) if ok else 0.0,
        "n_errors": sum(1 for t in turns if "judge_error" in t),
        "n_skipped": sum(1 for t in turns if "skipped" in t),
        "yes_no_confusion": conf,
    }


def _pruner_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [t for t in turns if "judge_error" not in t and "skipped" not in t]
    return {
        "n_turns": len(turns),
        "n_ok": len(ok),
        "n_errors": sum(1 for t in turns if "judge_error" in t),
        "n_skipped": sum(1 for t in turns if "skipped" in t),
        "mean_jaccard": sum(t["jaccard"] for t in ok) / len(ok) if ok else 0.0,
        "n_target_removed_by_qwen": sum(1 for t in ok if t.get("qwen_kept_target") is False),
        "n_target_removed_by_judge": sum(1 for t in ok if t.get("judge_kept_target") is False),
    }


# ---------------------------------------------------------------------------
# Conversation-level evaluation (unified oracle/pruner dispatch)
# ---------------------------------------------------------------------------


def evaluate_conversation(
    kind: Kind,
    ctx: ConversationContext,
    adapter: LLMAdapter,
    turn_workers: int = 4,
) -> dict[str, Any]:
    if kind == "oracle":
        history = ctx.oracle_history
        reasoning = ctx.oracle_reasoning_history
        agent_model = ctx.oracle_model
        summary_fn = _oracle_summary
    else:
        history = ctx.pruner_history
        reasoning = ctx.pruner_reasoning_history
        agent_model = ctx.pruner_model
        summary_fn = _pruner_summary
    if history is None:
        raise ValueError(f"{kind}.json missing under {ctx.conv_dir}")

    preceding = _build_preceding_user_map(history)
    asst_idx = _assistant_indices(history)
    system_msg = next((m for m in history if m.get("role") == "system"), None)

    def _one(i: int) -> dict[str, Any]:
        aidx = asst_idx[i]
        uidx = preceding.get(aidx)
        turn = ctx.turns[i] if i < len(ctx.turns) else {}
        if kind == "oracle":
            return _judge_oracle_turn(i, turn, history, reasoning, aidx, uidx, adapter)
        return _judge_pruner_turn(i, turn, history, reasoning, aidx, uidx,
                                  system_msg, ctx.target_label, adapter)

    n = min(len(asst_idx), len(ctx.turns))
    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(turn_workers, max(1, n))) as ex:
        futures = {ex.submit(_one, i): i for i in range(n)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                logger.exception("Turn %d failed: %s", i, exc)
                results[i] = {"turn_index": i + 1, "judge_error": str(exc)}

    turns_out = [results[i] for i in sorted(results)]
    model_key = f"{kind}_model"
    return {
        "judge_model": adapter.config.model,
        "judge_base_url": adapter.config.base_url,
        "target_id": ctx.target_id, "target_label": ctx.target_label,
        "experiment": ctx.experiment, model_key: agent_model,
        "turns": turns_out, "summary": summary_fn(turns_out),
    }


# ---------------------------------------------------------------------------
# Idempotent one-shot wrapper
# ---------------------------------------------------------------------------


def _already_done(out_path: Path, judge_model: str) -> bool:
    if not out_path.exists():
        return False
    try:
        return json.loads(out_path.read_text()).get("judge_model") == judge_model
    except (json.JSONDecodeError, OSError):
        # Corrupt / unreadable → regenerate.
        return False


def run_eval(
    kind: Kind,
    conv_dir: Path,
    adapter: LLMAdapter,
    turn_workers: int = 4,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Evaluate one conversation. Returns ``(output_path, was_skipped)``."""
    out = conv_dir / OUTPUT_FILENAME[kind]
    if not overwrite and _already_done(out, adapter.config.model):
        return out, True
    result = evaluate_conversation(
        kind, load_conversation(conv_dir), adapter, turn_workers,
    )
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return out, False


__all__ = [
    "Kind",
    "OUTPUT_FILENAME",
    "ConversationContext",
    "load_conversation",
    "parse_oracle_reply",
    "parse_pruner_reply",
    "build_judge_adapter",
    "evaluate_conversation",
    "run_eval",
]
