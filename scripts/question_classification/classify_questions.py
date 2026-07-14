#!/usr/bin/env python3
"""Classify seeker questions from Info Gainme benchmark runs.

For each seeker turn, the classifier labels:

    question_type_rationale    one sentence explaining the primary class
    question_type              semantic | lexical | direct_guess | malformed | other
    subclasses_rationale       one sentence explaining the sub-class tag(s) (may be empty)
    subclasses                 list of snake_case tags (may be empty — tags are orthogonal
                               and a single question can carry more than one, e.g.
                               ["comparative", "quantitative_threshold"])
    redundancy                 none | exact_duplicate | semantic_equivalent | strictly_implied
    redundant_with_turn        1-indexed turn this one duplicates, or null

Rationale fields always come BEFORE the label they justify so the model
reasons chain-of-thought style inside its structured output.

This script issues ONE LLM request per conversation: it feeds the whole
Q&A history of a game and gets back an array of classifications, one per
turn. That's roughly 10× fewer requests than classifying turn-by-turn,
and redundancy detection stays confident because the model already sees
every prior turn in context.

Usage
-----
    # Full sweep (default)
    python3 scripts/question_classification/classify_questions.py \
        --max-concurrency 32

    # Cap per stratum
    python3 scripts/question_classification/classify_questions.py --per-stratum 30

    # Single conversation (debug)
    python3 scripts/question_classification/classify_questions.py \
        --seeker-file outputs/models/.../conversations/.../seeker.json

Output
------
    outputs/question_classification/
        <experiment>/<target>/classification.json   per-conversation
        summary.json                                 aggregated counts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("classify_questions")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    DIRECT_GUESS = "direct_guess"
    MALFORMED = "malformed"
    OTHER = "other"


class RedundancyType(str, Enum):
    NONE = "none"
    EXACT_DUPLICATE = "exact_duplicate"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    STRICTLY_IMPLIED = "strictly_implied"


class QuestionClassification(BaseModel):
    """One classification per seeker turn.

    Field order is deliberate: the echoed ``question`` anchors the entry to a
    specific input turn (so we can detect the model dropping / reordering /
    inventing turns), then every rationale precedes its label so the model
    generates the justification first and commits to a class after.
    """

    question: str = Field(
        description=(
            "The seeker's question for this turn, echoed VERBATIM from the input. "
            "Used as an integrity anchor to verify the classification lines up with "
            "the correct turn — do not paraphrase, summarise, or translate."
        ),
    )
    question_type_rationale: str = Field(
        default="",
        description="One short sentence on why this question_type fits.",
    )
    question_type: QuestionType
    subclasses_rationale: str = Field(
        default="",
        description="One short sentence on why these sub-class tags fit. Empty when subclasses is [].",
    )
    subclasses: list[str] = Field(
        default_factory=list,
        description=(
            "Orthogonal snake_case tags supplementing question_type. A single question "
            "may carry more than one (e.g. ['comparative', 'quantitative_threshold']). "
            "Empty list when no sharper label applies."
        ),
    )
    redundancy_rationale: str = Field(
        default="",
        description=(
            "One short sentence explaining why this redundancy label fits. "
            "When redundancy is not 'none', mention which earlier turn makes this one redundant."
        ),
    )
    redundancy: RedundancyType


class BatchClassification(BaseModel):
    """Wrapper for one LLM response: one classification per turn, in order."""

    classifications: list[QuestionClassification]


# ---------------------------------------------------------------------------
# turns.jsonl parsing
# ---------------------------------------------------------------------------


@dataclass
class TurnQA:
    turn: int
    question: str
    oracle_answer: str


def extract_turns(turns_path: Path) -> list[TurnQA]:
    """Read turns from turns.jsonl — one JSON object per line."""
    turns: list[TurnQA] = []
    for line in turns_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        turns.append(TurnQA(
            turn=t["turn_index"],
            question=t["question"]["text"],
            oracle_answer=t["answer"]["text"],
        ))
    return turns


# ---------------------------------------------------------------------------
# Experiment discovery + stratified sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    model_slug: str
    experiment: str
    domain: str       # geo / objects / diseases
    mode: str         # fo / po
    cot: bool


def _slug(name: str) -> str:
    """Model name → output-dir slug (the triple uses '/' replaced by '-')."""
    return name.replace("/", "-")


def _triple_parts(dir_name: str) -> Optional[tuple[str, str, str]]:
    """Parse 's_<seeker>__o_<oracle>__p_<pruner>' → (seeker, oracle, pruner).

    Returns None if the dir name isn't triple-shaped. Segments are matched
    EXACTLY by the caller, so 'Nemotron-Cascade-8B' never matches
    'Nemotron-Cascade-8B-Thinking'.
    """
    if not dir_name.startswith("s_"):
        return None
    rest = dir_name[2:]
    if "__o_" not in rest or "__p_" not in rest:
        return None
    seeker, rest2 = rest.split("__o_", 1)
    oracle, pruner = rest2.split("__p_", 1)
    return seeker, oracle, pruner


def _parse_experiment(model_slug: str, experiment: str) -> Stratum | None:
    exp = experiment.lower()
    for prefix in ("geo", "objects", "diseases"):
        if exp.startswith(prefix):
            domain = prefix
            break
    else:
        return None
    mode = "po" if "_po" in exp else "fo"
    # "_no_cot" contains "_cot" — exclude it explicitly.
    has_no_cot = exp.endswith("_no_cot") or "_no_cot_" in exp
    has_cot = exp.endswith("_cot") or "_cot_" in exp
    return Stratum(model_slug, experiment, domain, mode, has_cot and not has_no_cot)


_RUN_SUFFIX_RE = re.compile(r"_run(\d+)$")


def _conversation_run_index(turns_path: Path) -> Optional[int]:
    """Extract run_index from ``conversations/<target>_runNN/turns.jsonl``."""
    m = _RUN_SUFFIX_RE.search(turns_path.parent.name)
    return int(m.group(1)) if m else None


def discover_conversations(
    outputs_root: Path,
    run_index: Optional[int] = None,
    seekers: Optional[set[str]] = None,
    oracle: Optional[str] = None,
) -> dict[Stratum, list[Path]]:
    """Walk outputs/models/<triple>/<experiment>/conversations/.

    The path list inside each stratum is sorted by conversation directory name
    (= ``<target>_runNN``), which matches the ordering used by run.csv after
    target_id sorting. ``sample_indices`` (in :func:`stratified_sample`) then
    consumes deterministic positions in that sorted list.

    ``run_index`` (e.g. ``1``) keeps only ``_runNN`` matching, so analysis can
    restrict to first-run-per-target — same convention as ``scripts/judge_eval``.
    """
    buckets: dict[Stratum, list[Path]] = defaultdict(list)
    models_root = outputs_root / "models"
    if not models_root.exists():
        return buckets
    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        if seekers is not None or oracle is not None:
            parts = _triple_parts(model_dir.name)
            if parts is None:
                continue
            s, o, _ = parts
            if seekers is not None and s not in seekers:
                continue
            if oracle is not None and o != oracle:
                continue
        for exp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            stratum = _parse_experiment(model_dir.name, exp_dir.name)
            if stratum is None:
                continue
            paths = sorted(exp_dir.glob("conversations/*/turns.jsonl"),
                           key=lambda p: p.parent.name)
            if run_index is not None:
                paths = [p for p in paths if _conversation_run_index(p) == run_index]
            buckets[stratum].extend(paths)
    return buckets


def stratified_sample(
    buckets: dict[Stratum, list[Path]],
    per_stratum: int | None,
    rng: random.Random,
    sample_indices: Optional[list[int]] = None,
) -> list[tuple[Stratum, Path]]:
    """Pick conversations from each bucket.

    Priority of selection mode (highest wins):
      - ``sample_indices`` (e.g. ``[0,10,20,...]``): deterministic positions in
        the sorted path list of each stratum. Mirrors ``judge_eval`` so the same
        targets are evaluated across analysis pipelines.
      - ``per_stratum``: random sample of N paths per stratum (legacy).
      - both ``None``: full sweep.
    """
    picks: list[tuple[Stratum, Path]] = []
    for stratum, paths in buckets.items():
        if not paths:
            continue
        if sample_indices is not None:
            chosen = [paths[i] for i in sample_indices if 0 <= i < len(paths)]
        elif per_stratum is None:
            chosen = list(paths)
        else:
            chosen = rng.sample(paths, min(per_stratum, len(paths)))
        picks.extend((stratum, p) for p in chosen)
    return picks


# ---------------------------------------------------------------------------
# LLM classification (one request per whole conversation)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a taxonomist labelling yes/no questions asked by a seeker agent playing an information-gain guessing game.

The seeker tries to identify a secret target (a city, an object, or a disease) by asking yes/no questions. You will receive the FULL Q&A history of one game and you must classify EVERY turn, in order.

For each turn, produce SEVEN fields IN THIS EXACT ORDER:

1. **question** — the seeker's question for this turn, copied VERBATIM from the input (no paraphrasing, no trimming, no translation). This anchors the entry to the correct turn.
2. **question_type_rationale** — one short sentence explaining why your question_type fits.
3. **question_type** — one of:
   - `semantic`     uses domain knowledge (location, symptoms, attributes, taxonomy). E.g. "Is it in Asia?", "Does it affect the brain?".
   - `lexical`      asks about the NAME/STRING only, no domain meaning. E.g. "Does the name start with A?".
   - `direct_guess` names a specific candidate. E.g. "Is it Mumbai?", "Is it dengue?".
   - `malformed`    not a valid yes/no question: bare statement, open-ended, echoing the Oracle/Computer, etc.
   - `other`        genuinely does not fit any of the above (rare).
4. **subclasses_rationale** — one short sentence justifying the chosen sub-class tag(s). Use `""` when `subclasses` is an empty list.
5. **subclasses** — a LIST of snake_case sub-tags, orthogonal to question_type. The list may be empty (`[]`) when nothing sharper applies, or contain one OR MORE tags when multiple apply (e.g. a question that is both a comparison AND a numeric threshold: `["comparative", "quantitative_threshold"]`). Do not duplicate tags. Common tags (you may coin new ones):
   - `hierarchical_category`    broad taxonomic level ("Is it an animal?")
   - `fine_grained_category`    narrow taxonomic level ("Is it a citrus fruit?")
   - `comparative`              any comparison to another entity — by size, age, frequency, etc. ("bigger than a microwave", "older than Rome")
   - `relational`               defined relative to another entity ("near India?", "borders France?")
   - `quantitative_threshold`   numeric threshold ("population > 1M?", "more than 4 legs?")
   - `compound_predicate`       conjoined conditions ("in Asia AND coastal?")
   - `meta_strategy`            about the game state, not the target
   - `statement`, `open_ended`  sub-types of `malformed`
6. **redundancy_rationale** — one short sentence explaining why this redundancy label fits. When redundancy is not `none`, mention which earlier turn (by number) makes this one redundant.
7. **redundancy** — one of:
   - `none`                brings new information.
   - `exact_duplicate`     literally the same wording as an earlier turn.
   - `semantic_equivalent` same question in different words (answer is determined).
   - `strictly_implied`    answer is already determined by prior Q&A.
   Redundancy for turn K is measured ONLY against turns 1..K-1 — never use information from later turns.

Return a JSON object with a single key `classifications` containing an array with EXACTLY ONE entry per turn, in the same order as the turns shown. No prose, no markdown fences.

### Worked example (3-turn game, diseases)

    Turn 1: Q "Is the target disease primarily affecting the respiratory system?" / A "Yes"
    Turn 2: Q "Is it bigger than the common cold in mortality?" / A "Yes"
    Turn 3: Q "Is it the common cold?" / A "No"

Expected output:

    {
      "classifications": [
        {
          "question": "Is the target disease primarily affecting the respiratory system?",
          "question_type_rationale": "Uses medical-domain knowledge to categorise the target by affected organ system.",
          "question_type": "semantic",
          "subclasses_rationale": "Asks about a broad organ-system category, a high-level taxonomic bucket.",
          "subclasses": ["hierarchical_category"],
          "redundancy_rationale": "This is the first turn, so no prior Q&A exists to make it redundant.",
          "redundancy": "none"
        },
        {
          "question": "Is it bigger than the common cold in mortality?",
          "question_type_rationale": "Probes a domain attribute (mortality) by comparison against a reference disease.",
          "question_type": "semantic",
          "subclasses_rationale": "Frames the attribute as a comparison with another entity, and the attribute itself is a numeric-like magnitude.",
          "subclasses": ["comparative", "quantitative_threshold"],
          "redundancy_rationale": "Asks about mortality magnitude, which was not covered by turn 1.",
          "redundancy": "none"
        },
        {
          "question": "Is it the common cold?",
          "question_type_rationale": "Names a specific candidate rather than probing an attribute.",
          "question_type": "direct_guess",
          "subclasses_rationale": "",
          "subclasses": [],
          "redundancy_rationale": "Turn 3 asks about the common cold, but turn 2 already established the target has higher mortality than the common cold, so this guess is already ruled out.",
          "redundancy": "strictly_implied"
        }
      ]
    }

The `question` field MUST match the input turn's Q text character-for-character. If it does not, the classification will be flagged as misaligned and rejected.
"""


def build_user_message(turns: list[TurnQA], domain: str) -> str:
    lines = [f"Domain: {domain}", "", f"Game turns ({len(turns)} total — classify every one, in order):"]
    for t in turns:
        lines.append(f"  Turn {t.turn}: {t.question}")
    lines.append("")
    lines.append(
        f"Return a JSON object with key `classifications` — an array of EXACTLY {len(turns)} entries in turn order."
    )
    return "\n".join(lines)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Loose comparison for the question-echo integrity check: strips
    surrounding whitespace and collapses internal whitespace runs, so a
    stray extra space or newline does not trip the mismatch check."""
    return _WS_RE.sub(" ", (s or "")).strip().lower()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(raw: str) -> str:
    """Strip <think> blocks + markdown fences, isolate the outer JSON object."""
    cleaned = _THINK_RE.sub("", raw).strip()
    fence = _CODE_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if cleaned.startswith("{"):
        return cleaned
    match = _JSON_OBJECT_RE.search(cleaned)
    return match.group(0) if match else cleaned


async def classify_conversation_batch(
    client: AsyncOpenAI,
    model: str,
    turns: list[TurnQA],
    domain: str,
    thinking: bool,
    semaphore: asyncio.Semaphore,
) -> tuple[list[QuestionClassification], dict[str, str]]:
    """Single LLM request: classify every turn in the conversation at once.

    Returns (classifications, raw_response) where raw_response is the model's
    full output (content + reasoning_content), saved by default for auditing.
    """
    # vLLM aceita só UM mecanismo de structured output por vez:
    # `guided_json` (extra_body) é mais estrito que `response_format={"type":"json_object"}`,
    # então passamos só o primeiro. (Versões antigas do vLLM toleravam ambos; as novas — ex: Qwen3-235B —
    # retornam BadRequestError "You can only use one kind of guided decoding".)
    extra_body: dict[str, Any] = {"guided_json": BatchClassification.model_json_schema()}
    # SEMPRE manda a flag explícita: alguns servidores vLLM (ex.: gemma4) sobem
    # com `enable_thinking: true` como default do chat-template, então NÃO
    # mandar nada deixa o thinking LIGADO mesmo com --no-thinking, e o reasoning
    # conflita com o `guided_json` (structured output) → batch inválido.
    # Chave correta é `enable_thinking` (chat templates Qwen3 + gemma4;
    # gpt-oss ignora chat_template_kwargs e usa reasoning_effort).
    extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}

    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(turns, domain)},
            ],
            extra_body=extra_body,
        )

    msg = completion.choices[0].message
    content = msg.content or ""
    # Different vLLM reasoning-parsers use different field names:
    #   gpt-oss (openai_gptoss), qwen3, ... → `reasoning_content`
    #   gemma4                              → `reasoning`
    # Capture both so raw_response is model-agnostic.
    reasoning_content = getattr(msg, "reasoning_content", "") or ""
    reasoning = getattr(msg, "reasoning", "") or ""
    raw = content or reasoning_content or reasoning or ""
    # Full model output, persisted by default (auditing / model comparison).
    raw_response = {
        "content": content,
        "reasoning_content": reasoning_content,
        "reasoning": reasoning,
    }
    payload = _extract_json_payload(raw)
    try:
        batch = BatchClassification.model_validate_json(payload)
    except Exception as e:  # noqa: BLE001
        err = RuntimeError(f"{type(e).__name__}: {e} | raw={raw[:400]!r}")
        err.raw_response = raw_response  # type: ignore[attr-defined]
        raise err from e

    got, want = len(batch.classifications), len(turns)
    if got != want:
        err = RuntimeError(f"model returned {got} classifications, expected {want}")
        err.raw_response = raw_response  # type: ignore[attr-defined]
        raise err

    return batch.classifications, raw_response


async def classify_conversation(
    turns_path: Path,
    stratum: Stratum,
    client: AsyncOpenAI,
    model: str,
    thinking: bool,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    turns = extract_turns(turns_path)

    raw_response: dict[str, str] = {"content": "", "reasoning_content": ""}
    try:
        classifications, raw_response = await classify_conversation_batch(
            client, model, turns, stratum.domain, thinking, semaphore
        )
        turn_payloads = []
        for t, cls in zip(turns, classifications):
            payload: dict[str, Any] = {
                "turn": t.turn,
                "question": t.question,
                "oracle_answer": t.oracle_answer,
                "classification": cls.model_dump(mode="json"),
            }
            if _norm(cls.question) != _norm(t.question):
                payload["question_echo_warning"] = (
                    f"echoed={cls.question!r} differs from input={t.question!r}"
                )
            turn_payloads.append(payload)
    except Exception as e:  # noqa: BLE001 — bubble up as per-conversation error
        raw_response = getattr(e, "raw_response", raw_response)
        turn_payloads = [
            {
                "turn": t.turn,
                "question": t.question,
                "oracle_answer": t.oracle_answer,
                "classification": {"error": f"{type(e).__name__}: {e}"},
            }
            for t in turns
        ]

    return {
        "turns_path": str(turns_path),
        "experiment": stratum.experiment,
        "model_slug": stratum.model_slug,
        "domain": stratum.domain,
        "mode": stratum.mode,
        "cot": stratum.cot,
        "target": turns_path.parent.name,
        "num_turns": len(turns),
        "turns": turn_payloads,
        "analysis_model": model,
        "raw_response": raw_response,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(conv_results: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    redundancy_counts: Counter[str] = Counter()
    subclass_counts: Counter[str] = Counter()
    subclass_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    per_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    errors = 0
    total_turns = 0

    for conv in conv_results:
        stratum_key = f"{conv['domain']}/{conv['mode']}/{'cot' if conv['cot'] else 'no_cot'}"
        for turn in conv["turns"]:
            total_turns += 1
            cls = turn["classification"]
            if "error" in cls:
                errors += 1
                continue
            qt = cls["question_type"]
            type_counts[qt] += 1
            per_stratum[stratum_key][qt] += 1
            redundancy_counts[cls["redundancy"]] += 1
            for tag in cls.get("subclasses") or []:
                label = str(tag).strip().lower()
                if not label:
                    continue
                subclass_counts[label] += 1
                subclass_by_type[qt][label] += 1

    return {
        "total_conversations": len(conv_results),
        "total_turns": total_turns,
        "errors": errors,
        "question_type_counts": dict(type_counts),
        "redundancy_counts": dict(redundancy_counts),
        "per_stratum_question_type_counts": {k: dict(v) for k, v in per_stratum.items()},
        "subclass_counts": dict(subclass_counts.most_common()),
        "subclass_by_question_type": {k: dict(v) for k, v in subclass_by_type.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_sample_indices(raw: Optional[str]) -> Optional[list[int]]:
    """Same parser as scripts/judge_eval/evaluate.py."""
    if raw is None:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


def _stratum_from_turns_file(path: Path) -> Stratum:
    parts = path.resolve().parts
    try:
        i = parts.index("models")
        return _parse_experiment(parts[i + 1], parts[i + 2]) or Stratum(
            parts[i + 1], parts[i + 2], "unknown", "fo", False
        )
    except (ValueError, IndexError):
        return Stratum("unknown", "unknown", "unknown", "fo", False)


async def _amain(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)

    if args.turns_file is not None:
        picks = [(_stratum_from_turns_file(args.turns_file), args.turns_file)]
    else:
        sample_indices = _parse_sample_indices(args.sample_indices)
        seekers = (
            {_slug(s.strip()) for s in args.seekers.split(",") if s.strip()}
            if args.seekers else None
        )
        oracle = _slug(args.oracle.strip()) if args.oracle else None
        buckets = discover_conversations(
            args.outputs_root, run_index=args.run_index, seekers=seekers, oracle=oracle
        )
        n_total = sum(len(v) for v in buckets.values())
        logger.info(
            "Discovered %d conversations across %d strata%s%s%s.",
            n_total, len(buckets),
            f" (run_index={args.run_index})" if args.run_index else "",
            f" (seekers={sorted(seekers)})" if seekers else "",
            f" (oracle={oracle})" if oracle else "",
        )
        for st, paths in sorted(buckets.items(), key=lambda kv: (kv[0].domain, kv[0].mode, kv[0].cot)):
            cot = "cot" if st.cot else "no_cot"
            logger.debug("  %s/%s/%s [%s/%s]: %d", st.domain, st.mode, cot, st.model_slug, st.experiment, len(paths))
        picks = stratified_sample(buckets, args.per_stratum, rng, sample_indices=sample_indices)
        if sample_indices is not None:
            mode = f"sample_indices={sample_indices}"
        elif args.per_stratum is not None:
            mode = f"per_stratum={args.per_stratum}, seed={args.seed}"
        else:
            mode = "all"
        logger.info("Sampled %d conversations (%s).", len(picks), mode)

    if args.dry_run:
        for st, path in picks:
            logger.info("  would classify: %s/%s/%s :: %s", st.domain, st.mode, "cot" if st.cot else "no_cot", path)
        return 0
    if not picks:
        print("No conversations to classify.", file=sys.stderr)
        return 1

    # Resumability: load already-classified turns_path keys from existing JSONL.
    out_jsonl = args.out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.force and out_jsonl.exists():
        out_jsonl.unlink()
    done_paths: set[str] = set()
    existing: list[dict[str, Any]] = []
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done_paths.add(rec["turns_path"])
                existing.append(rec)
            except Exception as e:  # noqa: BLE001
                logger.warning("skipping malformed JSONL line: %s", e)

    todo = [(st, p) for st, p in picks if str(p) not in done_paths]
    logger.info("Resume: %d already classified, %d to do.", len(existing), len(todo))

    results: list[dict[str, Any]] = []
    if todo:
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
        thinking = not args.no_thinking
        semaphore = asyncio.Semaphore(args.max_concurrency)
        total = len(todo)
        n_done = 0
        lock = asyncio.Lock()
        t0 = time.monotonic()

        async def _run(stratum: Stratum, path: Path) -> dict[str, Any]:
            nonlocal n_done
            res = await classify_conversation(path, stratum, client, args.model, thinking, semaphore)
            errs = sum(1 for t in res["turns"] if "error" in t["classification"])
            async with lock:
                # Conversa que falhou NÃO é persistida: assim o turns_path não
                # entra em done_paths e o resume seguinte a refaz sozinho (sem
                # precisar limpar o jsonl). O jsonl só acumula registros bons.
                if not errs:
                    with out_jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                n_done += 1
                elapsed = time.monotonic() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta_s = (total - n_done) / rate if rate > 0 else 0.0
                eta = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                level = logging.WARNING if errs else logging.INFO
                logger.log(
                    level,
                    "[%d/%d] %s/%s %s rate=%.2f conv/s eta=%s",
                    n_done, total, stratum.experiment, path.parent.name,
                    f"({errs} turn errors — NÃO salvo, refaz no próximo run)" if errs else "ok",
                    rate, eta,
                )
            return res

        try:
            results = list(await asyncio.gather(*(_run(st, p) for st, p in todo)))
        finally:
            await client.close()

    conv_results = existing + results
    summary = summarise(conv_results)
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    logger.info("=== Summary ===")
    logger.info("  conversations: %d", summary['total_conversations'])
    logger.info("  turns:         %d (errors: %d)", summary['total_turns'], summary['errors'])
    logger.info("  question_type: %s", summary['question_type_counts'])
    logger.info("  redundancy:    %s", summary['redundancy_counts'])
    if summary["subclass_counts"]:
        logger.info("  subclass tags (top):")
        for name, count in list(summary["subclass_counts"].items())[:20]:
            logger.info("    %4d  %s", count, name)
    logger.info("Wrote: %s  (%d conversations)", out_jsonl, len(conv_results))
    logger.info("Wrote: %s", summary_path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://200.137.197.131:60002/v1")
    p.add_argument("--api-key", default="NINGUEM-TA-PURO-2K26")
    p.add_argument("--model", default="nvidia/Kimi-K2.5-NVFP4")
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("outputs/question_classifications.jsonl"),
        help="Output JSONL file (one conversation per line). Appended to on resume.",
    )
    p.add_argument("--turns-file", type=Path, default=None, help="Classify a single turns.jsonl (overrides sampling).")
    p.add_argument(
        "--per-stratum",
        type=int,
        default=None,
        help="Conversations per stratum. Default: None = full sweep; pass an int to cap.",
    )
    p.add_argument("--run-index", type=int, default=None,
                   help="Keep only conversations whose dir ends with _runNN matching this number "
                        "(e.g. 1 → only run01). Same convention as scripts/judge_eval.")
    p.add_argument("--sample-indices", type=str, default=None,
                   help="Comma-separated 0-based positions in each stratum's sorted path list "
                        "(e.g. '0,10,20,30,40,50,60,70,80,90,100'). Deterministic and aligned "
                        "with judge_eval. Overrides --per-stratum/--seed when provided.")
    p.add_argument("--seekers", type=str, default=None,
                   help="Comma-separated seeker model names to keep (e.g. "
                        "'Qwen3-8B,google/gemma-4-31B-it'). '/' is slugified to '-' to "
                        "match the s_<seeker>__o_<oracle>__p_<pruner> dir. Exact segment "
                        "match. Omit = all seekers.")
    p.add_argument("--oracle", type=str, default=None,
                   help="Keep only triples whose oracle equals this (e.g. 'Qwen3-8B'). "
                        "Omit = any oracle.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-concurrency", type=int, default=16, help="Max in-flight LLM requests.")
    p.add_argument("--no-thinking", action="store_true", help="Disable reasoning mode on the classifier.")
    p.add_argument("--dry-run", action="store_true", help="List what would be classified and exit.")
    p.add_argument("--force", action="store_true", help="Re-classify all conversations, ignoring existing JSONL.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Verbosidade do logging (default: INFO; DEBUG inclui lista por stratum).")
    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # line-buffered por padrão; aparece em tempo real no tee
    )
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
