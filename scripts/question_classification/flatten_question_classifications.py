#!/usr/bin/env python3
"""Flatten the question-classifications JSONL into one wide CSV.

Each row of the output CSV is a single seeker turn with its full label set.
The resulting file feeds pandas / notebooks / SQL directly.

Input (default):
    outputs/question_classifications.jsonl   (one conversation JSON per line)

Output (default):
    outputs/question_classifications.csv

Columns:
    seeker, oracle, pruner, model_slug,
    experiment, domain, mode (fo/po), cot (0/1),
    target, run_index,
    turn, question, question_echoed, oracle_answer,
    question_type_rationale, question_type, subclasses_rationale, subclasses,
    redundancy_rationale, redundancy,
    error (empty on success)

``subclasses`` is a ";"-joined string (e.g. "comparative;quantitative_threshold")
so the CSV stays flat. Split with ``df.subclasses.str.split(";")`` in pandas
or explode the tags for per-tag aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


_MODEL_SLUG_RE = re.compile(r"^s_(?P<seeker>.+?)__o_(?P<oracle>.+?)__p_(?P<pruner>.+)$")
_RUN_SUFFIX_RE = re.compile(r"_run(\d+)$")


def parse_model_slug(slug: str) -> tuple[str, str, str]:
    m = _MODEL_SLUG_RE.match(slug)
    if m:
        return m["seeker"], m["oracle"], m["pruner"]
    return slug, "", ""


def parse_run_index(target_folder: str) -> tuple[str, int | None]:
    """Split ``disease-xxx-N_run02`` into (``disease-xxx-N``, 2).

    Human baselines have no ``_runNN`` suffix — returns (target, None) for those.
    """
    m = _RUN_SUFFIX_RE.search(target_folder)
    if m:
        return target_folder[: m.start()], int(m.group(1))
    return target_folder, None


COLUMNS = [
    "seeker",
    "oracle",
    "pruner",
    "model_slug",
    "experiment",
    "domain",
    "mode",
    "cot",
    "target",
    "run_index",
    "turn",
    "question",
    "question_echoed",
    "oracle_answer",
    "question_type_rationale",
    "question_type",
    "subclasses_rationale",
    "subclasses",
    "redundancy_rationale",
    "redundancy",
    "error",
]


def iter_rows(conv: dict[str, Any]) -> list[dict[str, Any]]:
    seeker, oracle, pruner = parse_model_slug(conv.get("model_slug", ""))
    target_folder = conv.get("target", "")
    target_base, run_idx = parse_run_index(target_folder)

    base = {
        "seeker": seeker,
        "oracle": oracle,
        "pruner": pruner,
        "model_slug": conv.get("model_slug", ""),
        "experiment": conv.get("experiment", ""),
        "domain": conv.get("domain", ""),
        "mode": conv.get("mode", ""),
        "cot": 1 if conv.get("cot") else 0,
        "target": target_base,
        "run_index": run_idx if run_idx is not None else "",
    }

    rows: list[dict[str, Any]] = []
    for t in conv.get("turns", []):
        cls = t.get("classification", {}) or {}
        is_error = "error" in cls
        subclasses, subclasses_rationale = _extract_subclasses(cls)
        row = dict(base)
        row.update(
            {
                "turn": t.get("turn", ""),
                "question": (t.get("question") or "").replace("\n", " ").strip(),
                "question_echoed": ("" if is_error else (cls.get("question") or "").replace("\n", " ").strip()),
                "oracle_answer": (t.get("oracle_answer") or "").replace("\n", " ").strip(),
                "question_type_rationale": "" if is_error else cls.get("question_type_rationale", ""),
                "question_type": "" if is_error else cls.get("question_type", ""),
                "subclasses_rationale": subclasses_rationale,
                "subclasses": ";".join(subclasses),
                "redundancy_rationale": "" if is_error else cls.get("redundancy_rationale", ""),
                "redundancy": "" if is_error else cls.get("redundancy", ""),
                "error": cls.get("error", ""),
            }
        )
        rows.append(row)
    return rows


def _extract_subclasses(cls: dict[str, Any]) -> tuple[list[str], str]:
    """Return (subclasses, rationale) handling both the new flat shape and the
    legacy nested one (``subclass: {proposed_class, rationale}``).
    """
    if "error" in cls:
        return [], ""
    # New shape: flat list at top level.
    if "subclasses" in cls:
        raw = cls.get("subclasses") or []
        tags = [str(x).strip() for x in raw if str(x).strip()]
        rationale = cls.get("subclasses_rationale", "") or cls.get("subclass_rationale", "") or ""
        return tags, rationale
    # Legacy shape: single object under "subclass".
    sub = cls.get("subclass")
    if isinstance(sub, dict):
        label = sub.get("proposed_class", "").strip()
        rat = sub.get("subclass_rationale") or sub.get("rationale", "") or ""
        return ([label] if label else []), rat
    if isinstance(sub, str) and sub.strip():
        return [sub.strip()], cls.get("subclass_rationale", "") or ""
    return [], ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("outputs/question_classifications.jsonl"),
        help="JSONL produced by classify_questions.py (one conversation per line).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/question_classifications.csv"),
        help="Destination CSV.",
    )
    args = p.parse_args()

    if not args.input_jsonl.exists():
        print(f"ERROR: {args.input_jsonl} not found. Run classify_questions.py first.", file=sys.stderr)
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    total_convs = 0
    skipped = 0
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i, line in enumerate(args.input_jsonl.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"  skip line {i}: {e}", file=sys.stderr)
                skipped += 1
                continue
            for row in iter_rows(conv):
                writer.writerow(row)
                total_turns += 1
            total_convs += 1
            if i % 500 == 0:
                print(f"  processed {i} conversations ({total_turns} rows so far)")

    print(f"\nWrote {total_turns} rows from {total_convs} conversations to {args.output_csv}")
    if skipped:
        print(f"Skipped {skipped} malformed line(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
