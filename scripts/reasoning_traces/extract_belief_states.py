"""Batch belief-state extraction over CoT conversations.

For every Chain-of-Thought conversation under ``outputs/`` (or a given ``runs.csv``)
this script runs the extractor LLM on each turn's reasoning, scores the recovered
belief state against the Pruner's true active set Omega_t, and writes the results to
an append-only unified JSONL plus a per-experiment summary.

It mirrors ``evaluate_all_seeker_choices.py``: same CoT discovery, the same
``SEEKERS`` / ``SAMPLE_INDICES`` / ``ONLY_RUN_INDEX`` filters, the same thread-pool
parallelism, and the same resumable skip-if-exists behaviour (keyed by the
conversation's ``seeker.json`` path).

Examples:
    # one experiment
    python3 scripts/reasoning_traces/extract_belief_states.py \
        outputs/models/<triple>/<exp>_cot/runs.csv \
        --base-url http://<NODE_IP>:<PORT>/v1 --model google/gemma-4-31B-it

    # everything, filtered to the 6 CoT seekers (run01, 15 samples)
    python3 scripts/reasoning_traces/extract_belief_states.py --all \
        --base-url http://<NODE_IP>:<PORT>/v1 --model google/gemma-4-31B-it \
        --seekers Qwen3-8B,Qwen3-4B-Thinking-2507,... --only-run-index 1 \
        --sample-indices 10,20,30 --max-workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# Make ``src`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.llm_config import LLMConfig  # noqa: E402
from src.analysis.belief_state_extraction import evaluate_conversation  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

UNIFIED_JSONL_NAME = "outputs/belief_states.jsonl"
SUMMARY_NAME = "belief_state_summary.json"
_WRITE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Discovery (parallels evaluate_all_seeker_choices.py)
# --------------------------------------------------------------------------- #
def find_cot_runs_csvs(base_dir: Path) -> list[Path]:
    """Return every ``runs.csv`` belonging to a CoT experiment."""
    out = []
    for csv_path in base_dir.rglob("runs.csv"):
        name = csv_path.parent.name
        # Skip no-CoT, the oracle-no-thinking contaminated dirs, and the
        # verbal-pool ablation -- none belong to the main CoT track.
        if "no_cot" in name or name.endswith("_ont") or "_with_prior" in name:
            continue
        if name.endswith("_cot") or "_cot_" in name:
            out.append(csv_path)
    # When both `<exp>` and `<exp>_with_kickoff` exist under the SAME triple dir,
    # `_with_kickoff` is the canonical version -- drop the plain sibling.
    present = {(p.parent.parent, p.parent.name) for p in out}
    return sorted(
        p for p in out
        if (p.parent.parent, f"{p.parent.name}_with_kickoff") not in present
    )


def _slug(text: str) -> str:
    return text.replace("/", "-")


def _seeker_segment(runs_csv: Path) -> str:
    """Extract the seeker slug from the ``s_<seeker>__o_..__p_..`` triple dir."""
    triple = runs_csv.parent.parent.name
    if triple.startswith("s_") and "__o_" in triple:
        return triple[2:].split("__o_")[0]
    return triple


def find_conversation_dirs(
    runs_csv: Path,
    outputs_base: Path,
    only_run_index: Optional[int],
    sample_indices: Optional[set[int]],
) -> list[Path]:
    """List conversation dirs for a ``runs.csv``, applying run/sample filters.

    Mirrors ``evaluate_all_seeker_choices.find_conversation_dirs_from_runs_csv``
    exactly so the belief pass covers the SAME conversations as the question-choice
    analysis: filter by ``run_index`` first, then take ``sample_indices`` as 0-based
    positions *within the filtered rows* (not within the full CSV).
    """
    with runs_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if only_run_index is not None:
        rows = [r for r in rows if str(r.get("run_index", "")).strip() == str(only_run_index)]
    if sample_indices is not None:
        ordered = sorted(sample_indices)
        rows = [rows[i] for i in ordered if 0 <= i < len(rows)]

    dirs: list[Path] = []
    seen: set[Path] = set()
    for row in rows:
        conv_path = row.get("conversation_path") or ""
        if not conv_path:
            continue
        path = Path(conv_path)
        if not path.is_absolute():
            path = outputs_base / conv_path
        path = path.resolve()
        if path not in seen and path.is_dir():
            seen.add(path)
            dirs.append(path)
    return dirs


# --------------------------------------------------------------------------- #
# Unified JSONL (resumable, mirrors evaluate_all_seeker_choices.py)
# --------------------------------------------------------------------------- #
def _key(conversation_dir: Path) -> str:
    return str((conversation_dir / "seeker.json").resolve())


def load_done_keys(unified_jsonl: Path) -> set[str]:
    if not unified_jsonl.exists():
        return set()
    done = set()
    with unified_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["seeker_path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_record(unified_jsonl: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _WRITE_LOCK:
        unified_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with unified_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def process_single_conversation(
    conversation_dir: Path,
    extractor_config: LLMConfig,
    unified_jsonl: Path,
    done_keys: set[str],
    max_turns: Optional[int],
    force: bool,
    save_io: bool = True,
) -> str:
    """Extract+score one conversation; returns 'ok' | 'skipped' | 'error'."""
    key = _key(conversation_dir)
    if key in done_keys and not force:
        return "skipped"
    try:
        evaluation = evaluate_conversation(
            conversation_dir, extractor_config, max_turns=max_turns, save_io=save_io
        )
    except Exception as exc:  # noqa: BLE001 - record nothing so the conv can retry later
        logger.warning("Failed %s: %s", conversation_dir, exc)
        return "error"
    record = {
        "seeker_path": key,
        "experiment_dir": str(conversation_dir.parent.parent),
        **evaluation,
    }
    append_record(unified_jsonl, record)
    return "ok"


def write_experiment_summary(runs_csv: Path, records: list[dict[str, Any]]) -> None:
    """Aggregate per-conversation summaries into ``belief_state_summary.json``."""
    if not records:
        return
    keys = [
        "explicit_tracking_rate", "mean_kept_precision", "mean_zombie_kept_rate",
        "mean_excluded_correct_rate", "mean_count_abs_error", "mean_n_named",
        "mean_n_constraints", "ig_per_turn",
    ]
    agg: dict[str, Any] = {"n_conversations": len(records)}
    for k in keys:
        vals = [r["summary"].get(k) for r in records]
        vals = [v for v in vals if isinstance(v, (int, float))]
        agg[k] = (sum(vals) / len(vals)) if vals else None
    agg["n_fatal_conversations"] = sum(1 for r in records if r["summary"].get("any_fatal_target_excluded"))
    out = runs_csv.parent / SUMMARY_NAME
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Batch belief-state extraction over CoT conversations.")
    parser.add_argument("runs_csv", nargs="?", help="A specific runs.csv (omit with --all).")
    parser.add_argument("--all", action="store_true", help="Process every CoT runs.csv under --outputs-base.")
    parser.add_argument("--outputs-base", default="outputs", help="Base outputs directory.")
    parser.add_argument("--model", default="google/gemma-4-31B-it", help="Extractor served-model-name.")
    parser.add_argument("--base-url", required=True, help="Extractor vLLM base URL (…/v1).")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=8, help="Conversations in parallel.")
    parser.add_argument("--max-turns", type=int, default=None, help="Cap turns per conversation.")
    parser.add_argument("--only-run-index", type=int, default=None)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated 0-based row positions.")
    parser.add_argument("--seekers", default=None, help="Comma-separated seeker slugs to keep (with --all).")
    parser.add_argument(
        "--oracle-pruner", default=None,
        help="Keep only triples with this oracle AND pruner (e.g. Qwen3-8B); "
        "excludes ablation triples. With --all.",
    )
    parser.add_argument("--unified-jsonl", default=UNIFIED_JSONL_NAME)
    parser.add_argument("--force", action="store_true", help="Re-extract even if already recorded.")
    parser.add_argument("--dry-run", action="store_true", help="List targets and exit.")
    parser.add_argument("--no-raw-io", action="store_true",
                        help="Do not store the extractor's prompt/raw output per turn (saves space).")
    args = parser.parse_args()

    setup_logging()
    outputs_base = Path(args.outputs_base)
    unified_jsonl = Path(args.unified_jsonl)

    if args.all:
        runs_csvs = find_cot_runs_csvs(outputs_base)
        if args.seekers:
            wanted = {_slug(s.strip()) for s in args.seekers.split(",") if s.strip()}
            runs_csvs = [c for c in runs_csvs if _slug(_seeker_segment(c)) in wanted]
        if args.oracle_pruner:
            suffix = f"__o_{_slug(args.oracle_pruner)}__p_{_slug(args.oracle_pruner)}"
            runs_csvs = [c for c in runs_csvs if c.parent.parent.name.endswith(suffix)]
    elif args.runs_csv:
        runs_csvs = [Path(args.runs_csv)]
    else:
        parser.error("Provide a runs.csv or --all.")

    sample_indices = None
    if args.sample_indices:
        sample_indices = {int(x) for x in args.sample_indices.split(",") if x.strip()}

    extractor_config = LLMConfig(
        model=args.model, api_key=args.api_key, base_url=args.base_url, timeout=args.timeout
    )

    done_keys = load_done_keys(unified_jsonl)
    logger.info("CoT experiments: %d | already done: %d", len(runs_csvs), len(done_keys))

    totals = {"ok": 0, "skipped": 0, "error": 0}
    for runs_csv in runs_csvs:
        conv_dirs = find_conversation_dirs(runs_csv, outputs_base, args.only_run_index, sample_indices)
        if not conv_dirs:
            continue
        logger.info("[%s] %d conversations", runs_csv.parent.name, len(conv_dirs))
        if args.dry_run:
            continue

        results: dict[Path, str] = {}
        if args.max_workers > 1:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {
                    pool.submit(
                        process_single_conversation, cd, extractor_config,
                        unified_jsonl, done_keys, args.max_turns, args.force, not args.no_raw_io,
                    ): cd
                    for cd in conv_dirs
                }
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
        else:
            for cd in conv_dirs:
                results[cd] = process_single_conversation(
                    cd, extractor_config, unified_jsonl, done_keys, args.max_turns, args.force, not args.no_raw_io
                )

        for status in results.values():
            totals[status] = totals.get(status, 0) + 1

        # Rebuild this experiment's summary from all of its records in the JSONL.
        exp_keys = {_key(cd) for cd in conv_dirs}
        exp_records = [
            json.loads(line)
            for line in unified_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("seeker_path") in exp_keys
        ]
        write_experiment_summary(runs_csv, exp_records)

    logger.info("Done. ok=%d skipped=%d error=%d", totals["ok"], totals["skipped"], totals["error"])


if __name__ == "__main__":
    main()
