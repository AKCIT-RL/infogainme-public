#!/usr/bin/env python3
"""Evaluate Seeker's question choices for all conversations in a runs.csv.

This script reads a runs.csv file, extracts unique conversation directories,
and evaluates the Seeker's question choices for each conversation.
"""

import argparse
import csv
import json
import sys
import math
import statistics
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict, Any, Optional
from dotenv import load_dotenv
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.question_evaluator import (
    evaluate_seeker_choices,
    _get_jsonl_index,
    NoTraceError,
    set_traces_jsonl,
)
from src.agents.llm_config import LLMConfig
from src.utils import ClaryLogger

load_dotenv()

logger = ClaryLogger.get_logger(__name__)

_SERVERS_YAML = project_root / "configs" / "servers.yaml"

# Single append-only JSONL with all per-conversation evaluation records.
# One record per line, keyed by `seeker_path` (str of the seeker.json path).
UNIFIED_JSONL_NAME = "question_evaluations.jsonl"

_WRITE_LOCK = threading.Lock()


def unified_jsonl_path(outputs_base_dir: Path) -> Path:
    return outputs_base_dir / UNIFIED_JSONL_NAME


def _seeker_path_key(conversation_dir: Path) -> str:
    return str(conversation_dir / "seeker.json")


def _load_done_index(unified_jsonl: Path) -> Set[str]:
    """Return the set of seeker_path keys already in the unified JSONL."""
    done: Set[str] = set()
    if not unified_jsonl.exists():
        return done
    with unified_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("seeker_path")
            if key:
                done.add(key)
    return done


def _append_record(unified_jsonl: Path, record: Dict[str, Any]) -> None:
    """Locked append of one JSON record (one line) to the unified JSONL."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        unified_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with unified_jsonl.open("a", encoding="utf-8") as f:
            f.write(line)


def _build_record(
    conversation_dir: Path,
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrap an evaluation result with the keys needed for the unified JSONL."""
    # outputs/models/<triple>/<exp>/conversations/<target>  →  experiment_dir = parent.parent
    return {
        "seeker_path": _seeker_path_key(conversation_dir),
        "conversation_dir": str(conversation_dir),
        "experiment_dir": str(conversation_dir.parent.parent),
        **evaluation,
    }


def migrate_per_conv_files(
    outputs_base_dir: Path,
    unified_jsonl: Path,
    done_keys: Set[str],
) -> int:
    """One-time migration: append legacy per-conversation question_evaluation.json
    files into the unified JSONL. Idempotent — skips entries already in done_keys.
    Returns number of records migrated."""
    migrated = 0
    for per_conv in outputs_base_dir.rglob("question_evaluation.json"):
        seeker_key = _seeker_path_key(per_conv.parent)
        if seeker_key in done_keys:
            continue
        try:
            data = json.loads(per_conv.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Migration skip (parse error) %s: %s", per_conv, e)
            continue
        if not (data.get("turns_evaluation") and data.get("summary")):
            continue
        record = _build_record(per_conv.parent, data)
        _append_record(unified_jsonl, record)
        done_keys.add(seeker_key)
        migrated += 1
    if migrated > 0:
        logger.info("📦 Migrated %d legacy per-conv files → %s", migrated, unified_jsonl)
    return migrated


def _load_evaluations_for_exp(
    unified_jsonl: Path,
    experiment_dir: Path,
) -> List[Dict[str, Any]]:
    """Return all evaluation records under the given experiment directory."""
    if not unified_jsonl.exists():
        return []
    exp_str = str(experiment_dir)
    records: List[Dict[str, Any]] = []
    with unified_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("experiment_dir") == exp_str:
                records.append(rec)
    return records


def _load_servers() -> Dict[str, str]:
    if not _SERVERS_YAML.exists():
        return {}
    with _SERVERS_YAML.open() as f:
        data = yaml.safe_load(f)
    return (data or {}).get("servers", {})


def _resolve_base_url(model: str, cli_url: Optional[str]) -> str:
    """Return CLI url if given, else look up model in servers.yaml."""
    if cli_url:
        return cli_url
    servers = _load_servers()
    url = servers.get(model)
    if not url:
        raise SystemExit(
            f"Model '{model}' not found in {_SERVERS_YAML}. "
            "Pass --base-url explicitly or add the model to configs/servers.yaml."
        )
    return url.rstrip("/")


def _slug(name: str) -> str:
    return name.replace("/", "-")


def _triple_parts(dir_name: str):
    """Parse 's_<seeker>__o_<oracle>__p_<pruner>' → (seeker, oracle, pruner) or None."""
    if not dir_name.startswith("s_") or "__o_" not in dir_name or "__p_" not in dir_name:
        return None
    rest = dir_name[2:]
    s, _, rest = rest.partition("__o_")
    o, _, p = rest.partition("__p_")
    return (s, o, p) if s and o and p else None


def find_conversation_dirs_from_runs_csv(
    runs_csv_path: Path,
    outputs_base_dir: Path,
    only_run_index: Optional[int] = None,
    sample_indices: Optional[List[int]] = None,
) -> List[Path]:
    """
    Reads a runs.csv file and returns a list of unique conversation directory paths.

    Args:
        runs_csv_path: Path to runs.csv file.
        outputs_base_dir: Base directory where conversation paths are relative to.
        only_run_index: If given, keep only rows matching this run_index (e.g. 1
            to evaluate just run01 of every target).
        sample_indices: If given, pick rows at these 0-based positions within the
            filtered list (mirrors judge_eval --sample-indices behaviour).

    Returns:
        List of conversation directory paths.
    """
    if not runs_csv_path.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv_path}")

    with runs_csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if only_run_index is not None:
        rows = [r for r in rows if str(r.get("run_index", "")).strip() == str(only_run_index)]

    if sample_indices is not None:
        rows = [rows[i] for i in sample_indices if 0 <= i < len(rows)]

    conversation_dirs: list[Path] = []
    seen: set[Path] = set()
    for row in rows:
        conversation_path_str = row.get("conversation_path")
        if not conversation_path_str:
            continue
        full_conversation_path = outputs_base_dir / conversation_path_str
        if full_conversation_path.exists() and full_conversation_path not in seen:
            seen.add(full_conversation_path)
            conversation_dirs.append(full_conversation_path)
        elif not full_conversation_path.exists():
            logger.warning("Conversation directory not found: %s", full_conversation_path)

    return conversation_dirs


def process_single_conversation(
    conversation_dir: Path,
    oracle_config: LLMConfig,
    pruner_config: LLMConfig,
    unified_jsonl: Path,
    done_keys: Set[str],
    force: bool = False,
    dataset_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Process a single conversation directory.

    Skip-check is by `seeker_path` membership in `done_keys` (built from the
    unified JSONL). New successful evaluations are appended to `unified_jsonl`.
    """
    seeker_key = _seeker_path_key(conversation_dir)

    if not force and seeker_key in done_keys:
        return {
            "status": "skipped",
            "conversation_dir": str(conversation_dir),
            "seeker_path": seeker_key,
            "reason": "already in unified JSONL",
        }

    # Reasoning traces are read from outputs/seeker_traces.jsonl by the evaluator.
    required_files = ["turns.jsonl", "metadata.json"]
    missing_files = [f for f in required_files if not (conversation_dir / f).exists()]
    if missing_files:
        return {
            "status": "error",
            "conversation_dir": str(conversation_dir),
            "seeker_path": seeker_key,
            "reason": f"Missing required files: {', '.join(missing_files)}",
        }

    try:
        results = evaluate_seeker_choices(
            conversation_dir,
            oracle_config,
            pruner_config,
            dataset_csv_path=dataset_csv_path,
        )
        record = _build_record(conversation_dir, results)
        _append_record(unified_jsonl, record)
        done_keys.add(seeker_key)
        return {
            "status": "success",
            "conversation_dir": str(conversation_dir),
            "seeker_path": seeker_key,
            "turns_evaluated": results["summary"]["total_turns_evaluated"],
            "optimal_choices": results["summary"]["optimal_choices"],
            "optimal_choice_rate": results["summary"]["optimal_choice_rate"],
        }
    except NoTraceError as e:
        # Don't persist an empty record — pollutes the JSONL and poisons the
        # done_keys cache so a later run (after traces are synthesized) skips
        # this conversation forever.
        return {
            "status": "skipped",
            "conversation_dir": str(conversation_dir),
            "seeker_path": seeker_key,
            "reason": str(e),
        }
    except Exception as e:
        logger.error("Error processing %s: %s", conversation_dir, e, exc_info=True)
        return {
            "status": "error",
            "conversation_dir": str(conversation_dir),
            "seeker_path": seeker_key,
            "reason": str(e),
        }


def calculate_aggregate_summary(
    evaluations: List[Dict[str, Any]],
    success_count: int,
    skipped_count: int,
    error_count: int,
    total_conversations: int,
) -> Dict[str, Any]:
    """Calculate aggregate statistics from a list of evaluation records.

    Args:
        evaluations: Raw evaluation dicts (each has `summary`, `turns_evaluation`,
            `target`). Pulled from the unified JSONL filtered by experiment.
        success_count, skipped_count, error_count: counts from this run.
        total_conversations: total conversations attempted in this run.
    """
    if not evaluations:
        return {
            "total_conversations": total_conversations,
            "total_evaluated": success_count + skipped_count,
            "total_success": success_count,
            "total_skipped": skipped_count,
            "total_errors": error_count,
        }
    
    # Aggregate statistics
    total_turns_evaluated = sum(e["summary"]["total_turns_evaluated"] for e in evaluations)
    total_optimal_choices = sum(e["summary"]["optimal_choices"] for e in evaluations)
    
    # Calculate averages
    optimal_choice_rates = [e["summary"]["optimal_choice_rate"] for e in evaluations]
    avg_optimal_choice_rate = sum(optimal_choice_rates) / len(optimal_choice_rates) if optimal_choice_rates else 0.0
    
    # Info gain statistics
    # Filter out None and negative values (negative values indicate evaluation errors)
    avg_chosen_ig_list = [e["summary"]["avg_chosen_info_gain"] for e in evaluations 
                         if e["summary"].get("avg_chosen_info_gain") is not None 
                         and e["summary"]["avg_chosen_info_gain"] >= 0.0]
    avg_optimal_ig_list = [e["summary"]["avg_optimal_info_gain"] for e in evaluations 
                          if e["summary"].get("avg_optimal_info_gain") is not None 
                          and e["summary"]["avg_optimal_info_gain"] >= 0.0]
    
    avg_chosen_info_gain = sum(avg_chosen_ig_list) / len(avg_chosen_ig_list) if avg_chosen_ig_list else 0.0
    avg_optimal_info_gain = sum(avg_optimal_ig_list) / len(avg_optimal_ig_list) if avg_optimal_ig_list else 0.0
    
    # Average questions considered per turn
    avg_questions_considered_list = []
    for e in evaluations:
        avg_questions = e["summary"].get("avg_questions_considered_per_turn")
        if avg_questions is None:
            avg_questions = 0.0
        avg_questions_considered_list.append(avg_questions)
    avg_questions_considered_per_turn = (sum(avg_questions_considered_list) / len(avg_questions_considered_list) 
                                        if avg_questions_considered_list else 0.0)
    
    # Total connection errors (use 0 as default for old files)
    total_connection_errors = sum(e["summary"].get("total_connection_errors", 0) for e in evaluations)
    
    # Group by target city
    by_target: Dict[str, Dict[str, Any]] = {}
    for e in evaluations:
        target_id = e["target"]["id"]
        target_label = e["target"]["label"]
        
        if target_id not in by_target:
            by_target[target_id] = {
                "target_id": target_id,
                "target_label": target_label,
                "conversations": 0,
                "total_turns_evaluated": 0,
                "total_optimal_choices": 0,
                "optimal_choice_rates": [],
                "avg_chosen_info_gains": [],
                "avg_optimal_info_gains": [],
                "avg_questions_considered": [],
                "total_connection_errors": 0
            }
        
        by_target[target_id]["conversations"] += 1
        by_target[target_id]["total_turns_evaluated"] += e["summary"]["total_turns_evaluated"]
        by_target[target_id]["total_optimal_choices"] += e["summary"]["optimal_choices"]
        by_target[target_id]["optimal_choice_rates"].append(e["summary"]["optimal_choice_rate"])
        
        # Filter out negative values (indicate evaluation errors)
        chosen_ig = e["summary"].get("avg_chosen_info_gain")
        if chosen_ig is not None and chosen_ig >= 0.0:
            by_target[target_id]["avg_chosen_info_gains"].append(chosen_ig)
        optimal_ig = e["summary"].get("avg_optimal_info_gain")
        if optimal_ig is not None and optimal_ig >= 0.0:
            by_target[target_id]["avg_optimal_info_gains"].append(optimal_ig)
        
        # Average questions considered per turn (use 0.0 as default for old files)
        avg_questions = e["summary"].get("avg_questions_considered_per_turn")
        if avg_questions is None:
            avg_questions = 0.0
        by_target[target_id]["avg_questions_considered"].append(avg_questions)
        
        # Connection errors (use 0 as default for old files)
        by_target[target_id]["total_connection_errors"] += e["summary"].get("total_connection_errors", 0)
    
    # Calculate per-target statistics
    by_target_summary = {}
    for target_id, data in by_target.items():
        rates = data["optimal_choice_rates"]
        chosen_igs = data["avg_chosen_info_gains"]
        optimal_igs = data["avg_optimal_info_gains"]
        avg_questions_list = data["avg_questions_considered"]
        
        # Calculate means
        avg_rate = sum(rates) / len(rates) if rates else 0.0
        avg_chosen_ig = sum(chosen_igs) / len(chosen_igs) if chosen_igs else None
        avg_optimal_ig = sum(optimal_igs) / len(optimal_igs) if optimal_igs else None
        avg_questions = sum(avg_questions_list) / len(avg_questions_list) if avg_questions_list else None
        
        # Calculate stds (populacional para cada target, já que temos todas as conversas)
        std_rate = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        std_chosen_ig = statistics.pstdev(chosen_igs) if chosen_igs and len(chosen_igs) > 1 else None
        std_optimal_ig = statistics.pstdev(optimal_igs) if optimal_igs and len(optimal_igs) > 1 else None
        std_questions = statistics.pstdev(avg_questions_list) if avg_questions_list and len(avg_questions_list) > 1 else None
        
        by_target_summary[target_id] = {
            "target_label": data["target_label"],
            "conversations": data["conversations"],
            "total_turns_evaluated": data["total_turns_evaluated"],
            "total_optimal_choices": data["total_optimal_choices"],
            "avg_optimal_choice_rate": avg_rate,
            "std_optimal_choice_rate": std_rate,
            "avg_chosen_info_gain": avg_chosen_ig,
            "std_chosen_info_gain": std_chosen_ig,
            "avg_optimal_info_gain": avg_optimal_ig,
            "std_optimal_info_gain": std_optimal_ig,
            "avg_questions_considered_per_turn": avg_questions,
            "std_questions_considered_per_turn": std_questions,
            "total_connection_errors": data["total_connection_errors"]
        }
    
    # Calculate global stds and SEs using hierarchical approach (by target)
    num_targets = len(by_target_summary)
    
    # Collect target-level means for std calculation
    target_rates = [target["avg_optimal_choice_rate"] for target in by_target_summary.values()]
    target_chosen_igs = [target["avg_chosen_info_gain"] for target in by_target_summary.values() 
                         if target["avg_chosen_info_gain"] is not None]
    target_optimal_igs = [target["avg_optimal_info_gain"] for target in by_target_summary.values() 
                          if target["avg_optimal_info_gain"] is not None]
    target_questions = [target["avg_questions_considered_per_turn"] for target in by_target_summary.values() 
                       if target["avg_questions_considered_per_turn"] is not None]
    
    # Calculate stds (amostral para inferência estatística)
    std_optimal_choice_rate = statistics.stdev(target_rates) if len(target_rates) > 1 else 0.0
    std_chosen_info_gain = statistics.stdev(target_chosen_igs) if len(target_chosen_igs) > 1 else None
    std_optimal_info_gain = statistics.stdev(target_optimal_igs) if len(target_optimal_igs) > 1 else None
    std_questions_considered_per_turn = statistics.stdev(target_questions) if len(target_questions) > 1 else None
    
    # Calculate SEs (hierarchical: std / sqrt(num_targets))
    se_optimal_choice_rate = std_optimal_choice_rate / math.sqrt(num_targets) if num_targets > 1 else 0.0
    se_chosen_info_gain = std_chosen_info_gain / math.sqrt(len(target_chosen_igs)) if std_chosen_info_gain is not None and target_chosen_igs and len(target_chosen_igs) > 1 else None
    se_optimal_info_gain = std_optimal_info_gain / math.sqrt(len(target_optimal_igs)) if std_optimal_info_gain is not None and target_optimal_igs and len(target_optimal_igs) > 1 else None
    se_questions_considered_per_turn = std_questions_considered_per_turn / math.sqrt(len(target_questions)) if std_questions_considered_per_turn is not None and target_questions and len(target_questions) > 1 else None
    
    return {
        "total_conversations": total_conversations,
        "total_evaluated": success_count + skipped_count,
        "total_success": success_count,
        "total_skipped": skipped_count,
        "total_errors": error_count,
        "aggregate_statistics": {
            "total_turns_evaluated": total_turns_evaluated,
            "total_optimal_choices": total_optimal_choices,
            "avg_optimal_choice_rate": avg_optimal_choice_rate,
            "std_optimal_choice_rate": std_optimal_choice_rate,
            "se_optimal_choice_rate": se_optimal_choice_rate,
            "avg_chosen_info_gain": avg_chosen_info_gain,
            "std_chosen_info_gain": std_chosen_info_gain,
            "se_chosen_info_gain": se_chosen_info_gain,
            "avg_optimal_info_gain": avg_optimal_info_gain,
            "std_optimal_info_gain": std_optimal_info_gain,
            "se_optimal_info_gain": se_optimal_info_gain,
            "avg_questions_considered_per_turn": avg_questions_considered_per_turn,
            "std_questions_considered_per_turn": std_questions_considered_per_turn,
            "se_questions_considered_per_turn": se_questions_considered_per_turn,
            "total_connection_errors": total_connection_errors
        },
        "by_target": by_target_summary
    }


def find_cot_runs_csvs(base_dir: Path) -> List[Path]:
    """Return all CoT runs.csv files under base_dir (same logic as synthesize_traces.py)."""
    return sorted(
        p for p in base_dir.rglob("runs.csv")
        if (p.parent.name.endswith("_cot") or "_cot_" in p.parent.name)
        and "no_cot" not in p.parent.name
    )


def run_for_csv(
    runs_csv_path: Path,
    outputs_base_dir: Path,
    oracle_config: LLMConfig,
    pruner_config: LLMConfig,
    unified_jsonl: Path,
    done_keys: Set[str],
    max_workers: int,
    force: bool,
    dry_run: bool,
    dataset_csv: Optional[Path],
    only_run_index: Optional[int] = None,
    sample_filter: Optional[Set[Path]] = None,
    sample_indices: Optional[List[int]] = None,
) -> int:
    """Process a single runs.csv. Returns 0 on success, 1 if any errors.

    ``sample_filter``: if given, only conversations whose resolved path is in
    this set are evaluated (used by --sample-indices to mirror the exact
    question_classification / synthesize_traces sample).
    """
    try:
        conversation_dirs = find_conversation_dirs_from_runs_csv(
            runs_csv_path, outputs_base_dir,
            only_run_index=only_run_index,
            sample_indices=sample_indices,
        )
    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        return 1

    if sample_filter is not None:
        conversation_dirs = [
            c for c in conversation_dirs if c.resolve() in sample_filter
        ]

    if not conversation_dirs:
        logger.warning("No conversation directories found in %s — skipping", runs_csv_path)
        return 0

    logger.info("📊 %s — %d conversations", runs_csv_path.parent.name, len(conversation_dirs))

    if dry_run:
        for i, conv_dir in enumerate(conversation_dirs, 1):
            seeker_key = _seeker_path_key(conv_dir)
            status = "Skip (in unified JSONL)" if seeker_key in done_keys else "Process"
            logger.info("[%d/%d] %s: %s", i, len(conversation_dirs), status, conv_dir)
        return 0

    results = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_conversation,
                    conv_dir, oracle_config, pruner_config,
                    unified_jsonl, done_keys, force, dataset_csv,
                ): conv_dir
                for conv_dir in conversation_dirs
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                emoji = "✅" if result["status"] == "success" else "⏭️" if result["status"] == "skipped" else "❌"
                logger.info("[%d/%d] %s %s: %s", i, len(conversation_dirs), emoji,
                            result["status"].capitalize(), result["conversation_dir"])
    else:
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            def _tqdm(x, **kw): return x
        for conv_dir in _tqdm(conversation_dirs, desc="Evaluating", unit="conv"):
            result = process_single_conversation(
                conv_dir, oracle_config, pruner_config,
                unified_jsonl, done_keys, force, dataset_csv,
            )
            results.append(result)

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")

    # Aggregate from the unified JSONL filtered by this experiment.
    evaluations = _load_evaluations_for_exp(unified_jsonl, runs_csv_path.parent)
    aggregate_summary = calculate_aggregate_summary(
        evaluations, success_count, skipped_count, error_count, len(conversation_dirs),
    )
    logger.info("  ✅ %d ok | ⏭️  %d skip | ❌ %d err",
                success_count, skipped_count, error_count)

    summary_path = runs_csv_path.parent / "question_evaluations_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)
    logger.info("💾 Summary → %s", summary_path)

    return 1 if error_count > 0 else 0


def main():
    """Main entry point for batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Seeker's question choices for all conversations in a runs.csv"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "runs_csv_path",
        type=Path,
        nargs="?",
        help="Path to a single runs.csv file"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all CoT runs.csv under --outputs-base-dir"
    )
    parser.add_argument(
        "--outputs-base-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory where conversation paths in runs.csv are relative to (default: ./outputs)"
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help="Path to domain CSV. Auto-detected from target id if omitted."
    )
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="Qwen3-8B",
        help="LLM model for Oracle simulation — resolved via configs/servers.yaml (default: Qwen3-8B)"
    )
    parser.add_argument(
        "--pruner-model",
        type=str,
        default="Qwen3-8B",
        help="LLM model for Pruner simulation — resolved via configs/servers.yaml (default: Qwen3-8B)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for LLM API. If omitted, resolved from configs/servers.yaml using the model name."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="NINGUEM-TA-PURO-2K26",
        help="API key for LLM (default: NINGUEM-TA-PURO-2K26)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for LLM generation (default: None, API decides)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate conversations even if already in the unified JSONL "
             "(new records are appended; the old line is left in place)"
    )
    parser.add_argument(
        "--no-migrate",
        action="store_true",
        help="Skip the one-time migration of legacy per-conversation "
             "question_evaluation.json files into the unified JSONL.",
    )
    parser.add_argument(
        "--only-run-index",
        type=int,
        default=None,
        help="If given, evaluate only conversations whose runs.csv row has this "
             "run_index (e.g. 1 to keep just run01).",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated 0-based row positions within the (run-index-filtered) "
             "runs.csv to evaluate (e.g. '10,20,30,...,150'). Mirrors judge_eval behaviour.",
    )
    parser.add_argument(
        "--seekers",
        type=str,
        default=None,
        help="Comma-separated seeker model slugs to process with --all "
             "(e.g. 'Qwen3-8B,google-gemma-4-31B-it'). '/' → '-'. Only with --all.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    parser.add_argument(
        "--traces-jsonl",
        type=Path,
        default=None,
        help=("Aggregate seeker_traces JSONL to read traces from. "
              "Default: <outputs-base-dir>/seeker_traces.jsonl. Use this to "
              "point at a separate aggregate (ex.: seeker_traces_gemma.jsonl).")
    )
    parser.add_argument(
        "--unified-jsonl",
        type=Path,
        default=None,
        help=("Output unified JSONL pra gravar/ler as avaliações. "
              "Default: <outputs-base-dir>/question_evaluations.jsonl. Use pra "
              "manter um eval separado (ex.: question_evaluations_gemma.jsonl) "
              "sem misturar com o agregado legado.")
    )

    args = parser.parse_args()
    
    ClaryLogger.configure()

    logger.info("🔍 Evaluate Seeker Question Choices")
    logger.info("🤖 Oracle: %s | Pruner: %s", args.oracle_model, args.pruner_model)
    logger.info("⚙️  workers=%d  force=%s", args.max_workers, args.force)

    oracle_base_url = _resolve_base_url(args.oracle_model, args.base_url)
    pruner_base_url = _resolve_base_url(args.pruner_model, args.base_url)
    oracle_config = LLMConfig(
        model=args.oracle_model, api_key=args.api_key,
        base_url=oracle_base_url, temperature=args.temperature,
    )
    pruner_config = LLMConfig(
        model=args.pruner_model, api_key=args.api_key,
        base_url=pruner_base_url, temperature=args.temperature,
    )

    # Resolve to absolute path so seeker_path keys match what synthesize_traces.py
    # writes (it uses an absolute OUTPUTS_BASE = project_root / "outputs"). A
    # relative outputs_base_dir produces relative seeker_path keys that never
    # match the absolute keys in seeker_traces.jsonl → silent empty results.
    args.outputs_base_dir = args.outputs_base_dir.resolve()

    unified_jsonl = (
        args.unified_jsonl.resolve() if args.unified_jsonl
        else unified_jsonl_path(args.outputs_base_dir)
    )
    done_keys = _load_done_index(unified_jsonl)
    logger.info("📚 Unified JSONL: %s (%d already evaluated)", unified_jsonl, len(done_keys))
    if not args.no_migrate:
        migrate_per_conv_files(args.outputs_base_dir, unified_jsonl, done_keys)

    # Pre-load the seeker_traces.jsonl index in the main thread. The file is
    # multi-GB; making 32 workers race the load via the GIL stalls everything
    # for minutes (the C-level splitlines/JSON parse never yields). Loading
    # once here populates the module-level cache for all workers to reuse.
    # --traces-jsonl override (caso queira um agregado alternativo, ex.:
    # outputs/seeker_traces_gemma.jsonl) é propagado pro evaluator via
    # set_traces_jsonl, e usado aqui no pre-load.
    seeker_traces_jsonl = (
        args.traces_jsonl.resolve() if args.traces_jsonl
        else args.outputs_base_dir / "seeker_traces.jsonl"
    )
    set_traces_jsonl(seeker_traces_jsonl)
    if seeker_traces_jsonl.exists():
        logger.info("📚 Pre-loading seeker_traces index (%s)…", seeker_traces_jsonl)
        _get_jsonl_index(seeker_traces_jsonl)

    # Comportamento padrão definitivo: as conversas a avaliar são EXATAMENTE
    # as presentes no agregado seeker_traces.jsonl (`--traces-jsonl`). Garante
    # match 1-pra-1 com o que o synthesize_traces produziu — sem mismatch de
    # ordering (runs.csv-row-order × dir-alfabético) que causava 90%+ Skipped.
    trace_index = _get_jsonl_index(seeker_traces_jsonl) if seeker_traces_jsonl.exists() else {}
    if not trace_index:
        logger.error("seeker_traces jsonl vazio ou ausente: %s", seeker_traces_jsonl)
        return 1
    sample_filter: Optional[Set[Path]] = {
        Path(seeker_path).parent.resolve() for seeker_path in trace_index
    }
    logger.info(
        "🎯 traces como fonte: %d conversas no %s",
        len(sample_filter), seeker_traces_jsonl.name,
    )

    sample_indices: Optional[List[int]] = None
    if args.sample_indices:
        sample_indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]

    allowed_seekers: Optional[set] = None
    if args.seekers:
        allowed_seekers = {_slug(s.strip()) for s in args.seekers.split(",") if s.strip()}

    if args.all:
        runs_csvs = find_cot_runs_csvs(args.outputs_base_dir)
        if allowed_seekers is not None:
            runs_csvs = [
                csv for csv in runs_csvs
                if _triple_parts(csv.parent.parent.name) is not None
                and _triple_parts(csv.parent.parent.name)[0] in allowed_seekers
            ]
        logger.info("🔎 %d CoT runs.csv found under %s", len(runs_csvs), args.outputs_base_dir)
        if args.only_run_index is not None:
            logger.info("🎯 Filtering to run_index=%d only", args.only_run_index)
        if sample_indices is not None:
            logger.info("🎯 sample_indices: %s", args.sample_indices)
        any_error = False
        for i, runs_csv in enumerate(runs_csvs, 1):
            logger.info("\n[%d/%d] %s", i, len(runs_csvs), runs_csv.parent.name)
            rc = run_for_csv(
                runs_csv, args.outputs_base_dir, oracle_config, pruner_config,
                unified_jsonl, done_keys,
                args.max_workers, args.force, args.dry_run, args.dataset_csv,
                only_run_index=args.only_run_index,
                sample_filter=sample_filter,
                sample_indices=sample_indices,
            )
            if rc != 0:
                any_error = True
        return 1 if any_error else 0
    else:
        if args.runs_csv_path is None:
            logger.error("Provide a runs.csv path or use --all")
            return 1
        return run_for_csv(
            args.runs_csv_path, args.outputs_base_dir, oracle_config, pruner_config,
            unified_jsonl, done_keys,
            args.max_workers, args.force, args.dry_run, args.dataset_csv,
            only_run_index=args.only_run_index,
            sample_filter=sample_filter,
            sample_indices=sample_indices,
        )


if __name__ == "__main__":
    exit(main())

