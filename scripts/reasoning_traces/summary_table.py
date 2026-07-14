#!/usr/bin/env python3
"""Generate the decision-quality summary table (Table 2 in the paper).

Reads outputs/question_evaluations_unified.csv (produced by
scripts/generate_question_evaluations_csv.py, which in turn requires
evaluate_all_seeker_choices.py to have been run) and aggregates by
(seeker model, observability) to produce mean ± std for:

    Avg Optimal Rate   — proportion of turns selecting the max-IG candidate
    Avg Chosen IG      — mean IG of the executed question (bits)
    Avg Optimal IG     — mean IG of the best candidate considered (bits)
    Avg Questions/Turn — candidates considered per turn

Run order:
    python3 scripts/evaluate_all_seeker_choices.py          # or evaluate_all_seeker_choices
    python3 scripts/generate_question_evaluations_csv.py
    python3 scripts/reasoning_traces/summary_table.py

Output:
    outputs/decision_quality_table.csv     machine-readable (raw numbers)
    outputs/decision_quality_table_fmt.csv human-readable  (mean ± std strings)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


MODEL_ALIASES: dict[str, str] = {
    # Add short display names here if needed, e.g.:
    # "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen3-235B",
}

OBS_MAP = {
    "FULLY_OBSERVABLE": "FO",
    "PARTIALLY_OBSERVABLE": "PO",
    "fully_observable": "FO",
    "partially_observable": "PO",
}

METRICS = [
    ("Avg Optimal Choice Rate",           "SE Optimal Choice Rate",           "avg_optimal_rate",    "se_optimal_rate"),
    ("Avg Chosen Info Gain",              "SE Chosen Info Gain",              "avg_chosen_ig",       "se_chosen_ig"),
    ("Avg Optimal Info Gain",             "SE Optimal Info Gain",             "avg_optimal_ig",      "se_optimal_ig"),
    ("Avg Questions Considered Per Turn", "SE Questions Considered Per Turn", "avg_questions_turn",  "se_questions_turn"),
]


def load_unified(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names to be tolerant of minor variations
    df.columns = [c.strip() for c in df.columns]
    return df


def build_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw_df, formatted_df) grouped by (model, obs)."""
    df = df.copy()
    df["model"] = df["Seeker Model"].map(lambda m: MODEL_ALIASES.get(str(m), str(m)))
    df["obs"] = df["Observabilidade"].map(lambda o: OBS_MAP.get(str(o), str(o)))

    # Weighted aggregate: each row is already a per-experiment summary.
    # We take the mean of the per-experiment averages (unweighted across experiments
    # within the same model+obs group) and propagate SE via root-sum-of-squares.
    group = df.groupby(["model", "obs"], sort=False)

    rows_raw: list[dict] = []
    rows_fmt: list[dict] = []

    for (model, obs), g in group:
        raw: dict = {"model": model, "obs": obs, "n_experiments": len(g)}
        fmt: dict = {"model": model, "obs": obs, "n_experiments": len(g)}

        for avg_col, se_col, avg_key, se_key in METRICS:
            if avg_col not in g.columns:
                raw[avg_key] = None
                raw[se_key] = None
                fmt[avg_key] = "N/A"
                continue
            avg_vals = g[avg_col].dropna()
            se_vals = g[se_col].dropna() if se_col in g.columns else pd.Series(dtype=float)

            mean = avg_vals.mean()
            # Combined SE across experiments (root mean square of individual SEs)
            se = (se_vals ** 2).mean() ** 0.5 if len(se_vals) > 0 else float("nan")

            raw[avg_key] = round(mean, 4)
            raw[se_key] = round(se, 4)
            fmt[avg_key] = f"{mean:.2f} ± {se:.2f}"

        rows_raw.append(raw)
        rows_fmt.append(fmt)

    col_order_raw = ["model", "obs", "n_experiments",
                     "avg_optimal_rate", "se_optimal_rate",
                     "avg_chosen_ig", "se_chosen_ig",
                     "avg_optimal_ig", "se_optimal_ig",
                     "avg_questions_turn", "se_questions_turn"]
    col_order_fmt = ["model", "obs", "n_experiments",
                     "avg_optimal_rate", "avg_chosen_ig", "avg_optimal_ig", "avg_questions_turn"]

    raw_df = pd.DataFrame(rows_raw)[[c for c in col_order_raw if c in pd.DataFrame(rows_raw).columns]]
    fmt_df = pd.DataFrame(rows_fmt)[[c for c in col_order_fmt if c in pd.DataFrame(rows_fmt).columns]]
    return raw_df.sort_values(["model", "obs"]).reset_index(drop=True), \
           fmt_df.sort_values(["model", "obs"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/question_evaluations_unified.csv"),
        help="CSV produced by generate_question_evaluations_csv.py.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
    )
    args = p.parse_args()

    if not args.input_csv.exists():
        print(
            f"ERROR: {args.input_csv} not found.\n"
            "Run scripts/generate_question_evaluations_csv.py first "
            "(which requires evaluate_all_seeker_choices.py to have been run).",
            file=sys.stderr,
        )
        return 1

    df = load_unified(args.input_csv)
    print(f"Loaded {len(df)} experiment rows from {args.input_csv}")

    raw_df, fmt_df = build_table(df)

    raw_path = args.out_dir / "decision_quality_table.csv"
    fmt_path = args.out_dir / "decision_quality_table_fmt.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    fmt_df.to_csv(fmt_path, index=False)

    print(f"\nWrote: {raw_path}")
    print(f"Wrote: {fmt_path}\n")

    with pd.option_context("display.max_rows", None, "display.width", 120, "display.max_columns", None):
        print(fmt_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
