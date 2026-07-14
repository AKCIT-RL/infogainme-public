#!/usr/bin/env python3
"""Aggregate the flat question-classification CSV into analysis artifacts.

Reads the CSV produced by ``flatten_question_classifications.py`` and writes
CSVs designed for plotting and reporting in a notebook:

    by_model.csv              seeker × question_type counts + fractions
    by_model_turn.csv         seeker × turn × question_type (for the
                               "binary-search curve" plot)
    redundancy_by_model.csv   seeker × redundancy counts + rates
    subclass_by_model.csv     seeker × subclass counts (top subclasses only)
    by_model_summary.csv      headline table per seeker:
                                 malformed_rate, redundancy_rate,
                                 hierarchical_rate, fine_grained_rate,
                                 comparative_rate, direct_guess_rate,
                                 semantic_rate, lexical_rate

Run order:

    python3 scripts/question_classification/flatten_question_classifications.py
    python3 scripts/question_classification/analyze_question_classifications.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_flat(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"run_index": "Int64", "redundant_with_turn": "Int64"})
    # Drop rows that couldn't be classified (non-empty error column).
    errored = df["error"].notna() & (df["error"].astype(str) != "") & (df["error"].astype(str) != "nan")
    if errored.any():
        print(f"  note: {int(errored.sum())} turn(s) had classification errors and are excluded from rates.")
    return df[~errored].copy()




# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


QUESTION_TYPES = ["semantic", "lexical", "direct_guess", "malformed", "other"]
REDUNDANCY_TYPES = ["none", "exact_duplicate", "semantic_equivalent", "strictly_implied"]


def _crosstab_with_rates(df: pd.DataFrame, row: str, col: str, values: list[str]) -> pd.DataFrame:
    """Cross-tab of counts + fractions, with a stable column order."""
    ct = pd.crosstab(df[row], df[col])
    # Ensure every expected label appears, even with zero.
    for v in values:
        if v not in ct.columns:
            ct[v] = 0
    ct = ct[values]
    ct["total"] = ct.sum(axis=1)
    for v in values:
        ct[f"{v}_frac"] = ct[v] / ct["total"].where(ct["total"] > 0, 1)
    return ct.reset_index()


def by_model(df: pd.DataFrame) -> pd.DataFrame:
    return _crosstab_with_rates(df, "seeker", "question_type", QUESTION_TYPES)


def by_model_turn(df: pd.DataFrame, max_turn: int = 30) -> pd.DataFrame:
    """Long-format table: one row per (seeker, turn, question_type) with count and fraction.

    Easy to plot with seaborn/matplotlib: ``x=turn, y=frac, hue=question_type, col=seeker``.
    """
    capped = df[df["turn"] <= max_turn].copy()
    grp = capped.groupby(["seeker", "turn", "question_type"]).size().rename("count").reset_index()
    totals = capped.groupby(["seeker", "turn"]).size().rename("total").reset_index()
    merged = grp.merge(totals, on=["seeker", "turn"], how="left")
    merged["frac"] = merged["count"] / merged["total"]
    return merged


def redundancy_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return _crosstab_with_rates(df, "seeker", "redundancy", REDUNDANCY_TYPES)


def _explode_subclasses(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the ``;``-joined ``subclasses`` column into one row per tag."""
    if "subclasses" not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["subclass"])
    expanded = df.assign(
        subclass=df["subclasses"].fillna("").astype(str).str.split(";")
    ).explode("subclass")
    expanded["subclass"] = expanded["subclass"].astype(str).str.strip()
    return expanded[expanded["subclass"] != ""]


def subclass_by_model(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top-N subclass tags per seeker, long format.

    Counts each tag individually: a turn tagged ``["comparative", "quantitative_threshold"]``
    contributes to both rows.
    """
    sc = _explode_subclasses(df)
    if sc.empty:
        return pd.DataFrame(columns=["seeker", "subclass", "count", "rank"])
    counts = sc.groupby(["seeker", "subclass"]).size().rename("count").reset_index()
    counts["rank"] = counts.groupby("seeker")["count"].rank(ascending=False, method="dense").astype(int)
    return counts[counts["rank"] <= top_n].sort_values(["seeker", "rank"]).reset_index(drop=True)


def by_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: one row per seeker model (question-classification metrics only).

    Subclass-derived rates (``hierarchical_rate``, ``fine_grained_rate``,
    ``comparative_rate``) are computed against the total turn count: each turn
    that carries the tag in its ``subclasses`` list counts once, regardless of
    how many other tags the same turn has.
    """
    exploded = _explode_subclasses(df)
    per_seeker_tag = (
        exploded.groupby(["seeker", "subclass"]).size().unstack(fill_value=0)
        if not exploded.empty
        else pd.DataFrame()
    )

    rows: list[dict] = []
    for seeker, g in df.groupby("seeker"):
        total = len(g)
        if total == 0:
            continue
        qt = g["question_type"].value_counts()
        red = g["redundancy"].value_counts()
        tags = per_seeker_tag.loc[seeker] if seeker in per_seeker_tag.index else pd.Series(dtype=int)

        redundant_n = sum(red.get(k, 0) for k in ("exact_duplicate", "semantic_equivalent", "strictly_implied"))
        rows.append({
            "seeker": seeker,
            "n_turns": total,
            "n_games": g[["experiment", "target", "run_index"]].drop_duplicates().shape[0],
            "malformed_rate": qt.get("malformed", 0) / total,
            "redundancy_rate": redundant_n / total,
            "hierarchical_rate": tags.get("hierarchical_category", 0) / total,
            "fine_grained_rate": tags.get("fine_grained_category", 0) / total,
            "comparative_rate": tags.get("comparative", 0) / total,
            "direct_guess_rate": qt.get("direct_guess", 0) / total,
            "semantic_rate": qt.get("semantic", 0) / total,
            "lexical_rate": qt.get("lexical", 0) / total,
        })
    return pd.DataFrame(rows).sort_values("seeker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--flat-csv",
        type=Path,
        default=Path("outputs/question_classifications.csv"),
        help="Flat CSV produced by flatten_question_classifications.py.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/question_classifications_analysis"),
    )
    p.add_argument("--subclass-top-n", type=int, default=15)
    p.add_argument("--max-turn", type=int, default=30)
    args = p.parse_args()

    if not args.flat_csv.exists():
        print(
            f"ERROR: {args.flat_csv} not found. Run flatten_question_classifications.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {args.flat_csv}...")
    df = load_flat(args.flat_csv)
    print(f"  {len(df)} turns × {df['seeker'].nunique()} seeker model(s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {
        "by_model.csv": by_model(df),
        "by_model_turn.csv": by_model_turn(df, max_turn=args.max_turn),
        "redundancy_by_model.csv": redundancy_by_model(df),
        "subclass_by_model.csv": subclass_by_model(df, top_n=args.subclass_top_n),
        "by_model_summary.csv": by_model_summary(df),
    }

    for name, tbl in tables.items():
        out = args.out_dir / name
        tbl.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(tbl)} rows)")

    # Print the headline summary inline — easiest to eyeball.
    summary = tables["by_model_summary.csv"]
    if not summary.empty:
        print("\n=== by_model_summary (headline) ===")
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", None):
            print(summary.round(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
