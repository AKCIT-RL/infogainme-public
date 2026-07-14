"""Aggregate belief-state extraction results into the scaffold-claim evidence.

Reads the unified ``outputs/belief_states.jsonl`` produced by
``extract_belief_states.py`` and produces, per observability mode (FO/IO/PO):

  * belief-tracking fidelity (kept_precision, zombie rate, explicit-tracking rate,
    fatal target-exclusion rate), and
  * the key mechanistic test: the correlation between a conversation's belief
    fidelity and its information gain per turn.

The paper's claim that CoT scaffolds *belief tracking* (not generic reasoning)
predicts that fidelity should correlate with IG in the implicit-tracking modes
(IO, PO) but not in FO, where the active set is always shown.

Writes ``outputs/belief_states_by_mode.csv`` and ``figures/belief_state_scaffold.png``.

Run:
    .venv/bin/python scripts/reasoning_traces/analyze_belief_states.py \
        --jsonl outputs/belief_states.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Metric definitions live here (the analyzer is the single source of truth); we
# reuse only the name-normalization / pool-matching helpers from the extractor.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.belief_state_extraction import _norm, match_to_pool  # noqa: E402

MODES = ["FO", "IO", "PO"]
_MODE_RE = {"FO": re.compile(r"_fo_"), "IO": re.compile(r"_io_"), "PO": re.compile(r"_po_")}


def infer_mode(experiment_dir: str) -> Optional[str]:
    """Infer FO/IO/PO from an experiment directory name."""
    name = Path(experiment_dir).name
    for mode, rx in _MODE_RE.items():
        if rx.search(name):
            return mode
    return None


def infer_seeker(experiment_dir: str) -> str:
    """Extract the seeker slug from the ``s_<seeker>__o_..__p_..`` triple dir."""
    triple = Path(experiment_dir).parent.name
    if triple.startswith("s_") and "__o_" in triple:
        return triple[2:].split("__o_")[0]
    return triple


def _pearson(xs: list[float], ys: list[float]) -> tuple[Optional[float], Optional[float]]:
    if len(xs) < 5:
        return None, None
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(xs, ys)
        return float(r), float(p)
    except Exception:  # noqa: BLE001
        return None, None


def _mean(values: list[Any]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def _median(values: list[Any]) -> Optional[float]:
    nums = sorted(v for v in values if isinstance(v, (int, float)))
    if not nums:
        return None
    n = len(nums)
    mid = n // 2
    return nums[mid] if n % 2 else (nums[mid - 1] + nums[mid]) / 2


def _se(values: list[Any]) -> Optional[float]:
    """Standard error of the mean over the given (per-conversation) values.

    Used with one value per conversation so the SE is cluster-robust (turns are
    nested within conversations and are not independent).
    """
    nums = [v for v in values if isinstance(v, (int, float))]
    n = len(nums)
    if n < 2:
        return None
    mean = sum(nums) / n
    var = sum((v - mean) ** 2 for v in nums) / (n - 1)  # sample variance
    return (var / n) ** 0.5


def load_records(jsonl: Path) -> list[dict[str, Any]]:
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def turn_metrics(belief: dict[str, Any], omega_labels: list[str],
                 pool_index: dict[str, str], target_label: Optional[str]) -> dict[str, Any]:
    """Compute all per-turn belief metrics from the raw belief and Omega_t labels.

    This is the authoritative metric definition: it works directly off the stored
    ``belief`` lists and ``omega_labels`` (the true active set), so metrics can be
    changed by re-running the analyzer alone -- no re-extraction needed.

    Args:
        belief: ``{constraints, kept_candidates, excluded_candidates, ...}``.
        omega_labels: labels of the true active set Omega_t this turn.
        pool_index: normalized-label -> canonical-label map for Omega_0.
        target_label: the hidden target's label (or None).

    Returns:
        Dict with usage flags, sizes, and set-similarity vs Omega_t (precision /
        recall / Jaccard), all None where undefined (no named candidates).
    """
    omega = set(omega_labels)
    kept = match_to_pool(belief.get("kept_candidates", []), pool_index)
    excluded = match_to_pool(belief.get("excluded_candidates", []), pool_index)
    constraints = belief.get("constraints", [])
    n_named_raw = len(belief.get("kept_candidates", [])) + len(belief.get("excluded_candidates", []))

    # Size-estimate error: the model's stated remaining count vs the true |Omega_t|.
    believed_count = belief.get("believed_count")
    count_abs_error = count_sq_error = None
    if isinstance(believed_count, int):
        err = believed_count - len(omega)
        count_abs_error = abs(err)
        count_sq_error = err * err

    m: dict[str, Any] = {
        "omega_size": len(omega),
        "used_enum": n_named_raw > 0,
        "used_constraint": len(constraints) > 0,
        "n_named": n_named_raw,
        "n_constraints": len(constraints),
        "precision": None,
        "recall": None,
        "jaccard": None,
        "believed_count": believed_count,
        "count_abs_error": count_abs_error,
        "count_sq_error": count_sq_error,
        "target_drop": bool(target_label and target_label in omega and target_label in excluded),
    }
    if kept:
        inter = len(kept & omega)
        union = len(kept | omega)
        m["precision"] = inter / len(kept)
        m["recall"] = inter / len(omega) if omega else 1.0
        m["jaccard"] = inter / union if union else 1.0
    elif not omega:
        # Both the believed and true sets are empty -> identical by convention.
        m["precision"] = m["recall"] = m["jaccard"] = 1.0
    # else: model named nothing while Omega_t is non-empty -> left None (this is
    # "did not enumerate", handled by Construct 1, not an accuracy failure).
    return m


def record_turns(rec: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[float]]:
    """Recompute per-turn metrics for one conversation + its IG/turn.

    Uses turn-0 ``omega_labels`` as the candidate pool Omega_0.
    """
    turns = rec.get("turns", [])
    if not turns:
        return [], None
    pool_labels = turns[0].get("omega_labels") or []
    pool_index = {_norm(lbl): lbl for lbl in pool_labels}
    target_label = rec.get("target_label")
    out = []
    for i, t in enumerate(turns):
        m = turn_metrics(t.get("belief", {}), t.get("omega_labels") or [], pool_index, target_label)
        # Each prior turn answers one question, so by turn position i there are i
        # established constraints; completeness = how many the model still restates.
        m["turn_position"] = i
        m["constraint_completeness"] = (m["n_constraints"] / i) if i > 0 else None
        out.append(m)
    igs = [t.get("info_gain") for t in turns if isinstance(t.get("info_gain"), (int, float))]
    ig_per_turn = sum(igs) / len(igs) if igs else None
    return out, ig_per_turn


def _aggregate_group(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the full metric dict for one group of conversations.

    Shared by the per-mode and per-(seeker, mode) aggregations so both expose
    identical columns. Metrics are recomputed from raw belief + omega_labels.
    """
    # Pooled-over-turns lists give the point estimates (each turn weighted equally).
    enum_flags, constraint_flags, any_flags = [], [], []
    prec_turns, recall_turns, jaccard_turns, drop_turns = [], [], [], []
    n_named_turns, n_constr_turns = [], []
    count_abs_errs, count_sq_errs = [], []
    # Per-conversation values give cluster-robust SEs (conversation = unit; turns
    # are nested within conversations, so they are not independent).
    conv: dict[str, list[float]] = defaultdict(list)
    pairs: list[tuple[float, float]] = []  # (conv mean precision, conv IG/turn)
    completeness_turns: list[float] = []   # n_constraints / turn_position (pooled)
    constr_turn_corrs: list[float] = []    # per-conv corr(n_constraints, turn_position)

    n_turns = 0
    for rec in recs:
        tms, ig_per_turn = record_turns(rec)
        c_enum, c_constr, c_any = [], [], []
        c_named, c_ncon, c_abs, c_sq = [], [], [], []
        c_prec, c_recall, c_jac, c_drop = [], [], [], []
        c_comp, cc_xs, cc_ys = [], [], []
        for m in tms:
            n_turns += 1
            enum_flags.append(1.0 if m["used_enum"] else 0.0); c_enum.append(1.0 if m["used_enum"] else 0.0)
            constraint_flags.append(1.0 if m["used_constraint"] else 0.0); c_constr.append(1.0 if m["used_constraint"] else 0.0)
            tracked = 1.0 if (m["used_enum"] or m["used_constraint"]) else 0.0
            any_flags.append(tracked); c_any.append(tracked)
            drop_turns.append(1.0 if m["target_drop"] else 0.0); c_drop.append(1.0 if m["target_drop"] else 0.0)
            if m["used_enum"]:
                n_named_turns.append(m["n_named"]); c_named.append(m["n_named"])
            if m["used_constraint"]:
                n_constr_turns.append(m["n_constraints"]); c_ncon.append(m["n_constraints"])
            if isinstance(m["count_abs_error"], (int, float)):
                count_abs_errs.append(m["count_abs_error"]); c_abs.append(m["count_abs_error"])
                count_sq_errs.append(m["count_sq_error"]); c_sq.append(m["count_sq_error"])
            # Constraint accumulation: completeness ratio + (count, position) pairs.
            if isinstance(m["constraint_completeness"], (int, float)):
                completeness_turns.append(m["constraint_completeness"])
                c_comp.append(m["constraint_completeness"])
            cc_xs.append(m["turn_position"]); cc_ys.append(m["n_constraints"])
            # Quality only defined when pool-matched named candidates exist.
            if isinstance(m["precision"], (int, float)):
                prec_turns.append(m["precision"]); c_prec.append(m["precision"])
                if isinstance(m["recall"], (int, float)):
                    recall_turns.append(m["recall"]); c_recall.append(m["recall"])
                if isinstance(m["jaccard"], (int, float)):
                    jaccard_turns.append(m["jaccard"]); c_jac.append(m["jaccard"])
        # Per-conversation correlation between #constraints and turn position.
        cr, _ = _pearson(cc_xs, cc_ys)
        if cr is not None:
            constr_turn_corrs.append(cr)
        if c_comp:
            conv["constraint_completeness"].append(_mean(c_comp))
        # One value per conversation (its own mean) feeds the SE.
        if c_any:
            conv["belief_tracking_rate"].append(_mean(c_any))
            conv["enumerative_rate"].append(_mean(c_enum))
            conv["constraint_rate"].append(_mean(c_constr))
            conv["target_drop_rate"].append(_mean(c_drop))
        if c_named: conv["n_named_mean"].append(_mean(c_named))
        if c_ncon: conv["n_constraints_mean"].append(_mean(c_ncon))
        if c_abs:
            conv["count_mae"].append(_mean(c_abs))
            conv["count_mse"].append(_mean(c_sq))
        if c_prec:
            conv["belief_precision"].append(_mean(c_prec))
            conv["belief_recall"].append(_mean(c_recall))
            conv["belief_jaccard"].append(_mean(c_jac))
            if isinstance(ig_per_turn, (int, float)):
                pairs.append((_mean(c_prec), ig_per_turn))

    r, p = _pearson([a for a, _ in pairs], [b for _, b in pairs])
    # Naming follows the paper: the believed set is the estimate \hat{\Omega}_t of
    # the true active set \Omega_t; set similarity is reported as Jaccard (as in
    # the paper's "Pruner Jaccard"), decomposed into precision and recall.
    # Means are pooled over turns; SEs are cluster-robust (one value per
    # conversation, SE across conversations).
    return {
        "n_conversations": len(recs),
        "n_turns": n_turns,
        # Construct 1: does the model track a belief at all (per turn)?
        "belief_tracking_rate": _mean(any_flags),
        "belief_tracking_rate_se": _se(conv["belief_tracking_rate"]),
        "enumerative_rate": _mean(enum_flags),
        "enumerative_rate_se": _se(conv["enumerative_rate"]),
        "constraint_rate": _mean(constraint_flags),
        "constraint_rate_se": _se(conv["constraint_rate"]),
        # Belief size (over turns that use that form)
        "n_named_mean": _mean(n_named_turns),
        "n_named_se": _se(conv["n_named_mean"]),
        "n_named_median": _median(n_named_turns),
        "n_constraints_mean": _mean(n_constr_turns),
        "n_constraints_se": _se(conv["n_constraints_mean"]),
        "n_constraints_median": _median(n_constr_turns),
        # Constraint accumulation (each prior turn answers one question -> +1 constraint)
        "constraint_completeness": _mean(completeness_turns),
        "constraint_completeness_se": _se(conv["constraint_completeness"]),
        "constraint_turn_corr": _mean(constr_turn_corrs),
        # Size-estimate error of stated remaining count vs |Omega_t|
        "n_turns_with_count": len(count_abs_errs),
        "count_mae": _mean(count_abs_errs),
        "count_mae_se": _se(conv["count_mae"]),
        "count_mse": _mean(count_sq_errs),
        "count_rmse": (_mean(count_sq_errs) ** 0.5) if count_sq_errs else None,
        # Construct 2: how well does the belief match \Omega_t (when it names any)?
        "n_turns_with_belief": len(prec_turns),
        "belief_jaccard": _mean(jaccard_turns),
        "belief_jaccard_se": _se(conv["belief_jaccard"]),
        "belief_precision": _mean(prec_turns),
        "belief_precision_se": _se(conv["belief_precision"]),
        "belief_recall": _mean(recall_turns),
        "belief_recall_se": _se(conv["belief_recall"]),
        "target_drop_rate": _mean(drop_turns),
        "target_drop_rate_se": _se(conv["target_drop_rate"]),
        # Mechanistic link (per-conversation belief precision vs IG/turn)
        "corr_belief_ig": r,
        "corr_belief_ig_p": p,
        "corr_n": len(pairs),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-mode aggregate stats (FO/IO/PO)."""
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        mode = infer_mode(rec.get("experiment_dir", ""))
        if mode:
            by_mode[mode].append(rec)
    return {mode: _aggregate_group(by_mode.get(mode, [])) for mode in MODES}


def aggregate_by_seeker(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per-(seeker, mode) aggregate stats, same columns as ``aggregate``."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        mode = infer_mode(rec.get("experiment_dir", ""))
        seeker = infer_seeker(rec.get("experiment_dir", ""))
        if mode:
            groups[(seeker, mode)].append(rec)
    return {k: _aggregate_group(v) for k, v in groups.items()}


_CSV_COLS = [
    "mode", "n_conversations", "n_turns",
    # Construct 1: does the model track a belief at all? (+ cluster-robust SE)
    "belief_tracking_rate", "belief_tracking_rate_se",
    "enumerative_rate", "enumerative_rate_se",
    "constraint_rate", "constraint_rate_se",
    # Belief size
    "n_named_mean", "n_named_se", "n_named_median",
    "n_constraints_mean", "n_constraints_se", "n_constraints_median",
    # Constraint accumulation over turns
    "constraint_completeness", "constraint_completeness_se", "constraint_turn_corr",
    # Size-estimate error of stated remaining count vs |Omega_t|
    "n_turns_with_count", "count_mae", "count_mae_se", "count_mse", "count_rmse",
    # Construct 2: belief vs Omega_t (Jaccard, decomposed into precision/recall)
    "n_turns_with_belief",
    "belief_jaccard", "belief_jaccard_se",
    "belief_precision", "belief_precision_se",
    "belief_recall", "belief_recall_se",
    "target_drop_rate", "target_drop_rate_se",
    # Mechanistic link
    "corr_belief_ig", "corr_belief_ig_p", "corr_n",
]


def _fmt_cell(v: Any) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def write_csv(stats: dict[str, dict[str, Any]], out_csv: Path) -> None:
    lines = [",".join(_CSV_COLS)]
    for mode in MODES:
        s = stats[mode]
        lines.append(",".join([mode] + [_fmt_cell(s[c]) for c in _CSV_COLS[1:]]))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seeker_csv(stats: dict[tuple[str, str], dict[str, Any]], out_csv: Path) -> None:
    """Per-(seeker, mode) CSV with the same metric columns as the per-mode CSV."""
    cols = ["seeker"] + _CSV_COLS  # seeker, mode, <metrics...>
    lines = [",".join(cols)]
    for seeker, mode in sorted(stats):
        s = stats[(seeker, mode)]
        lines.append(",".join([seeker, mode] + [_fmt_cell(s[c]) for c in _CSV_COLS[1:]]))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_enriched_jsonl(records: list[dict[str, Any]], out_jsonl: Path) -> None:
    """Write each record verbatim plus analyzer-recomputed per-turn metrics.

    Every original field is preserved; each turn gains ``belief_metrics`` (the
    output of ``turn_metrics`` computed from belief + omega_labels) and each record
    gains ``ig_per_turn`` and ``mode``/``seeker`` tags for convenience.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            tms, ig_per_turn = record_turns(rec)
            for turn, m in zip(rec.get("turns", []), tms):
                turn["belief_metrics"] = m
            rec["ig_per_turn"] = ig_per_turn
            rec["mode"] = infer_mode(rec.get("experiment_dir", ""))
            rec["seeker"] = infer_seeker(rec.get("experiment_dir", ""))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_figure(stats: dict[str, dict[str, Any]], out_fig: Path) -> None:
    x = range(len(MODES))
    colors = ["#bdbdbd", "#2c7fb8", "#41ab5d"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # Panel 1 -- Construct 1: does the model track a belief at all?
    ax = axes[0]
    width = 0.4
    enum = [stats[m]["enumerative_rate"] or 0 for m in MODES]
    cons = [stats[m]["constraint_rate"] or 0 for m in MODES]
    ax.bar([i - width / 2 for i in x], enum, width, label="names candidates", color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.bar([i + width / 2 for i in x], cons, width, label="states constraints", color="#fdae61", edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODES); ax.set_ylim(0, 1)
    ax.set_title("(1) Belief tracked in reasoning?\n(fraction of turns)")
    ax.set_xlabel("Observability"); ax.legend(fontsize=8)

    # Panel 2 -- Construct 2: how well does the belief match Omega_t?
    ax = axes[1]
    jac = [stats[m]["belief_jaccard"] or 0 for m in MODES]
    ax.bar(x, jac, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODES); ax.set_ylim(0, 1)
    ax.set_title("(2) Belief accuracy\n(Jaccard of $\\hat{\\Omega}_t$ vs $\\Omega_t$)")
    ax.set_xlabel("Observability")
    for i, v in zip(x, jac):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Panel 3 -- mechanistic link: corr(belief accuracy, IG) by mode.
    ax = axes[2]
    corr = [stats[m]["corr_belief_ig"] or 0 for m in MODES]
    ax.bar(x, corr, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODES)
    ax.set_title("Belief accuracy $\\to$ IG\ncorr(belief precision, IG/turn)")
    ax.set_xlabel("Observability")
    for i, v, m in zip(x, corr, MODES):
        p = stats[m]["corr_belief_ig_p"]
        label = f"{v:+.2f}" + (f"\np={p:.2f}" if isinstance(p, float) else "")
        ax.text(i, v + (0.01 if v >= 0 else -0.05), label, ha="center", fontsize=8)

    fig.suptitle(
        "Belief tracking in thinking: (1) whether the model does it, (2) how well, and its link to performance",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_fig.parent.mkdir(exist_ok=True)
    fig.savefig(out_fig, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate belief-state results by observability mode.")
    parser.add_argument("--jsonl", default="outputs/belief_states.jsonl")
    parser.add_argument("--out-csv", default="outputs/belief_states_by_mode.csv")
    parser.add_argument("--out-seeker-csv", default="outputs/belief_states_by_seeker_mode.csv")
    parser.add_argument("--out-jsonl", default="outputs/belief_states_with_metrics.jsonl",
                        help="Per-turn records enriched with analyzer-recomputed metrics.")
    parser.add_argument("--out-fig", default="figures/belief_state_scaffold.png")
    args = parser.parse_args()

    records = load_records(Path(args.jsonl))
    stats = aggregate(records)
    seeker_stats = aggregate_by_seeker(records)

    fmt = lambda v: ("  n/a" if v is None else f"{v:.2f}")  # noqa: E731
    print("C1: belief tracked?            | C2: belief vs Omega_t        | link")
    print(f"{'mode':4s} {'nConv':>6s} {'nTurn':>6s} | {'track':>6s} {'enum':>6s} {'constr':>7s} | "
          f"{'nBel':>5s} {'jaccard':>8s} {'prec':>6s} {'recall':>7s} {'tgtDrop':>8s} | {'corr(b,IG)':>11s} {'p':>6s}")
    for mode in MODES:
        s = stats[mode]
        print(f"{mode:4s} {s['n_conversations']:6d} {s['n_turns']:6d} | "
              f"{fmt(s['belief_tracking_rate']):>6s} {fmt(s['enumerative_rate']):>6s} {fmt(s['constraint_rate']):>7s} | "
              f"{s['n_turns_with_belief']:5d} {fmt(s['belief_jaccard']):>8s} {fmt(s['belief_precision']):>6s} "
              f"{fmt(s['belief_recall']):>7s} {fmt(s['target_drop_rate']):>8s} | "
              f"{fmt(s['corr_belief_ig']):>11s} {fmt(s['corr_belief_ig_p']):>6s}")

    print("\nBelief size (#named/turn over turns that name candidates):")
    print(f"{'mode':4s} {'#named_mean':>11s} {'#named_med':>10s} {'#constr_mean':>12s} {'#constr_med':>11s}")
    for mode in MODES:
        s = stats[mode]
        print(f"{mode:4s} {fmt(s['n_named_mean']):>11s} {fmt(s['n_named_median']):>10s} "
              f"{fmt(s['n_constraints_mean']):>12s} {fmt(s['n_constraints_median']):>11s}")

    write_csv(stats, Path(args.out_csv))
    write_seeker_csv(seeker_stats, Path(args.out_seeker_csv))
    write_enriched_jsonl(records, Path(args.out_jsonl))
    make_figure(stats, Path(args.out_fig))
    print(f"\nWrote {args.out_csv}, {args.out_seeker_csv}, {args.out_jsonl} and {args.out_fig}")


if __name__ == "__main__":
    main()
