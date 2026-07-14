#!/usr/bin/env python3
"""Dual-axis plot: |Ω_t| (left) and IG_t per turn (right) over turns.

Reads turns.jsonl files directly and aggregates across conversations.
Mean ± SE band shown for each metric.

Usage:
    # single aggregated curve
    python scripts/analysis/plot_omega_ig_dual_axis.py

    # one line per model (canonical 10 seekers, CoT split)
    python scripts/analysis/plot_omega_ig_dual_axis.py --per-model --canonical \\
        --out outputs/plots/omega_ig_canonical.pdf

    # one line per model, custom seeker filter
    python scripts/analysis/plot_omega_ig_dual_axis.py --per-model \\
        --seekers "Qwen3-8B,Qwen3-4B" --out outputs/plots/omega_ig_qwen.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# ── canonical model list (from paper Table) ──────────────────────────────────

# Slug substrings that identify each canonical seeker (order = legend order)
CANONICAL_SLUGS: list[str] = [
    "Llama-3.1-8B-Instruct",
    "paprika_Meta-Llama-3.1-8B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-4B-Thinking-2507",
    "Qwen3-8B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-30B-A3B-Thinking-2507",
    "google-gemma-4-E4B-it",
    "google-gemma-4-31B-it",
    "Nemotron-Cascade-8B",
]

# Display-name substrings that define the desired panel order (matched with 'in')
CANONICAL_DISPLAY_ORDER: list[str] = [
    "Qwen3-30B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Nemotron",
    "Gemma-4-31B",
    "Gemma-4-E4B",
    "paprika",
    "Llama",
]

# These slugs are Thinking-branded models — excluded from no-CoT canonical plots
THINKING_SLUGS: set[str] = {
    "Qwen3-4B-Thinking-2507",
    "Qwen3-30B-A3B-Thinking-2507",
    "Nemotron-Cascade-8B-Thinking",
}

# Clean display names keyed by (slug, is_cot)
_DN: dict[tuple[str, bool], str] = {
    ("Llama-3.1-8B-Instruct",              False): "Llama-3.1-8B-Instruct",
    ("paprika_Meta-Llama-3.1-8B-Instruct", False): "paprika-Llama-3.1-8B",
    ("paprika_Meta-Llama-3.1-8B-Instruct", True):  "paprika-Llama-3.1-8B",
    # Instruct/Thinking slugs: always same display name regardless of exp cot flag
    ("Qwen3-4B-Instruct-2507",             False): "Qwen3-4B-Instruct",
    ("Qwen3-4B-Instruct-2507",             True):  "Qwen3-4B-Instruct",
    ("Qwen3-4B-Thinking-2507",             False): "Qwen3-4B-Thinking",
    ("Qwen3-4B-Thinking-2507",             True):  "Qwen3-4B-Thinking",
    ("Qwen3-8B",                           False): "Qwen3-8B",
    ("Qwen3-8B",                           True):  "Qwen3-8B (CoT)",
    ("Qwen3-30B-A3B-Instruct-2507",        False): "Qwen3-30B-A3B-Instruct",
    ("Qwen3-30B-A3B-Instruct-2507",        True):  "Qwen3-30B-A3B-Instruct",
    ("Qwen3-30B-A3B-Thinking-2507",        False): "Qwen3-30B-A3B-Thinking",
    ("Qwen3-30B-A3B-Thinking-2507",        True):  "Qwen3-30B-A3B-Thinking",
    ("google-gemma-4-E4B-it",              False): "Gemma-4-E4B-IT",
    ("google-gemma-4-E4B-it",             True):  "Gemma-4-E4B-IT (CoT)",
    ("google-gemma-4-31B-it",             False): "Gemma-4-31B-IT",
    ("google-gemma-4-31B-it",             True):  "Gemma-4-31B-IT (CoT)",
    ("Nemotron-Cascade-8B",               False): "Nemotron-Cascade-8B",
    ("Nemotron-Cascade-8B",               True):  "Nemotron-Cascade-8B (CoT)",
    ("Nemotron-Cascade-8B-Thinking",      False): "Nemotron-Cascade-8B (CoT)",
    ("Nemotron-Cascade-8B-Thinking",      True):  "Nemotron-Cascade-8B (CoT)",
}


def _display_name(slug: str, is_cot: bool) -> str:
    if (slug, is_cot) in _DN:
        return _DN[(slug, is_cot)]
    return f"{slug} (CoT)" if is_cot else slug


# ── helpers ───────────────────────────────────────────────────────────────────

def load_turns(path: Path) -> list[dict]:
    turns = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return sorted(turns, key=lambda t: t["turn_index"])


def _is_cot_exp(exp_dir: Path) -> bool:
    name = exp_dir.name
    return "_cot" in name and "_no_cot" not in name


def _match(exp_dir: Path, domain: str | None, obs: str | None,
           cot: bool | None) -> bool:
    name = exp_dir.name.lower()
    if domain and domain.lower() not in name:
        return False
    if obs:
        tag = "_fo_" if obs in ("fo", "io", "fully") else "_po_"
        if tag not in name:
            return False
    if cot is True and not (_is_cot_exp(exp_dir)):
        return False
    if cot is False and _is_cot_exp(exp_dir):
        return False
    return True


def _seeker_slug(model_dir: Path) -> str:
    return model_dir.name.split("__o_")[0].removeprefix("s_")


def _ingest_conversation(
    turns: list[dict],
    per_turn: dict[int, list[tuple[float, float]]],
    max_turn: int,
    zero_pad: bool,
) -> None:
    # turn 0: initial candidate set size before any question
    first = turns[0]
    omega_before = first.get("active_candidates_before")
    if omega_before is not None:
        per_turn[0].append((float(omega_before), 0.0))

    last_omega = None
    for t in turns:
        idx = int(t["turn_index"])
        if idx > max_turn:
            continue
        omega = t.get("active_candidates_after")
        ig = t.get("info_gain")
        if omega is not None and ig is not None:
            per_turn[idx].append((float(omega), float(ig)))
            last_omega = float(omega)
    if zero_pad and last_omega is not None:
        last_idx = max(int(t["turn_index"]) for t in turns)
        for pad in range(last_idx + 1, max_turn + 1):
            per_turn[pad].append((last_omega, 0.0))


def collect_turns_jsonl(
    outputs_root: Path,
    seeker_filters: list[str] | None,   # substrings matching seeker slug
    domain: str | None,
    obs: str | None,
    cot: bool | None,
    max_turn: int = 30,
    zero_pad: bool = True,
    per_model: bool = False,
) -> tuple[dict, int]:
    """Return (data, n_conversations).

    per_model=False → data = dict[turn_idx → [(omega, ig)]]
    per_model=True  → data = dict[label → dict[turn_idx → [(omega, ig)]]]
                      label = display_name(slug, is_cot)
    """
    n_conversations = 0
    data: dict = defaultdict(lambda: defaultdict(list)) if per_model else defaultdict(list)

    models_root = (outputs_root / "models"
                   if (outputs_root / "models").exists() else outputs_root)
    model_dirs = list(models_root.glob("s_*__o_*__p_*"))

    if seeker_filters:
        fl = [f.lower() for f in seeker_filters]
        model_dirs = [d for d in model_dirs
                      if any(f in _seeker_slug(d).lower() for f in fl)]

    for model_dir in model_dirs:
        slug = _seeker_slug(model_dir)
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            if not _match(exp_dir, domain, obs, cot):
                continue
            conv_root = exp_dir / "conversations"
            if not conv_root.exists():
                continue
            is_cot = _is_cot_exp(exp_dir)
            label = _display_name(slug, is_cot)

            for conv_dir in conv_root.iterdir():
                turns_file = conv_dir / "turns.jsonl"
                if not turns_file.exists():
                    continue
                try:
                    turns = load_turns(turns_file)
                except Exception:
                    continue
                if not turns:
                    continue
                n_conversations += 1
                target = data[label] if per_model else data
                _ingest_conversation(turns, target, max_turn, zero_pad)

    return data, n_conversations


def _aggregate(per_turn: dict[int, list[tuple[float, float]]]):
    turns_sorted = sorted(per_turn)
    m_omega, se_omega, m_ig, se_ig = [], [], [], []
    for t in turns_sorted:
        vals = per_turn[t]
        omegas = np.array([v[0] for v in vals])
        igs    = np.array([v[1] for v in vals])
        n = len(vals)
        m_omega.append(omegas.mean())
        se_omega.append(omegas.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        m_ig.append(igs.mean())
        se_ig.append(igs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
    return (np.array(turns_sorted),
            np.array(m_omega), np.array(se_omega),
            np.array(m_ig),    np.array(se_ig))


# ── family colors ─────────────────────────────────────────────────────────────

FAMILY_COLOR: dict[str, str] = {
    "llama":     "#F97316",   # vivid orange
    "qwen":      "#2563EB",   # vivid blue
    "gemma":     "#059669",   # vivid green
    "nemotron":  "#7C3AED",   # vivid purple
    "other":     "#374151",   # dark grey fallback
}

PANEL_BG = "#EBEBEB"


def _family(label: str) -> str:
    l = label.lower()
    if "llama" in l:   return "llama"
    if "qwen"  in l:   return "qwen"
    if "gemma" in l:   return "gemma"
    if "nemotron" in l: return "nemotron"
    return "other"


def _family_color(label: str) -> str:
    return FAMILY_COLOR[_family(label)]


# ── color palette (used only by plot_per_model) ───────────────────────────────

_TAB20 = plt.colormaps["tab20"]

def _palette(labels: list[str]) -> dict[str, tuple]:
    base_labels = [l.replace(" (CoT)", "") for l in labels]
    unique_bases = list(dict.fromkeys(base_labels))
    colors = {}
    for i, base in enumerate(unique_bases):
        colors[base]            = _TAB20(2 * (i % 10))
        colors[f"{base} (CoT)"] = _TAB20(2 * (i % 10) + 1)
    return {l: colors.get(l, _TAB20(i / max(len(labels) - 1, 1)))
            for i, l in enumerate(labels)}


# ── plotting ──────────────────────────────────────────────────────────────────

COLOR_OMEGA = "#6B7280"  # dark gray — distinct from all family colors
COLOR_IG    = "#DC2626"


def plot_facet(
    data: dict[str, dict],
    out: Path,
    title: str = "",
    zero_pad: bool = True,
    label_order: list[str] | None = None,
    ncols: int = 4,
    ylim_omega: float | None = None,
    ylim_ig: float | None = None,
) -> None:
    """Grid of small dual-axis panels — one per model, original blue-line + red-bar style."""
    labels = label_order if label_order else sorted(data.keys())
    labels = [l for l in labels if l in data]
    n = len(labels)
    nrows = (n + ncols - 1) // ncols

    # shared y limits — compute across all models first, then allow override
    all_omega, all_ig = [], []
    agg_cache: dict[str, tuple] = {}
    for label in labels:
        t, mo, se_mo, mi, se_mi = _aggregate(data[label])
        agg_cache[label] = (t, mo, se_mo, mi, se_mi)
        all_omega.extend(mo)
        all_ig.extend(mi)
    ylim_omega = (0, ylim_omega if ylim_omega else max(all_omega) * 1.05)
    ylim_ig    = (0, ylim_ig    if ylim_ig    else max(all_ig)    * 1.15)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.0),
                             squeeze=False)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax1 = axes[row][col]
        ax2 = ax1.twinx()

        t_arr, m_omega, se_omega, m_ig, se_ig = agg_cache[label]
        c_omega = COLOR_OMEGA            # fixed blue   → |Ω_t| line + left axis
        c_ig    = _family_color(label)   # family color → IG_t bars + right axis

        ax2.set_facecolor(PANEL_BG)   # background on ax2 (behind bars)
        ax1.set_facecolor("none")     # ax1 transparent so ax2 background shows
        ax1.set_zorder(ax2.get_zorder() + 1)  # bring line in front of bars

        ax1.plot(t_arr, m_omega, color=c_omega, lw=1.8, marker="o", ms=2.5)
        ax1.set_ylim(*ylim_omega)
        ax1.tick_params(axis="y", labelcolor=c_omega, labelsize=7)
        ax1.set_ylabel(r"$|\Omega_t|$", color=c_omega, fontsize=8)

        ax2.bar(t_arr, m_ig, color=c_ig, alpha=0.85, width=0.7)
        ax2.set_ylim(*ylim_ig)
        ax2.tick_params(axis="y", labelcolor=c_ig, labelsize=7)
        ax2.set_ylabel(r"$\mathrm{IG}_t$", color=c_ig, fontsize=8)

        ax1.set_title(label, fontsize=8, pad=3, color="black")
        ax1.tick_params(axis="x", labelsize=7)
        ax1.set_xlabel("Turn", fontsize=7)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
        ax1.grid(axis="x", ls="--", lw=0.4, alpha=0.5, color="white")
        ax1.grid(axis="y", ls=":",  lw=0.4, alpha=0.5, color="white")

    # hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out)


def plot_per_model(
    data: dict[str, dict],
    out: Path,
    title: str = "",
    zero_pad: bool = True,
    label_order: list[str] | None = None,
) -> None:
    """Dual-axis: solid lines = |Ω_t| (left), dashed = IG_t (right), one color per model."""
    labels = label_order if label_order else sorted(data.keys())
    labels = [l for l in labels if l in data]   # drop missing
    colors = _palette(labels)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    for label in labels:
        per_turn = data[label]
        if not per_turn:
            continue
        t_arr, m_omega, _, m_ig, _ = _aggregate(per_turn)
        c = colors[label]
        ax1.plot(t_arr, m_omega, color=c, lw=1.6, ls="-",  label=label)
        ax2.plot(t_arr, m_ig,    color=c, lw=1.6, ls="--")

    ax1.set_xlabel("Turn", fontsize=12)
    ax1.set_ylabel(r"$|\Omega_t|$  (active candidates)", fontsize=12)
    ax2.set_ylabel(r"$\mathrm{IG}_t$  (bits / turn)",   fontsize=12)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(axis="x", ls="--", lw=0.5, alpha=0.4)
    ax1.grid(axis="y", ls=":", lw=0.5, alpha=0.3)

    # line-style legend (inset, upper right)
    style_handles = [
        Line2D([0], [0], color="grey", lw=1.6, ls="-",  label=r"$|\Omega_t|$ (left)"),
        Line2D([0], [0], color="grey", lw=1.6, ls="--", label=r"$\mathrm{IG}_t$ (right)"),
    ]
    ax2.legend(handles=style_handles, loc="upper right", fontsize=9,
               framealpha=0.9, title="Line style")

    # model color legend (outside right)
    model_handles = [Line2D([0], [0], color=colors[l], lw=2, label=l) for l in labels]
    fig.legend(model_handles, labels,
               loc="upper left", bbox_to_anchor=(1.02, 1.0),
               fontsize=8, framealpha=0.9, title="Seeker model",
               borderaxespad=0)

    pad_note = " · zero-padded" if zero_pad else ""
    fig.suptitle(title or f"Per-model trajectories{pad_note}", fontsize=12)
    fig.tight_layout()
    _save(fig, out)


def plot_single_conversation(turns: list[dict], out: Path, title: str = "") -> None:
    t_arr  = np.array([t["turn_index"] for t in turns])
    omega  = np.array([t["active_candidates_after"] for t in turns])
    ig     = np.array([t["info_gain"] for t in turns])
    fig, ax1 = plt.subplots(figsize=(7, 4))
    _draw(ax1, t_arr, omega, None, t_arr, ig, None, title)
    fig.tight_layout()
    _save(fig, out)


def plot_aggregated(
    per_turn: dict,
    out: Path,
    title: str = "",
    n_conversations: int = 0,
    zero_pad: bool = True,
) -> None:
    t_arr, m_omega, se_omega, m_ig, se_ig = _aggregate(per_turn)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    _draw(ax1, t_arr, m_omega, se_omega, t_arr, m_ig, se_ig,
          title, n_convs=n_conversations, zero_pad=zero_pad)
    fig.tight_layout()
    _save(fig, out)


def _draw(ax1, turns_omega, mean_omega, se_omega,
          turns_ig, mean_ig, se_ig,
          title="", n_convs=0, zero_pad=True) -> None:
    ax2 = ax1.twinx()

    ax1.plot(turns_omega, mean_omega, color=COLOR_OMEGA, lw=2.0,
             marker="o", ms=4, label=r"$|\Omega_t|$")
    if se_omega is not None:
        ax1.fill_between(turns_omega, mean_omega - se_omega, mean_omega + se_omega,
                         color=COLOR_OMEGA, alpha=0.15)
    ax1.set_xlabel("Turn", fontsize=12)
    ax1.set_ylabel(r"$|\Omega_t|$  (active candidates)", color=COLOR_OMEGA, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLOR_OMEGA)
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax2.bar(turns_ig, mean_ig, color=COLOR_IG, alpha=0.35, width=0.6,
            label=r"$\mathrm{IG}_t$")
    if se_ig is not None:
        ax2.errorbar(turns_ig, mean_ig, yerr=se_ig,
                     fmt="none", ecolor=COLOR_IG, elinewidth=1.2, capsize=3)
    ax2.set_ylabel(r"$\mathrm{IG}_t$  (bits / turn)", color=COLOR_IG, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLOR_IG)
    ax2.set_ylim(bottom=0)

    ax1.set_xlim(turns_omega[0] - 0.5, turns_omega[-1] + 0.5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(axis="x", ls="--", lw=0.5, alpha=0.4)
    ax1.grid(axis="y", ls=":", lw=0.5, alpha=0.3)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10, framealpha=0.8)

    if title:
        ax1.set_title(title, fontsize=11, pad=6)
    elif n_convs:
        pad_note = ", zero-padded" if zero_pad else ""
        ax1.set_title(
            f"Candidate set size and IG per turn  (n = {n_convs:,} conversations{pad_note})",
            fontsize=10, pad=6)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument("--domain", choices=["geo", "objects", "diseases"])
    p.add_argument("--obs", choices=["fo", "io", "po"])
    group = p.add_mutually_exclusive_group()
    group.add_argument("--cot", action="store_true", default=None)
    group.add_argument("--no-cot", dest="cot", action="store_false")
    p.add_argument("--conversation", metavar="TURNS_JSONL")
    p.add_argument("--max-turn", type=int, default=30)
    p.add_argument("--per-model", action="store_true",
                   help="One line per (model, CoT) pair (overlapping)")
    p.add_argument("--facet", action="store_true",
                   help="Grid of panels, one per model, original dual-axis style")
    p.add_argument("--ncols", type=int, default=4,
                   help="Columns in facet grid (default: 4)")
    p.add_argument("--ylim-omega", type=float, default=None,
                   help="Fixed upper y-limit for |Ω_t| (default: auto)")
    p.add_argument("--ylim-ig", type=float, default=None,
                   help="Fixed upper y-limit for IG_t (default: auto)")
    p.add_argument("--canonical", action="store_true",
                   help="Filter to canonical 10 seekers (from paper table)")
    p.add_argument("--seekers", metavar="A,B,...",
                   help="Comma-separated seeker slug substrings (with --per-model)")
    p.add_argument("--no-zero-pad", dest="zero_pad", action="store_false", default=True)
    p.add_argument("--exclude-families", metavar="A,B,...",
                   help="Comma-separated families to exclude (e.g. 'llama')")
    p.add_argument("--append-llama-no-cot", action="store_true",
                   help="Append Llama models (no-CoT data) at the end of the plot")
    p.add_argument("--title", default="")
    p.add_argument("--out", default="outputs/plots/omega_ig_dual_axis.pdf")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)

    if args.conversation:
        turns = load_turns(Path(args.conversation))
        turns = [t for t in turns if t["turn_index"] <= args.max_turn]
        plot_single_conversation(turns, out, title=args.title)
        return

    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        print(f"Error: '{outputs_root}' not found.", file=sys.stderr)
        sys.exit(1)

    # build seeker filter list
    seeker_filters: list[str] | None = None
    if args.canonical:
        seeker_filters = CANONICAL_SLUGS
    elif args.seekers:
        seeker_filters = [s.strip() for s in args.seekers.split(",")]

    print("Scanning conversations…", flush=True)
    data, n_convs = collect_turns_jsonl(
        outputs_root,
        seeker_filters=seeker_filters,
        domain=args.domain,
        obs=args.obs,
        cot=args.cot,
        max_turn=args.max_turn,
        zero_pad=args.zero_pad,
        per_model=(args.per_model or args.facet),
    )

    if not data:
        print("No matching turns.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    pad_note = " (zero-padded)" if args.zero_pad else ""
    print(f"Found {n_convs:,} conversations{pad_note}.")

    if args.per_model or args.facet:
        # canonical legend order: same as CANONICAL_SLUGS, CoT after no-CoT
        if args.canonical:
            ordered: list[str] = []
            for slug in CANONICAL_SLUGS:
                ordered.append(_display_name(slug, False))
                ordered.append(_display_name(slug, True))
            label_order = list(dict.fromkeys(ordered))
        else:
            label_order = sorted(data.keys())

        present = [l for l in label_order if l in data]

        # exclude Thinking-branded models from no-CoT canonical plots
        if args.canonical and args.cot is False:
            thinking_display = {_display_name(s, False) for s in THINKING_SLUGS} | \
                               {_display_name(s, True)  for s in THINKING_SLUGS}
            present = [l for l in present if l not in thinking_display]

        # exclude families (e.g. --exclude-families llama)
        if args.exclude_families:
            excl = [f.strip().lower() for f in args.exclude_families.split(",")]
            present = [l for l in present if _family(l) not in excl]

        # append Llama no-CoT data at the end (used in CoT plot for baseline)
        if args.append_llama_no_cot:
            llama_slugs = [s for s in CANONICAL_SLUGS if "llama" in s.lower()]
            llama_data, _ = collect_turns_jsonl(
                outputs_root,
                seeker_filters=llama_slugs,
                domain=args.domain,
                obs=args.obs,
                cot=False,          # always no-CoT for Llama
                max_turn=args.max_turn,
                zero_pad=args.zero_pad,
                per_model=True,
            )
            # add in canonical slug order
            for slug in llama_slugs:
                label = _display_name(slug, False)
                if label in llama_data:
                    data[label] = llama_data[label]
                    if label not in present:
                        present.append(label)

        # apply canonical display order when --canonical is set
        if args.canonical:
            def _order_key(label: str) -> int:
                for i, pattern in enumerate(CANONICAL_DISPLAY_ORDER):
                    if pattern.lower() in label.lower():
                        return i
                return len(CANONICAL_DISPLAY_ORDER)
            present.sort(key=_order_key)

        print(f"Plotting {len(present)} panels: {', '.join(present)}")

        if args.facet:
            plot_facet(data, out, title=args.title,
                       zero_pad=args.zero_pad, label_order=present,
                       ncols=args.ncols,
                       ylim_omega=args.ylim_omega,
                       ylim_ig=args.ylim_ig)
        else:
            plot_per_model(data, out, title=args.title,
                           zero_pad=args.zero_pad, label_order=present)
    else:
        plot_aggregated(data, out, title=args.title,
                        n_conversations=n_convs, zero_pad=args.zero_pad)


if __name__ == "__main__":
    main()
