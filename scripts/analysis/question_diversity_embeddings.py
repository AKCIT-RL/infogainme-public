#!/usr/bin/env python3
"""Compute per-game question diversity via sentence embeddings.

Diversity = mean pairwise cosine distance across all questions in a game:
    diversity = 1/(C(n,2)) * sum_{i<j} (1 - cos(e_i, e_j))

Games with fewer than 2 questions receive NaN.

Output: outputs/views_artigo/question_diversity_by_game.csv
Columns: seeker, target, mode, domain, n_questions, diversity_mean_cosine

Usage:
    uv run python3 scripts/analysis/question_diversity_embeddings.py
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────

# Thinking-branded slugs are always CoT — skip their no-CoT experiments
ALWAYS_COT_SLUGS: set[str] = {
    "Qwen3-4B-Thinking-2507",
    "Qwen3-30B-A3B-Thinking-2507",
}

OUTPUTS_ROOT       = Path("outputs/models")
CANONICAL_ORACLE   = "o_Qwen3-8B__p_Qwen3-8B"   # only standard oracle/pruner
OUT_FILE           = Path("outputs/views_artigo/question_diversity_by_game.csv")
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"
BATCH_SIZE    = 512

# ── canonical seekers (from configs/full, excluding ablations) ────────────────

def _canonical_seekers() -> set[str]:
    """Return canonical seeker slugs normalized to match directory names.

    YAMLs may use 'google/gemma-4-31B-it' (HF path) while directories use
    'google-gemma-4-31B-it' (slash → hyphen). We normalize to the directory
    form. Nemotron-Cascade-8B-Thinking lives in its own directory but has no
    dedicated config; we include it explicitly as a CoT variant.
    Qwen3-0.6B is excluded (not in the paper's canonical model table).
    """
    EXCLUDE = {"Qwen3-0.6B"}
    models: set[str] = set()
    for yf in Path("configs/full").rglob("*.yaml"):
        if "ablation" in str(yf):
            continue
        cfg = yaml.safe_load(yf.read_text())
        seeker = cfg.get("models", {}).get("seeker", {}).get("model")
        if seeker:
            slug = seeker.replace("/", "-")   # normalize slash → hyphen
            if slug not in EXCLUDE:
                models.add(slug)
    # Nemotron-Cascade-8B-Thinking is a stale directory (served-model-name artifact,
    # _ont data only) — do NOT include; CoT data lives in Nemotron-Cascade-8B _cot exps.
    return models


# ── sample targets (16 per domain) ───────────────────────────────────────────

SAMPLE_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]


def _sample_targets(targets: list[str]) -> set[str]:
    s = sorted(targets)
    return {s[i] for i in SAMPLE_INDICES if i < len(s)}


N_RUNS = 2   # number of runs per target to collect (standardized across all models)


def _build_all_samples() -> set[str]:
    """Return set of conversation dir names (target_runNN) for all runs of sampled targets."""
    geo = pd.read_csv("data/geo/top_160_pop_cities.csv")
    geo_bases = (
        geo["city_id"].astype(str).map(lambda x: f"city-{x}")
    ).tolist()

    dis = pd.read_csv("data/diseases/diseases_160.csv").reset_index(drop=True)
    dis_bases = dis.apply(
        lambda r: f"disease-{r['disease'].replace(' ', '_')}-{r.name}",
        axis=1,
    ).tolist()

    obj = pd.read_csv("data/objects/objects_full.csv")
    obj["cat_idx"] = obj.groupby("category").cumcount()
    obj_bases = obj.apply(
        lambda r: f"object-{r['category'].lower().replace(' ', '_')}-{r['cat_idx']}",
        axis=1,
    ).tolist()

    sampled_bases = (
        _sample_targets(geo_bases)
        | _sample_targets(dis_bases)
        | _sample_targets(obj_bases)
    )
    # expand to all runs
    return {
        f"{base}_run{run:02d}"
        for base in sampled_bases
        for run in range(1, N_RUNS + 1)
    }


def _target_base(target: str) -> str:
    """Strip _runNN suffix: 'city-1234_run02' → 'city-1234'."""
    return re.sub(r"_run\d+$", "", target)


# ── helpers ───────────────────────────────────────────────────────────────────

_MODE_MAP = {
    "FULLY_OBSERVABLE":    "FO",
    "INITIALLY_OBSERVABLE": "IO",
    "PARTIALLY_OBSERVABLE": "PO",
}

_MODE_RE = re.compile(r"_(fo|io|po)_", re.IGNORECASE)


def _mode_from_exp_name(name: str) -> str:
    m = _MODE_RE.search(name)
    return m.group(1).upper() if m else "?"


# Suffixes that always mark non-canonical variants
_NON_CANONICAL_RE = re.compile(
    r"(_ont|_with_prior|_ablation)($|_)", re.IGNORECASE
)


def _is_canonical_exp(name: str, siblings: set[str] | None = None) -> bool:
    if _NON_CANONICAL_RE.search(name):
        return False
    # When a _with_kickoff version exists, it is canonical — skip the plain version.
    if "_with_kickoff" not in name:
        if siblings and (name + "_with_kickoff") in siblings:
            return False   # a _with_kickoff variant exists — defer to it
    return True


def _is_cot_exp(name: str) -> bool:
    return "_cot" in name and "_no_cot" not in name


def _domain_from_target(target: str) -> str:
    if target.startswith("city-"):     return "geo"
    if target.startswith("disease-"):  return "diseases"
    if target.startswith("object-"):   return "objects"
    return "?"


def _load_questions(turns_file: Path) -> list[str]:
    questions = []
    with turns_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            q = t.get("question", {})
            text = q.get("text") if isinstance(q, dict) else None
            if text:
                questions.append(text.strip())
    return questions


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    canonical_seekers = set(_canonical_seekers())
    all_samples       = _build_all_samples()
    print(f"Canonical seekers : {len(canonical_seekers)}")
    print(f"Sample targets    : {len(all_samples)}")

    # ── collect (seeker, target, mode, questions) tuples ──────────────────────
    records: list[dict] = []

    for model_dir in sorted(OUTPUTS_ROOT.glob(f"s_*__{CANONICAL_ORACLE}")):
        seeker_slug = model_dir.name.split("__o_")[0].removeprefix("s_")
        if seeker_slug not in canonical_seekers:
            continue

        siblings = {d.name for d in model_dir.iterdir() if d.is_dir()}
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if not _is_canonical_exp(exp_dir.name, siblings):
                continue
            mode   = _mode_from_exp_name(exp_dir.name)
            is_cot = _is_cot_exp(exp_dir.name)

            # Thinking models only contribute CoT experiments
            if seeker_slug in ALWAYS_COT_SLUGS and not is_cot:
                continue

            conv_root = exp_dir / "conversations"
            if not conv_root.exists():
                continue

            for conv_dir in conv_root.iterdir():
                target = conv_dir.name          # e.g. city-1234_run01
                if target not in all_samples:
                    continue
                tf = conv_dir / "turns.jsonl"
                if not tf.exists():
                    continue
                questions = _load_questions(tf)
                records.append({
                    "seeker":    seeker_slug,
                    "target":    target,
                    "mode":      mode,
                    "cot":       is_cot,
                    "domain":    _domain_from_target(target),
                    "path":      str(tf),
                    "questions": questions,
                })

    print(f"Game records found: {len(records)}")
    if not records:
        print("Nothing to process.")
        return

    # ── embed all unique questions in one batch ────────────────────────────────
    print(f"Loading embedding model '{EMBED_MODEL}'…")
    model = SentenceTransformer(EMBED_MODEL)

    all_texts: list[str] = []
    for r in records:
        all_texts.extend(r["questions"])
    unique_texts = list(dict.fromkeys(all_texts))   # deduplicate, preserve order
    print(f"Encoding {len(unique_texts):,} unique questions…")

    embeddings_matrix = model.encode(
        unique_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit norm → cos sim = dot product
    )
    text2emb = {t: embeddings_matrix[i] for i, t in enumerate(unique_texts)}

    # ── compute diversity per game ─────────────────────────────────────────────
    rows = []
    for r in records:
        qs = r["questions"]
        n  = len(qs)
        if n < 2:
            div = float("nan")
        else:
            embs = np.stack([text2emb[q] for q in qs])
            sim   = embs @ embs.T
            i_idx, j_idx = np.triu_indices(n, k=1)
            div = float(np.mean(1.0 - sim[i_idx, j_idx]))

        rows.append({
            "seeker":           r["seeker"],
            "target_base":      _target_base(r["target"]),
            "run":              r["target"],
            "mode":             r["mode"],
            "cot":              r["cot"],
            "domain":           r["domain"],
            "path":             r["path"],
            "n_questions":      n,
            "diversity_cosine": div,
        })

    df_runs = pd.DataFrame(rows)

    # ── aggregate: mean ± SE across runs per (seeker, target_base, mode, cot) ──
    agg = (
        df_runs
        .groupby(["seeker", "target_base", "mode", "cot", "domain"])
        .agg(
            n_runs        =("diversity_cosine", "count"),
            diversity_mean=("diversity_cosine", "mean"),
            diversity_se  =("diversity_cosine", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")),
            paths         =("path", lambda x: "|".join(sorted(x))),
        )
        .reset_index()
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(agg):,} rows → {OUT_FILE}")
    summary = (
        agg.groupby(["seeker", "cot"])
        .agg(diversity_mean=("diversity_mean", "mean"),
             diversity_se=("diversity_se", "mean"))
        .sort_values("diversity_mean", ascending=False)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
