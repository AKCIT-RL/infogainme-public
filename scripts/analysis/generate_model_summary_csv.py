"""Gera model_summary.csv — métricas agrupadas por modelo e observabilidade.

Lê unified_experiments.csv e seeker_traces.json para calcular:
  - Win Rate (mean ± std)
  - Mean IG/Turn (mean ± std)
  - Mean IG (mean ± std)
  - Mean Turns (mean ± std)
  - Avg Q/Turn (mean ± std) — apenas experimentos CoT

Nota: Avg Q/Turn é derivado dos seeker_traces.json (questions_considered por turno).
      Experimentos sem _cot no nome são ignorados para essa métrica pois não têm
      raciocínio explícito (modelos Instruct).

Usage:
    python scripts/generate_model_summary_csv.py
    python scripts/generate_model_summary_csv.py [outputs_dir] [output_csv]
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load_unified_csv(unified_csv: Path) -> list[dict]:
    with unified_csv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_cot(exp_name: str) -> bool:
    """Retorna True apenas para experimentos CoT (exclui _no_cot)."""
    return exp_name.endswith("_cot") or "_cot_" in exp_name


def _compute_avg_questions_per_turn(outputs_dir: Path) -> dict[tuple[str, str], list[float]]:
    """Retorna avg questions/turn por conversa, indexado por (model, observability).

    Considera apenas experimentos CoT (excluindo _no_cot).
    """
    qturn: dict[tuple[str, str], list[float]] = defaultdict(list)
    for tf in (outputs_dir / "models").glob("**/conversations/*/seeker_traces.json"):
        exp_name = tf.parts[-4] if len(tf.parts) >= 4 else ""
        if not _is_cot(exp_name):
            continue
        try:
            data = json.loads(tf.read_text(encoding="utf-8"))
            history = data.get("history", [])
            if not history:
                continue
            obs = data.get("observability_mode", "")
            model = data.get("config", {}).get("model", "")
            if not model or not obs:
                continue
            q_counts = [
                len(t.get("reasoning_trace", {}).get("questions_considered", []))
                for t in history
            ]
            if q_counts:
                qturn[(model, obs)].append(statistics.mean(q_counts))
        except Exception:
            pass
    return qturn


def _fv(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def _round(v: float, n: int = 4) -> float | str:
    return "" if math.isnan(v) else round(v, n)


def main() -> int:
    repo_root = Path(__file__).parent.parent.parent
    outputs_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else repo_root / "outputs"
    output_csv = Path(sys.argv[2]) if len(sys.argv) >= 3 else outputs_dir / "model_summary.csv"

    unified_csv = outputs_dir / "unified_experiments.csv"
    if not unified_csv.exists():
        print(f"❌ unified_experiments.csv não encontrado em: {outputs_dir}")
        print("   Rode primeiro: python scripts/generate_unified_csv.py")
        return 1

    rows = _load_unified_csv(unified_csv)

    # Agrupar por (model, observability) — apenas experimentos CoT
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if _is_cot(r.get("Experimento", "")):
            groups[(r["Seeker Model"], r["Observabilidade"])].append(r)

    # Avg Q/Turn dos traces (só CoT)
    print(f"🔎 Calculando Avg Q/Turn a partir dos traces em: {outputs_dir}")
    qturn = _compute_avg_questions_per_turn(outputs_dir)

    headers = [
        "Model", "Obs", "N",
        "Win Rate Mean", "Win Rate Std",
        "Mean IG/Turn Mean", "Mean IG/Turn Std",
        "Mean IG Mean", "Mean IG Std",
        "Mean Turns Mean", "Mean Turns Std",
        "Avg Q/Turn Mean", "Avg Q/Turn Std",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for (model, obs), vals in sorted(groups.items()):
            obs_short = "FO" if "FULLY" in obs else "PO"

            def fv_col(key: str) -> tuple[float, float]:
                v = [float(r[key]) for r in vals if r.get(key)]
                return _fv(v)

            wr_m, wr_s   = fv_col("Win Rate")
            igt_m, igt_s = fv_col("Mean Info Gain/Turn")
            ig_m, ig_s   = fv_col("Mean Info Gain")
            t_m, t_s     = fv_col("Mean Turns")

            qt_vals = qturn.get((model, obs), [])
            qt_m, qt_s = _fv(qt_vals)

            writer.writerow({
                "Model": model,
                "Obs": obs_short,
                "N": len(vals),
                "Win Rate Mean":      _round(wr_m),
                "Win Rate Std":       _round(wr_s),
                "Mean IG/Turn Mean":  _round(igt_m),
                "Mean IG/Turn Std":   _round(igt_s),
                "Mean IG Mean":       _round(ig_m),
                "Mean IG Std":        _round(ig_s),
                "Mean Turns Mean":    _round(t_m, 2),
                "Mean Turns Std":     _round(t_s, 2),
                "Avg Q/Turn Mean":    _round(qt_m),
                "Avg Q/Turn Std":     _round(qt_s),
            })

    print(f"✅ model_summary.csv salvo em: {output_csv}")
    print(f"📦 Grupos: {len(groups)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
