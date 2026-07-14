"""Gera um CSV unificado com métricas globais de todos os experimentos.

Usage:
    # Varre ./outputs e salva em ./outputs/unified_experiments.csv
    python scripts/analysis/generate_unified_csv.py

    # Filtrar apenas run_index=1 (lê summary_run01.json, salva unified_experiments_run01.csv)
    python scripts/analysis/generate_unified_csv.py --only-run 1

    # Informar diretório base de outputs e caminho de saída
    python scripts/analysis/generate_unified_csv.py --base-dir path/to/outputs --output path/to/out.csv

Colunas geradas:
    Experimento, Dataset, Seeker Model, Oracle Model, Pruner Model, Observabilidade, Total Runs, Win Rate,
    Mean Turns, Mean Info Gain/Turn, Mean Info Gain, Mean Initial Entropy,
    Mean Seeker Tokens, Mean Seeker Reasoning Tokens, Mean Seeker Final Tokens,
    SE Win Rate, SE Mean Turns, SE Mean Info Gain/Turn, SE Mean Info Gain, SE Mean Initial Entropy,
    SE Mean Seeker Tokens, SE Mean Seeker Reasoning Tokens, SE Mean Seeker Final Tokens, id

id: `s_<seeker>__o_<oracle>__p_<pruner>__<observabilidade>__<experimento>` (mesma convenção das pastas em outputs/models/).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.loader import load_experiment_results


def _compose_experiment_id(row: dict) -> str:
    """Id único alinhado a `outputs/models/s_*__o_*__p_*`/ — inclui os três papéis explicitamente."""
    s = (row.get("Seeker Model") or "").strip()
    o = (row.get("Oracle Model") or "").strip()
    p = (row.get("Pruner Model") or "").strip()
    obs = (row.get("Observabilidade") or "").strip()
    exp = (row.get("Experimento") or "").strip()
    return f"s_{s}__o_{o}__p_{p}__{obs}__{exp}"


def _infer_dataset(experiment_name: str | None) -> str | None:
    """Deriva o dataset (geo/objects/diseases) pelo prefixo do experiment_name."""
    if not experiment_name:
        return None
    exp = experiment_name.lower()
    for ds in ("geo", "objects", "diseases"):
        if exp.startswith(ds + "_") or f"_{ds}_" in exp:
            return ds
    return None


HEADERS = [
    "Experimento",
    "Dataset",
    "Seeker Model",
    "Oracle Model",
    "Pruner Model",
    "Observabilidade",
    "Total Runs",
    "Win Rate",
    "Mean Turns",
    "Mean Info Gain/Turn",
    "Mean Info Gain",
    "Mean Initial Entropy",
    "Mean Seeker Tokens",
    "Mean Seeker Reasoning Tokens",
    "Mean Seeker Final Tokens",
    "SE Win Rate",
    "SE Mean Turns",
    "SE Mean Info Gain/Turn",
    "SE Mean Info Gain",
    "SE Mean Initial Entropy",
    "SE Mean Seeker Tokens",
    "SE Mean Seeker Reasoning Tokens",
    "SE Mean Seeker Final Tokens",
    "id",
]


def _extract_from_summary(summary_path: Path) -> dict | None:
    """Extrai métricas do summary.json se existir."""
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        global_metrics = data.get("global_metrics", {}) or {}
        models = data.get("models", {}) or {}
        return {
            "Experimento": data.get("experiment_name"),
            "Dataset": _infer_dataset(data.get("experiment_name")),
            "Seeker Model": models.get("seeker"),
            "Oracle Model": models.get("oracle"),
            "Pruner Model": models.get("pruner"),
            "Observabilidade": (data.get("config", {}) or {}).get("observability"),
            "Total Runs": global_metrics.get("total_runs"),
            "Win Rate": global_metrics.get("win_rate"),
            "Mean Turns": global_metrics.get("mean_turns"),
            "Mean Info Gain/Turn": global_metrics.get("mean_avg_info_gain_per_turn"),
            "Mean Info Gain": global_metrics.get("mean_info_gain"),
            "Mean Initial Entropy": global_metrics.get("mean_h_start"),
            "Mean Seeker Tokens": global_metrics.get("mean_seeker_tokens"),
            "Mean Seeker Reasoning Tokens": global_metrics.get("mean_seeker_reasoning_tokens"),
            "Mean Seeker Final Tokens": global_metrics.get("mean_seeker_final_tokens"),
            "SE Win Rate": global_metrics.get("se_win_rate"),
            "SE Mean Turns": global_metrics.get("se_mean_turns"),
            "SE Mean Info Gain/Turn": global_metrics.get("se_mean_avg_info_gain_per_turn"),
            "SE Mean Info Gain": global_metrics.get("se_mean_info_gain"),
            "SE Mean Initial Entropy": global_metrics.get("se_mean_h_start"),
            "SE Mean Seeker Tokens": global_metrics.get("se_mean_seeker_tokens"),
            "SE Mean Seeker Reasoning Tokens": global_metrics.get("se_mean_seeker_reasoning_tokens"),
            "SE Mean Seeker Final Tokens": global_metrics.get("se_mean_seeker_final_tokens"),
        }
    except Exception:
        return None


def _extract_from_runs_csv(runs_csv: Path) -> dict | None:
    """Calcula métricas carregando o runs.csv se summary.json não existir."""
    try:
        results = load_experiment_results(runs_csv)
        return {
            "Experimento": results.experiment_name,
            "Dataset": _infer_dataset(results.experiment_name),
            "Seeker Model": results.seeker_model,
            "Oracle Model": results.oracle_model,
            "Pruner Model": results.pruner_model,
            "Observabilidade": results.observability,
            "Total Runs": results.total_runs,
            "Win Rate": round(results.global_win_rate, 4),
            "Mean Turns": round(results.mean_turns, 2),
            "Mean Info Gain/Turn": round(results.mean_avg_info_gain_per_turn, 4),
            "Mean Info Gain": round(results.mean_info_gain, 4),
            "Mean Initial Entropy": round(results.mean_h_start, 4),
            "Mean Seeker Tokens": round(results.mean_seeker_tokens, 0),
            "Mean Seeker Reasoning Tokens": round(results.mean_seeker_reasoning_tokens, 0) if results.mean_seeker_reasoning_tokens is not None else None,
            "Mean Seeker Final Tokens": round(results.mean_seeker_final_tokens, 0),
            "SE Win Rate": round(results.se_win_rate, 4),
            "SE Mean Turns": round(results.se_mean_turns, 2),
            "SE Mean Info Gain/Turn": round(results.se_mean_avg_info_gain_per_turn, 4),
            "SE Mean Info Gain": round(results.se_mean_info_gain, 4),
            "SE Mean Initial Entropy": round(results.se_mean_h_start, 4),
            "SE Mean Seeker Tokens": round(results.se_mean_seeker_tokens, 0),
            "SE Mean Seeker Reasoning Tokens": round(results.se_mean_seeker_reasoning_tokens, 0) if results.se_mean_seeker_reasoning_tokens is not None else None,
            "SE Mean Seeker Final Tokens": round(results.se_mean_seeker_final_tokens, 0),
        }
    except Exception:
        return None


def _summary_filename(only_run) -> str:
    """summary.json (None), summary_runNN.json (int/1-tuple) ou summary_runNN-MM.json (tuple)."""
    if only_run is None:
        return "summary.json"
    if isinstance(only_run, int):
        return f"summary_run{only_run:02d}.json"
    if len(only_run) == 1:
        return f"summary_run{only_run[0]:02d}.json"
    return "summary_run" + "-".join(f"{r:02d}" for r in only_run) + ".json"


def _iter_experiments(base_outputs_dir: Path, only_run=None) -> list[dict]:
    """Percorre o diretório base e coleta linhas para o CSV unificado.

    Args:
        base_outputs_dir: Diretório raiz de outputs.
        only_run: ``None``, int, ou tupla de ints. Determina qual summary_*.json ler.
                  Pastas sem o arquivo filtrado são puladas (não há fallback para runs.csv
                  quando only_run está ativo, pois o CSV filtrado ainda não foi gerado).
    """
    rows: list[dict] = []
    summary_filename = _summary_filename(only_run)

    # Preferir summary(_runNN).json quando disponível
    for summary_path in sorted(base_outputs_dir.rglob(summary_filename)):
        row = _extract_from_summary(summary_path)
        if row:
            row["id"] = _compose_experiment_id(row)
            rows.append(row)

    # Para pastas sem summary, tentar via runs.csv — apenas no modo padrão (sem only_run)
    if only_run is None:
        seen_ids = {r["id"] for r in rows}
        for runs_csv in sorted(base_outputs_dir.rglob("runs.csv")):
            if (runs_csv.parent / "summary.json").exists():
                continue
            row = _extract_from_runs_csv(runs_csv)
            if not row:
                continue
            row["id"] = _compose_experiment_id(row)
            if row["id"] in seen_ids:
                continue
            rows.append(row)
            seen_ids.add(row["id"])

    return rows


def main() -> int:
    """Ponto de entrada para geração do CSV unificado."""
    repo_root = Path(__file__).parent.parent.parent
    default_base = repo_root / "outputs"

    parser = argparse.ArgumentParser(
        description="Gera CSV unificado com métricas de todos os experimentos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python scripts/analysis/generate_unified_csv.py
  python scripts/analysis/generate_unified_csv.py --only-run 1
  python scripts/analysis/generate_unified_csv.py --base-dir outputs/ --output out.csv
        """,
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Diretório base de outputs (default: outputs/).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Caminho do CSV de saída (default: outputs/unified_experiments[_runNN].csv).",
    )
    parser.add_argument(
        "--only-run",
        type=str,
        default=None,
        metavar="N[,M,...]",
        help=(
            "Lê summary_runNN.json (ou summary_runNN-MM.json) em vez de summary.json. "
            "Aceita lista separada por vírgula (ex.: '1,2'). "
            "Salva em unified_experiments_runNN[-MM].csv."
        ),
    )

    args = parser.parse_args()

    # Parse "1,2" → (1, 2); "1" → (1,); None → None
    only_run = None
    if args.only_run:
        only_run = tuple(sorted({int(p) for p in str(args.only_run).split(",") if p.strip()}))

    base_outputs_dir = Path(args.base_dir) if args.base_dir else default_base

    if only_run is None:
        run_suffix = ""
    elif len(only_run) == 1:
        run_suffix = f"_run{only_run[0]:02d}"
    else:
        run_suffix = "_run" + "-".join(f"{r:02d}" for r in only_run)
    default_out = base_outputs_dir / f"unified_experiments{run_suffix}.csv"
    output_csv = Path(args.output) if args.output else default_out

    if not base_outputs_dir.exists():
        print(f"❌ Diretório de outputs não encontrado: {base_outputs_dir}")
        return 1

    summary_name = _summary_filename(only_run)
    print(f"🔎 Lendo experimentos em: {base_outputs_dir}")
    print(f"📄 Fonte: {summary_name}")
    rows = _iter_experiments(base_outputs_dir, only_run=only_run)

    if not rows:
        print(f"❌ Nenhum experimento encontrado ({summary_name} ou runs.csv)")
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"✅ CSV unificado salvo em: {output_csv}")
    print(f"📦 Total de experimentos: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


