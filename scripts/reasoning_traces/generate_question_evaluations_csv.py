"""Gera um CSV unificado com métricas de avaliação de escolhas do Seeker.

Usage:
    # Varre ./outputs e salva em ./outputs/question_evaluations_unified.csv
    python scripts/generate_question_evaluations_csv.py

    # Informar diretório base de outputs e caminho de saída
    python scripts/generate_question_evaluations_csv.py [base_outputs_dir] [output_csv_path]

Colunas geradas:
    Experimento, Seeker Model, Observabilidade, Total Turns Evaluated,
    Total Optimal Choices, Avg Optimal Choice Rate, Avg Chosen Info Gain,
    Avg Optimal Info Gain, Avg Questions Considered Per Turn,
    SE Optimal Choice Rate, SE Chosen Info Gain, SE Optimal Info Gain,
    SE Questions Considered Per Turn, id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Garantir imports do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.loader import load_experiment_results


def _extract_experiment_info(summary_path: Path | None, runs_csv: Path | None) -> dict | None:
    """Extrai informações do experimento de summary.json ou runs.csv.
    
    Args:
        summary_path: Caminho para summary.json, se existir.
        runs_csv: Caminho para runs.csv, se existir.
        
    Returns:
        Dicionário com informações do experimento ou None se não encontrar.
    """
    # Tentar summary.json primeiro
    if summary_path and summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return {
                "Experimento": data.get("experiment_name"),
                "Seeker Model": (data.get("models", {}) or {}).get("seeker"),
                "Observabilidade": (data.get("config", {}) or {}).get("observability"),
            }
        except Exception:
            pass
    
    # Fallback para runs.csv
    if runs_csv and runs_csv.exists():
        try:
            results = load_experiment_results(runs_csv)
            return {
                "Experimento": results.experiment_name,
                "Seeker Model": results.seeker_model,
                "Observabilidade": results.observability,
            }
        except Exception:
            pass
    
    return None


def _extract_question_evaluations(eval_summary_path: Path) -> dict | None:
    """Extrai métricas do question_evaluations_summary.json.
    
    Extrai métricas do aggregate_statistics:
    - total_turns_evaluated
    - total_optimal_choices
    - avg_optimal_choice_rate
    - avg_chosen_info_gain
    - avg_optimal_info_gain
    - avg_questions_considered_per_turn
    - se_optimal_choice_rate
    - se_chosen_info_gain
    - se_optimal_info_gain
    - se_questions_considered_per_turn
    
    Args:
        eval_summary_path: Caminho para question_evaluations_summary.json.
        
    Returns:
        Dicionário com métricas extraídas ou None se houver erro.
    """
    try:
        data = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        aggregate_stats = data.get("aggregate_statistics", {}) or {}
        
        return {
            "Total Turns Evaluated": aggregate_stats.get("total_turns_evaluated"),
            "Total Optimal Choices": aggregate_stats.get("total_optimal_choices"),
            "Avg Optimal Choice Rate": aggregate_stats.get("avg_optimal_choice_rate"),
            "Avg Chosen Info Gain": aggregate_stats.get("avg_chosen_info_gain"),
            "Avg Optimal Info Gain": aggregate_stats.get("avg_optimal_info_gain"),
            "Avg Questions Considered Per Turn": aggregate_stats.get("avg_questions_considered_per_turn"),
            "SE Optimal Choice Rate": aggregate_stats.get("se_optimal_choice_rate"),
            "SE Chosen Info Gain": aggregate_stats.get("se_chosen_info_gain"),
            "SE Optimal Info Gain": aggregate_stats.get("se_optimal_info_gain"),
            "SE Questions Considered Per Turn": aggregate_stats.get("se_questions_considered_per_turn"),
        }
    except Exception as e:
        print(f"⚠️  Erro ao processar {eval_summary_path}: {e}")
        return None


def _iter_question_evaluations(base_outputs_dir: Path) -> list[dict]:
    """Percorre o diretório base e coleta linhas para o CSV unificado.
    
    Args:
        base_outputs_dir: Diretório base onde procurar os arquivos.
        
    Returns:
        Lista de dicionários com dados de cada experimento.
    """
    rows: list[dict] = []
    
    # Buscar todos os question_evaluations_summary.json
    for eval_summary_path in sorted(base_outputs_dir.rglob("question_evaluations_summary.json")):
        # Tentar encontrar summary.json ou runs.csv no mesmo diretório
        parent_dir = eval_summary_path.parent
        summary_path = parent_dir / "summary.json"
        runs_csv = parent_dir / "runs.csv"
        
        # Extrair informações do experimento
        exp_info = _extract_experiment_info(summary_path, runs_csv)
        
        # Extrair métricas de avaliação
        eval_metrics = _extract_question_evaluations(eval_summary_path)
        
        if exp_info and eval_metrics:
            # Combinar informações
            row = {**exp_info, **eval_metrics}
            
            # Adicionar ID composto
            row["id"] = (
                f"{row.get('Seeker Model', '')}_"
                f"{row.get('Observabilidade', '')}_"
                f"{row.get('Experimento', '')}"
            )
            
            rows.append(row)
        else:
            print(f"⚠️  Não foi possível extrair informações completas de: {eval_summary_path}")
    
    return rows


def main() -> int:
    """Ponto de entrada para geração do CSV unificado.
    
    Returns:
        Código de saída (0 = sucesso, 1 = erro).
    """
    repo_root = Path(__file__).parent.parent.parent
    default_base = repo_root / "outputs"
    default_out = default_base / "question_evaluations_unified.csv"

    # Argumentos opcionais
    if len(sys.argv) >= 2:
        base_outputs_dir = Path(sys.argv[1])
    else:
        base_outputs_dir = default_base

    if len(sys.argv) >= 3:
        output_csv = Path(sys.argv[2])
    else:
        output_csv = default_out

    if not base_outputs_dir.exists():
        print(f"❌ Diretório de outputs não encontrado: {base_outputs_dir}")
        print(f"Usage: python {Path(__file__).name} [base_outputs_dir] [output_csv]")
        return 1

    print(f"🔎 Lendo avaliações de escolhas em: {base_outputs_dir}")
    rows = _iter_question_evaluations(base_outputs_dir)

    if not rows:
        print("❌ Nenhum question_evaluations_summary.json encontrado")
        return 1

    # Criar DataFrame e ordenar colunas
    df = pd.DataFrame(rows)
    
    # Definir ordem das colunas: identificadoras primeiro, depois métricas, por último id
    column_order = [
        "Experimento",
        "Seeker Model",
        "Observabilidade",
        "Total Turns Evaluated",
        "Total Optimal Choices",
        "Avg Optimal Choice Rate",
        "SE Optimal Choice Rate",
        "Avg Chosen Info Gain",
        "SE Chosen Info Gain",
        "Avg Optimal Info Gain",
        "SE Optimal Info Gain",
        "Avg Questions Considered Per Turn",
        "SE Questions Considered Per Turn",
        "id",
    ]
    
    # Garantir que todas as colunas existam
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    # Salvar CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ CSV unificado salvo em: {output_csv}")
    print(f"📦 Total de experimentos: {len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

