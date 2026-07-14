"""Loader for experiment results from CSV files."""

import csv
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from .data_types import GameRun, CityStats, ExperimentResults
from ..utils.token_counter import count_seeker_tokens


def _safe_int(value) -> int | None:
    """Converte value para int, retornando None em caso de falha."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def load_experiment_results(csv_path: Path, only_run: "int | tuple[int, ...] | list[int] | set[int] | None" = None) -> ExperimentResults:
    """
    Carrega um arquivo runs.csv e retorna ExperimentResults estruturado.

    Args:
        csv_path: Caminho para runs.csv
        only_run: Se especificado, filtra apenas as linhas cujo run_index está
            no conjunto fornecido. Aceita um único int (compat) ou tupla/lista/set
            de ints para múltiplos runs (ex.: ``(1, 2)`` para combinar run01+run02).

    Returns:
        ExperimentResults com todas as métricas calculadas

    Raises:
        ValueError: Se o CSV estiver vazio ou malformado
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vazio: {csv_path}")

    if only_run is not None:
        allowed = {only_run} if isinstance(only_run, int) else set(only_run)
        rows = [r for r in rows if _safe_int(r.get("run_index")) in allowed]
        if not rows:
            raise ValueError(f"Nenhuma linha com run_index in {sorted(allowed)} em: {csv_path}")
    
    # Extrair metadata do experimento (da primeira linha)
    first = rows[0]
    experiment_name = first["experiment_name"]
    seeker_model = first["seeker_model"]
    oracle_model = first["oracle_model"]
    pruner_model = first["pruner_model"]
    observability = first["observability"]
    max_turns = int(first["max_turns"])
    
    # Agrupar runs por cidade
    cities_data = defaultdict(list)
    csv_dir = csv_path.parent
    
    for row in tqdm(rows, desc="Processando runs e contando tokens", unit="run"):
        # Calcular tokens do Seeker se conversation_path disponível
        seeker_total_tokens = 0
        seeker_reasoning_tokens = None
        seeker_final_tokens = 0
        
        conversation_path = row.get("conversation_path")
        if conversation_path:
            # conversation_path é relativo ao output_base (outputs/)
            # csv_dir está em outputs/models/.../experiment_name/
            # Precisamos subir até outputs/ e então adicionar conversation_path
            output_base = csv_dir.parent.parent.parent  # outputs/
            conv_dir = output_base / conversation_path
            seeker_json_path = conv_dir / "seeker.json"
            token_cache_path = conv_dir / "token_cache.json"

            if seeker_json_path.exists():
                try:
                    # Usar cache se existir e for mais novo que seeker.json
                    if (
                        token_cache_path.exists()
                        and token_cache_path.stat().st_mtime >= seeker_json_path.stat().st_mtime
                    ):
                        cache = json.loads(token_cache_path.read_text(encoding="utf-8"))
                        seeker_total_tokens = cache.get("total", 0)
                        seeker_reasoning_tokens = cache.get("reasoning")  # None se não houver
                        seeker_final_tokens = cache.get("final", 0)
                    else:
                        with seeker_json_path.open("r", encoding="utf-8") as f:
                            seeker_data = json.load(f)

                        reasoning_history = seeker_data.get("reasoning_history", [])
                        history = seeker_data.get("history", [])
                        model = seeker_data.get("config", {}).get("model")

                        total, reasoning, final = count_seeker_tokens(
                            reasoning_history, history, model
                        )
                        seeker_total_tokens = total
                        seeker_reasoning_tokens = reasoning
                        seeker_final_tokens = final

                        # Salvar cache
                        token_cache_path.write_text(
                            json.dumps({"total": total, "reasoning": reasoning, "final": final}),
                            encoding="utf-8",
                        )
                except Exception:
                    pass
        
        game_run = GameRun(
            target_id=row["target_id"],
            target_label=row["target_label"],
            run_index=int(row["run_index"]),
            turns=int(row["turns"]),
            h_start=float(row["h_start"]),
            h_end=float(row["h_end"]),
            total_info_gain=float(row["total_info_gain"]),
            avg_info_gain_per_turn=float(row.get("avg_info_gain_per_turn", 0.0)),
            win=bool(int(row["win"])),
            compliance_rate=float(row["compliance_rate"]),
            conversation_path=conversation_path or None,
            seeker_total_tokens=seeker_total_tokens,
            seeker_reasoning_tokens=seeker_reasoning_tokens,
            seeker_final_tokens=seeker_final_tokens,
        )
        cities_data[game_run.target_id].append(game_run)
    
    # Criar CityStats para cada cidade
    cities = {}
    for city_id, runs in cities_data.items():
        city_label = runs[0].target_label  # Todas runs têm mesmo label
        cities[city_id] = CityStats(
            city_id=city_id,
            city_label=city_label,
            runs=runs,
        )
    
    return ExperimentResults(
        experiment_name=experiment_name,
        seeker_model=seeker_model,
        oracle_model=oracle_model,
        pruner_model=pruner_model,
        observability=observability,
        max_turns=max_turns,
        cities=cities,
    )

