"""Writers for saving analysis results to JSON files."""

import json
from pathlib import Path

from .data_types import ExperimentResults


def save_summary(results: ExperimentResults, output_path: Path) -> None:
    """
    Salva summary.json com todas as métricas agregadas.
    
    Args:
        results: ExperimentResults
        output_path: Caminho para summary.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = results.summary_dict()
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"✅ Summary saved to: {output_path}")


def save_city_variance(results: ExperimentResults, output_path: Path) -> None:
    """
    Salva variance.json com foco nas estatísticas por cidade.
    
    Args:
        results: ExperimentResults
        output_path: Caminho para variance.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    variance_data = {
        "experiment_name": results.experiment_name,
        "total_cities": len(results.cities),
        "cities": {
            city_id: {
                "label": city.city_label,
                "runs": city.num_runs,
                "info_gain": {
                    "mean": round(city.mean_info_gain, 4),
                    "variance": round(city.var_info_gain, 4),
                    "std": round(city.std_info_gain, 4),
                },
                "turns": {
                    "mean": round(city.mean_turns, 2),
                    "std": round(city.std_turns, 2),
                },
                "win_rate": round(city.win_rate, 4),
            }
            for city_id, city in sorted(results.cities.items())
        }
    }
    
    output_path.write_text(json.dumps(variance_data, indent=2, ensure_ascii=False))
    print(f"✅ Variance saved to: {output_path}")

