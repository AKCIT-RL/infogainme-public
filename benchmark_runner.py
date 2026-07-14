#!/usr/bin/env python3
"""Run multi-game benchmark experiments using YAML configuration.

Supports both geographic (geo) and flat object/disease datasets via dataset.type in config.
"""

import argparse
import logging
from datetime import datetime
from os import getenv
from pathlib import Path
from dotenv import load_dotenv

from src.logging_config import setup_logging
from src.domain.geo.loader import load_geo_candidates
from src.domain.objects import load_flat_object_candidates
from src.domain.diseases import load_flat_disease_candidates
from src.utils.config_loader import load_benchmark_config
from src.benchmark import BenchmarkRunner

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full benchmark experiment."""
    parser = argparse.ArgumentParser(description="Run benchmark experiments")
    parser.add_argument("--config", type=Path, default="configs/geo_full_no_cot.yaml")
    parser.add_argument("--servers-override", type=Path, default=None, help="Path to servers override YAML file")
    parser.add_argument("--runs-per-target", type=int, default=None,
                        help="Override runs_per_target from config (e.g. 1 for a quick first pass)")
    args = parser.parse_args()

    load_dotenv()
    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        # Setup basic logging so the error is visible
        setup_logging()
        logger.error("OPENAI_API_KEY environment variable not set!")
        return

    try:
        benchmark_config, config = load_benchmark_config(args.config, api_key, args.servers_override)
    except Exception as e:
        setup_logging()
        logger.error("Error loading configuration: %s", e)
        return

    debug = config["debug"]["enabled"]
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "geo")
    num_targets = dataset_cfg.get("num_targets")
    runs_per_target = args.runs_per_target if args.runs_per_target is not None else dataset_cfg.get("runs_per_target", 1)
    output_base = Path(config["output"]["base_dir"])

    log_file = output_base / "logs" / f"{benchmark_config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file=log_file)

    if dataset_type == "objects":
        csv_path = Path(dataset_cfg["csv_path"])
        pool, _ = load_flat_object_candidates(csv_path=csv_path)
        all_targets = sorted(pool.candidates, key=lambda c: c.id)
    elif dataset_type == "diseases":
        csv_path = Path(dataset_cfg["csv_path"])
        pool, _ = load_flat_disease_candidates(csv_path=csv_path)
        all_targets = sorted(pool.candidates, key=lambda c: c.id)
    else:
        csv_path = Path(dataset_cfg["csv_path"])
        pool, _ = load_geo_candidates(csv_path=csv_path)
        all_targets = sorted(pool.candidates, key=lambda c: c.id)

    targets = all_targets[:num_targets] if num_targets else all_targets
    total_games = len(targets) * runs_per_target

    logger.info("Clary Quest - Multi-Game Benchmark")
    logger.info("Config: %s", args.config)
    logger.info(
        "Dataset: %s | Targets: %d | Runs/target: %d",
        dataset_type, len(targets), runs_per_target,
    )
    logger.info("Experiment: %s", benchmark_config.experiment_name)
    logger.info(
        "Models: seeker=%s | oracle=%s | pruner=%s",
        benchmark_config.seeker_config.model,
        benchmark_config.oracle_config.model,
        benchmark_config.pruner_config.model,
    )
    logger.info(
        "Settings: %s | %d turns | total_games=%d",
        benchmark_config.observability_mode.name,
        benchmark_config.max_turns,
        total_games,
    )

    max_workers = config.get("experiment", {}).get("max_workers", 1)

    runner = BenchmarkRunner(config=benchmark_config, output_base=output_base)
    csv_path = runner.run(
        pool=pool,
        targets=targets,
        runs_per_target=runs_per_target,
        debug=debug,
        max_workers=max_workers,
    )

    logger.info("Benchmark complete. Results: %s", csv_path)


if __name__ == "__main__":
    main()
