"""BenchmarkRunner for multi-game experiments.

Runs multiple games across targets and writes incremental results to CSV.
"""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import csv
import json

logger = logging.getLogger(__name__)

from .benchmark_config import BenchmarkConfig
from .candidates import Candidate, CandidatePool
from .orchestrator import Orchestrator


def _safe_name(text: str) -> str:
    return (
        text.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


class BenchmarkRunner:
    """Run multi-game benchmarks and persist results incrementally.

    Attributes:
        config: Benchmark configuration grouping agent models and options.
        output_base: Base directory to store results.
    """

    def __init__(self, config: BenchmarkConfig, output_base: str | Path = "outputs") -> None:
        self.config = config
        self.output_base = Path(output_base)

    def _output_dir(self) -> Path:
        seeker = _safe_name(self.config.seeker_config.model)
        oracle = _safe_name(self.config.oracle_config.model)
        pruner = _safe_name(self.config.pruner_config.model)
        exp = _safe_name(self.config.experiment_name or "default")
        return self.output_base / f"models/s_{seeker}__o_{oracle}__p_{pruner}" / exp

    def _csv_path(self) -> Path:
        return self._output_dir() / "runs.csv"

    def _ensure_header(self, csv_path: Path) -> None:
        if csv_path.exists():
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "experiment_name",
                    "seeker_model",
                    "oracle_model",
                    "pruner_model",
                    "observability",
                    "max_turns",
                    "target_id",
                    "target_label",
                    "run_index",
                    "turns",
                    "h_start",
                    "h_end",
                    "total_info_gain",
                    "avg_info_gain_per_turn",
                    "win",
                    "compliance_rate",
                    "conversation_path",
                ]
            )

    def _get_completed_runs(self, csv_path: Path) -> set[tuple[str, int]]:
        """Return set of (target_id, run_index) already completed for this experiment."""
        if not csv_path.exists():
            return set()

        completed = set()
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["experiment_name"] == (self.config.experiment_name or "default"):
                    completed.add((row["target_id"], int(row["run_index"])))

        return completed

    def run(
        self,
        *,
        pool: CandidatePool,
        targets: Iterable[Candidate],
        runs_per_target: int = 1,
        debug: bool = False,  # unused — verbosity controlled by log level
        max_workers: int = 1,
    ) -> Path:
        """Run the benchmark across targets and append results to CSV.

        Supports resuming interrupted experiments by detecting completed runs.
        When max_workers > 1 each game runs in its own thread with an isolated
        pool copy; CSV writes are serialised with a lock.

        Returns:
            Path to the 'runs.csv' file with appended results.
        """
        csv_path = self._csv_path()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header(csv_path)

        completed_runs = self._get_completed_runs(csv_path)

        work_items: list[tuple[Candidate, int]] = []
        skipped_count = 0

        for target in targets:
            for run_idx in range(1, int(runs_per_target) + 1):
                if (target.id, run_idx) in completed_runs:
                    skipped_count += 1
                    logger.debug(
                        "Skipping %s [%s] run %d/%d (already completed)",
                        target.label, target.id, run_idx, runs_per_target,
                    )
                    continue
                work_items.append((target, run_idx))

        total_planned = len(work_items) + skipped_count

        if skipped_count > 0:
            logger.info(
                "Resume | planned=%d | skipped=%d | to_run=%d",
                total_planned, skipped_count, len(work_items),
            )

        csv_lock = threading.Lock()

        def _run_one(target: Candidate, run_idx: int) -> None:
            logger.info("%s [%s] run %d/%d", target.label, target.id, run_idx, runs_per_target)

            game_pool = copy.deepcopy(pool)

            safe_id = _safe_name(target.id)
            conv_dir = (
                self._output_dir() / "conversations" /
                f"{safe_id}_run{run_idx:02d}"
            )

            orch = Orchestrator.from_target(
                target=target,
                pool=game_pool,
                seeker_config=self.config.seeker_config,
                oracle_config=self.config.oracle_config,
                pruner_config=self.config.pruner_config,
                observability_mode=self.config.observability_mode,
                max_turns=self.config.max_turns,
                domain_config=self.config.domain_config,
            )

            orch.run(debug=debug)

            conv_path_str = ""
            if self.config.save_conversations:
                orch.export_conversation(conv_dir)

                metadata_path = conv_dir / "metadata.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    metadata["config"]["experiment_name"] = self.config.experiment_name
                    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

                conv_path_str = str(conv_dir.relative_to(self.output_base))

            turns = orch.turns
            win = any(t.answer.game_over for t in turns)
            compliance_rate = (
                sum(1 for t in turns if t.answer.compliant) / len(turns)
            ) if turns else 0.0
            summary = orch.get_summary()

            row = [
                self.config.experiment_name or "default",
                self.config.seeker_config.model,
                self.config.oracle_config.model,
                self.config.pruner_config.model,
                self.config.observability_mode.name,
                self.config.max_turns,
                target.id,
                target.label,
                run_idx,
                summary.get("turns"),
                summary.get("h_start"),
                summary.get("h_end"),
                summary.get("total_info_gain"),
                summary.get("avg_info_gain_per_turn"),
                int(win),
                round(compliance_rate, 4),
                conv_path_str,
            ]

            with csv_lock:
                with csv_path.open("a", newline="") as f:
                    csv.writer(f).writerow(row)

            logger.info("%s done | win=%s | turns=%s", target.label, bool(win), summary.get("turns"))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_one, t, ri): (t, ri)
                for t, ri in work_items
            }
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    t, ri = futures[future]
                    logger.error("%s run %d failed: %s", t.label, ri, exc)

        return csv_path
