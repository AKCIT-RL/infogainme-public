"""Centralized logging configuration for clary_quest.

Call `setup_logging()` once at the application entry point.
All modules use `logging.getLogger(__name__)` to get their logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(debug: bool = False, log_file: Path | str | None = None) -> None:
    """Configure root logger with a timestamped format.

    Args:
        debug: If True, set level to DEBUG; otherwise INFO.
        log_file: Optional path to write logs to a file (in addition to stdout).
    """
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
