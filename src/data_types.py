"""Shared types.

Provides `TurnState`, `PruningResult`, and enums per UML.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Literal, Optional, Any
from pydantic import BaseModel


class ObservabilityMode(Enum):
    FULLY_OBSERVABLE = "FO"
    PARTIALLY_OBSERVABLE = "PO"
    INITIALLY_OBSERVABLE = "IO"

@dataclass
class Question:
    text: str


class OracleResponse(BaseModel):
    rationale: str
    answer: Literal["Yes", "No", "Yes! You found the target!"]
    game_over: bool

@dataclass
class Answer:
    rationale: str
    text: str
    compliant: bool
    game_over: bool = False


class PrunerResponse(BaseModel):
    rationale: str
    keep_labels: list[str]

@dataclass
class PruningResult:
    pruned_labels: set[str]
    rationale: str


@dataclass
class TurnState:
    turn_index: int
    h_before: float
    h_after: float
    info_gain: float
    pruned_count: int
    question: Question
    answer: Answer

    # Additional metadata for conversation export
    pruning_result: Optional[PruningResult] = None
    active_candidates_before: Optional[int] = None
    active_candidates_after: Optional[int] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    duration_seconds: Optional[float] = None
    candidates_snapshot: Optional[list[str]] = None

    def to_export_dict(self) -> dict[str, Any]:
        """Convert TurnState to dictionary for JSONL export.

        Uses dataclasses.asdict() to preserve all attributes, then converts
        sets to lists for JSON serialization.

        Returns:
            Dictionary with all turn data ready for JSON export.
        """
        data = asdict(self)

        # Convert pruning_result.pruned_labels from set to list for JSON serialization
        if data.get("pruning_result") and "pruned_labels" in data["pruning_result"]:
            data["pruning_result"]["pruned_labels"] = list(data["pruning_result"]["pruned_labels"])

        return data
