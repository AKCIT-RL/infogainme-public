"""CandidatePool: simple flat list of candidates with pruning state.

Replaces KnowledgeGraph for all domains. No hierarchy, no edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A single candidate that could be the game target.

    Attributes:
        id: Stable unique identifier (e.g. "city:1234", "disease:flu:0").
        label: Human-readable name (e.g. "Tokyo", "Influenza").
        attrs: Arbitrary metadata dict (symptoms, aliases, country, etc.).
    """

    id: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePool:
    """Flat pool of candidates with active/pruned tracking.

    Attributes:
        candidates: All candidates (never modified after creation).
    """

    candidates: list[Candidate]
    _active: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._active = {c.label for c in self.candidates}

    def get_active(self) -> list[Candidate]:
        """Return candidates not yet pruned."""
        return [c for c in self.candidates if c.label in self._active]

    def prune(self, labels: set[str]) -> int:
        """Remove labels from active set.

        Args:
            labels: Set of candidate labels to eliminate.

        Returns:
            Number of candidates actually pruned.
        """
        before = len(self._active)
        self._active -= labels
        return before - len(self._active)

    def reset(self) -> None:
        """Restore all candidates to active state."""
        self._active = {c.label for c in self.candidates}

    def to_text(self) -> str:
        """Compact comma-separated list of active candidate names (for Seeker)."""
        active = self.get_active()
        names = ", ".join(c.label for c in sorted(active, key=lambda c: c.label))
        return f"Active candidates ({len(active)}):\n{names}"

    def to_rich_text(self) -> str:
        """Detailed list with attrs for the Pruner."""
        active = self.get_active()
        lines = [f"Active candidates ({len(active)}):"]
        for c in sorted(active, key=lambda c: c.label):
            parts = []
            for k, v in c.attrs.items():
                if k in ("type", "layer"):
                    continue
                if isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                else:
                    parts.append(f"{k}: {v}")
            if parts:
                lines.append(f"- {c.label} [{' | '.join(parts)}]")
            else:
                lines.append(f"- {c.label}")
        return "\n".join(lines)
