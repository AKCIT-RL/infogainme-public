"""Loader for flat (non-hierarchical) object dataset from CSV.

Creates a CandidatePool with one Candidate per object.

CSV format: category,label,aliases
- category: e.g. Sports, Animals
- label: main name (e.g. Airplane)
- aliases: optional, semicolon-separated (e.g. Plane;Aircraft)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Tuple

from ...candidates import Candidate, CandidatePool
from ..types import DomainConfig, OBJECTS_DOMAIN


def _slug(text: str) -> str:
    """Create a URL-safe slug from text."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "_", s).strip("_")


def _parse_aliases(aliases_str: str) -> List[str]:
    """Parse aliases string: 'A;B;C' -> ['A', 'B', 'C']."""
    if not aliases_str or not aliases_str.strip():
        return []
    return [p.strip() for p in aliases_str.split(";") if p.strip()]


def load_flat_object_candidates(
    csv_path: Path,
    domain_config: DomainConfig = OBJECTS_DOMAIN,
) -> Tuple[CandidatePool, DomainConfig]:
    """Load a CandidatePool from objects CSV.

    CSV columns: category, label, aliases
    - category: object category (Sports, Animals, etc.)
    - label: main display name
    - aliases: optional semicolon-separated alternatives for Oracle matching

    Args:
        csv_path: Path to CSV file.
        domain_config: Domain configuration. Defaults to OBJECTS_DOMAIN.

    Returns:
        Tuple of (CandidatePool with object candidates, DomainConfig).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Objects CSV not found: {csv_path}")

    candidates: list[Candidate] = []
    prefix = domain_config.node_id_prefix.rstrip(":")
    cat_indices: dict[str, int] = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("category") or "").strip()
            label = (row.get("label") or "").strip()
            if not category or not label:
                continue
            aliases = _parse_aliases(row.get("aliases") or "")

            cat_slug = _slug(category)
            idx = cat_indices.get(cat_slug, 0)
            cat_indices[cat_slug] = idx + 1

            candidate_id = f"{prefix}:{cat_slug}:{idx}"
            attrs: dict = {"category": category}
            if aliases:
                attrs["aliases"] = aliases

            candidates.append(
                Candidate(
                    id=candidate_id,
                    label=label,
                    attrs=attrs,
                )
            )

    return CandidatePool(candidates=candidates), domain_config


# Backward-compatible alias
load_flat_object_graph = load_flat_object_candidates
