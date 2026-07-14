"""Loader for flat disease dataset from CSV.

Creates a CandidatePool with one Candidate per disease.

CSV format: disease,symptoms,aliases
- disease: disease name (e.g. panic disorder)
- symptoms: semicolon-separated list of symptoms
- aliases: optional semicolon-separated alternatives for Oracle matching
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Tuple

from ...candidates import Candidate, CandidatePool
from ..types import DomainConfig, DISEASES_DOMAIN


def _slug(text: str) -> str:
    """Create a URL-safe slug from text."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "_", s).strip("_")


def _parse_list(value: str) -> List[str]:
    """Parse semicolon-separated string into list."""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(";") if p.strip()]


def load_flat_disease_candidates(
    csv_path: Path,
    domain_config: DomainConfig = DISEASES_DOMAIN,
) -> Tuple[CandidatePool, DomainConfig]:
    """Load a CandidatePool from diseases CSV.

    CSV columns: disease, symptoms, aliases
    - disease: disease name
    - symptoms: semicolon-separated list of associated symptoms
    - aliases: optional semicolon-separated alternatives for Oracle matching

    Args:
        csv_path: Path to CSV file.
        domain_config: Domain configuration. Defaults to DISEASES_DOMAIN.

    Returns:
        Tuple of (CandidatePool with disease candidates, DomainConfig).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Diseases CSV not found: {csv_path}")

    candidates: list[Candidate] = []
    prefix = domain_config.node_id_prefix.rstrip(":")
    idx = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = (row.get("disease") or "").strip()
            if not disease:
                continue
            symptoms = _parse_list(row.get("symptoms") or "")
            aliases = _parse_list(row.get("aliases") or "")

            disease_slug = _slug(disease)
            candidate_id = f"{prefix}:{disease_slug}:{idx}"
            idx += 1

            attrs: dict = {"category": "medical"}
            if symptoms:
                attrs["symptoms"] = symptoms
            if aliases:
                attrs["aliases"] = aliases

            candidates.append(
                Candidate(
                    id=candidate_id,
                    label=disease,
                    attrs=attrs,
                )
            )

    return CandidatePool(candidates=candidates), domain_config


# Backward-compatible alias
load_flat_disease_graph = load_flat_disease_candidates
