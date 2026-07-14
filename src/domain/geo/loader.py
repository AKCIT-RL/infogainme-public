"""Geo candidates loader.

Loads a geographic CandidatePool from CSV for the benchmark.
Each city becomes a Candidate with country/region metadata.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple

from ...candidates import Candidate, CandidatePool
from ..types import DomainConfig, GEO_DOMAIN


def load_geo_candidates(
    csv_path: Path,
    domain_config: DomainConfig = GEO_DOMAIN,
) -> Tuple[CandidatePool, DomainConfig]:
    """Load geographic CandidatePool from world_flat.csv.

    Each city becomes a Candidate with attrs for country, state,
    region, and subregion.

    Args:
        csv_path: Path to the CSV file containing the geographic data.
        domain_config: Domain configuration. Defaults to GEO_DOMAIN.

    Returns:
        Tuple of (CandidatePool with city candidates, DomainConfig).

    Raises:
        FileNotFoundError: If the csv_path doesn't exist.
        ValueError: If insufficient data after filtering.
    """
    assert csv_path.exists(), f"Data file not found: {csv_path}"

    df_flat = pd.read_csv(csv_path, low_memory=False)

    col_city_id = "city_id"
    col_city_name = "city_name"
    col_state_id = "state_id"
    col_state_name = "state_name"
    col_country_id = "country_id"
    col_country_name = "country_name"
    col_region_id = "region_id"
    col_region_name = "region_name"
    col_subregion_id = "subregion_id"
    col_subregion_name = "subregion_name"

    cols_needed = [
        col_city_id, col_city_name,
        col_state_id, col_state_name,
        col_country_id, col_country_name,
        col_region_id, col_region_name,
        col_subregion_id, col_subregion_name,
    ]

    df_min = df_flat[cols_needed].dropna(subset=[col_city_id, col_state_id, col_country_id]).copy()
    unique_cities = df_min.drop_duplicates(subset=[col_city_id])

    prefix = domain_config.node_id_prefix.rstrip(":")
    candidates: list[Candidate] = []

    for _, row in unique_cities.iterrows():
        city_id = int(row[col_city_id])
        candidate_id = f"{prefix}:{city_id}"
        city_name = str(row[col_city_name])

        attrs: dict = {}
        if pd.notna(row[col_country_name]):
            attrs["country"] = str(row[col_country_name])
        if pd.notna(row[col_state_name]):
            attrs["state"] = str(row[col_state_name])
        if pd.notna(row[col_region_name]):
            attrs["region"] = str(row[col_region_name])
        if pd.notna(row[col_subregion_name]):
            attrs["subregion"] = str(row[col_subregion_name])

        candidates.append(
            Candidate(
                id=candidate_id,
                label=city_name,
                attrs=attrs,
            )
        )

    return CandidatePool(candidates=candidates), domain_config


# Backward-compatible alias (returns only the pool, dropping domain_config)
def load_geo_graph(csv_path: Path) -> CandidatePool:
    """Backward-compatible wrapper — returns CandidatePool instead of KnowledgeGraph."""
    pool, _ = load_geo_candidates(csv_path=csv_path)
    return pool


if __name__ == "__main__":
    pool, domain_config = load_geo_candidates(Path("data/top_10_pop_cities.csv"))
    active = pool.get_active()
    print(f"Loaded {len(active)} city candidates")
    for c in sorted(active, key=lambda c: c.label)[:5]:
        print(f"  {c.id}: {c.label} ({c.attrs})")
