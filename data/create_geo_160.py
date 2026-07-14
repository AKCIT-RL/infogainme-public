#!/usr/bin/env python3
"""Generate data/geo/top_160_pop_cities.csv for benchmark experiments.

Sources:
  - GeoNames cities15000.txt  → city data + population
  - GeoNames admin1CodesASCII.txt → state/province names
  - dr5hn/countries-states-cities-database (GitHub JSON) → country_id, region, subregion

Run with:
  python3 data/create_geo_160.py
"""

import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

OUT_PATH = Path("data/geo/top_160_pop_cities.csv")
N = 160

# ---------------------------------------------------------------------------
# 1. GeoNames: top N cities by population
# ---------------------------------------------------------------------------
print("Fetching GeoNames cities15000 dataset...")
resp = requests.get("https://download.geonames.org/export/dump/cities15000.zip", timeout=120)
resp.raise_for_status()
with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
    with z.open("cities15000.txt") as f:
        raw = f.read().decode("utf-8")

GEO_COLS = [
    "geonameid", "name", "asciiname", "alternatenames",
    "latitude", "longitude", "feature_class", "feature_code",
    "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]
geonames = pd.read_csv(
    io.StringIO(raw), sep="\t", header=None, names=GEO_COLS,
    dtype={"geonameid": "Int64", "population": "Int64"},
    low_memory=False,
)
cities_gn = geonames[geonames["feature_class"] == "P"].copy()
top_gn = (
    cities_gn.sort_values("population", ascending=False)
    .drop_duplicates(subset=["asciiname", "country_code"])
    .head(N)
    .reset_index(drop=True)
)
print(f"GeoNames top {N} cities selected.")

# ---------------------------------------------------------------------------
# 2. GeoNames admin1CodesASCII: state/province names
# ---------------------------------------------------------------------------
print("Fetching GeoNames admin1 codes...")
resp2 = requests.get(
    "https://download.geonames.org/export/dump/admin1CodesASCII.txt", timeout=60
)
resp2.raise_for_status()
admin1 = pd.read_csv(
    io.StringIO(resp2.text), sep="\t", header=None,
    names=["code", "name", "name_ascii", "geonameid"],
)
# code format: "US.CA" → split into country_code + admin1_code
admin1[["a1_country", "a1_code"]] = admin1["code"].str.split(".", n=1, expand=True)
admin1 = admin1.rename(columns={"name_ascii": "state_name"})[
    ["a1_country", "a1_code", "state_name"]
]

# ---------------------------------------------------------------------------
# 3. dr5hn: countries, regions, subregions (no cities.json needed)
# ---------------------------------------------------------------------------
BASE = "https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/master"

def fetch_json(name):
    r = requests.get(f"{BASE}/json/{name}.json", timeout=60)
    r.raise_for_status()
    return pd.read_json(io.StringIO(r.text))

print("Fetching dr5hn countries/regions/subregions/states...")
countries  = fetch_json("countries")
regions    = fetch_json("regions")
subregions = fetch_json("subregions")
states_db  = fetch_json("states")

for col in ["id", "region_id", "subregion_id"]:
    countries[col] = pd.to_numeric(countries[col], errors="coerce").astype("Int64")
for col in ["id", "country_id"]:
    states_db[col] = pd.to_numeric(states_db[col], errors="coerce").astype("Int64")
regions["id"]           = pd.to_numeric(regions["id"],           errors="coerce").astype("Int64")
subregions["id"]        = pd.to_numeric(subregions["id"],        errors="coerce").astype("Int64")
subregions["region_id"] = pd.to_numeric(subregions["region_id"], errors="coerce").astype("Int64")

# Build country lookup: iso2 → country_id, country_name, region_id, subregion_id
countries_sel = countries.rename(columns={
    "id": "country_id", "name": "country_name", "iso2": "country_iso2",
})[["country_id", "country_name", "country_iso2", "region_id", "subregion_id"]]

regions_sel    = regions.rename(columns={"id": "region_id_ref",    "name": "region_name"})[["region_id_ref",    "region_name"]]
subregions_sel = subregions.rename(columns={"id": "subregion_id_ref", "name": "subregion_name"})[["subregion_id_ref", "subregion_name"]]

# Build states lookup: country_id + iso2 → state_id, state_name
states_sel = states_db.rename(columns={"id": "state_id", "name": "state_name_db", "iso2": "state_code"})[
    ["state_id", "state_name_db", "state_code", "country_id"]
]

# ---------------------------------------------------------------------------
# 4. Merge everything together
# ---------------------------------------------------------------------------
df = top_gn[["geonameid", "asciiname", "country_code", "admin1_code", "population"]].copy()
df = df.rename(columns={"geonameid": "city_id", "asciiname": "city_name"})

# Join country info
df = df.merge(
    countries_sel,
    left_on="country_code", right_on="country_iso2",
    how="left",
)

# Join region/subregion names
df = df.merge(regions_sel,    left_on="region_id",    right_on="region_id_ref",    how="left")
df = df.merge(subregions_sel, left_on="subregion_id", right_on="subregion_id_ref", how="left")

# Join state name from GeoNames admin1
df = df.merge(
    admin1, left_on=["country_code", "admin1_code"],
    right_on=["a1_country", "a1_code"], how="left",
)

# Join state_id from dr5hn (match by country_id + state_code ≈ admin1_code)
df = df.merge(
    states_sel[["state_id", "state_name_db", "state_code", "country_id"]].rename(
        columns={"country_id": "s_country_id"}
    ),
    left_on=["country_id", "admin1_code"],
    right_on=["s_country_id", "state_code"],
    how="left",
)

# Fallback state_name: prefer GeoNames admin1 name, else dr5hn name
df["state_name"] = df["state_name"].fillna(df["state_name_db"])

# ---------------------------------------------------------------------------
# 5. Synthesise state_id for cities that didn't match dr5hn states
#    (state_id is required by the loader's dropna check but unused thereafter)
# ---------------------------------------------------------------------------
# Use a synthetic ID: 900_000_000 + country_id * 1000 + numeric admin1 slot
admin1_slot = df.groupby("country_id")["admin1_code"].transform(
    lambda s: pd.factorize(s)[0]
)
synthetic_state_id = (900_000_000 + df["country_id"].fillna(0).astype(int) * 1000 + admin1_slot).astype("Int64")
df["state_id"] = df["state_id"].fillna(synthetic_state_id)

# ---------------------------------------------------------------------------
# 6. Build final CSV (same schema as existing top_*_pop_cities.csv)
# ---------------------------------------------------------------------------
final = df[[
    "city_id", "city_name", "state_id", "state_name",
    "country_id", "country_name",
    "region_id", "region_name",
    "subregion_id", "subregion_name",
    "population",
]].rename(columns={"population": "population_2025"})

final = final.sort_values("population_2025", ascending=False).reset_index(drop=True)

# Report coverage
no_country = final["country_id"].isna().sum()
no_state   = final["state_id"].isna().sum()
no_region  = final["region_id"].isna().sum()
print(f"\nCoverage — no country_id: {no_country} | no state_id: {no_state} | no region_id: {no_region}")
if no_country > 0:
    print("  Missing country_id:")
    print(final[final["country_id"].isna()][["city_name", "country_name"]].to_string())

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
final.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(final)} cities → {OUT_PATH}")
print(final[["city_name", "country_name", "population_2025"]].to_string())
