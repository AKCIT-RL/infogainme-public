#!/usr/bin/env python3
"""Generate data/diseases/diseases_160.csv for benchmark experiments.

Randomly samples 160 diseases from diseases_full.csv (773 available).
Uses a fixed random seed for reproducibility.

Run with:
  python3 data/create_diseases_160.py
"""

import pandas as pd
from pathlib import Path

SRC  = Path("data/diseases/diseases_full.csv")
OUT  = Path("data/diseases/diseases_160.csv")
N    = 160
SEED = 42

df = pd.read_csv(SRC)
print(f"Total disponível: {len(df)}")

sample = df.sample(n=N, random_state=SEED).sort_values("disease").reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(OUT, index=False)
print(f"Saved {len(sample)} diseases → {OUT}")
print(sample["disease"].tolist())
