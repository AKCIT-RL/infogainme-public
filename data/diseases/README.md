# Diseases dataset (flat, non-hierarchical)

Derived from `Final_Augmented_dataset_Diseases_and_Symptoms.csv`. Each disease has associated symptoms (union across augmented rows).

## Preprocessing

```bash
python scripts/prepare_diseases_csv.py
```

## CSV format

`disease,symptoms,aliases`

- **disease**: disease name (e.g. panic disorder)
- **symptoms**: semicolon-separated list of associated symptoms
- **aliases**: optional, semicolon-separated alternatives for Oracle matching

## Files

- `diseases_test.csv`: 40 diseases, quick experiments
- `diseases_full.csv`: 773 diseases, complete benchmark
