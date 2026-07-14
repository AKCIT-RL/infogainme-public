# InfoGainMe

**InfoGainMe** is a benchmark that measures **information gain** in LLM conversations using a three-agent architecture — Seeker, Oracle, and Pruner — across three domains (geography, objects, diseases) and three observability modes.

---

## Overview

The Seeker asks yes/no questions to identify a secret target from a candidate set Ω₀. The Oracle answers truthfully, and the Pruner eliminates inconsistent candidates after each turn. **Information gain** is Shannon entropy reduction: `IG_t = H(Ω_{t-1}) - H(Ω_t)`, where `H = log₂|Ω|`.

Three observability modes control what the Seeker sees:
- **FO (Fully Observable)**: active candidate set shown every turn
- **IO (Initially Observable)**: shown once at the start, then hidden
- **PO (Partially Observable)**: never shown

## Installation

```bash
git clone <repo-url>
cd infogainme-public
pip install -r requirements.txt
cp .env.example .env  # fill in your tokens
```

## Running the benchmark

Requires a running [vLLM](https://github.com/vllm-project/vllm) instance. Update `configs/servers.yaml` with your endpoint, then:

```bash
python benchmark_runner.py --config configs/full/8b/geo_160_8b_fo_cot.yaml
```

Results are written to `outputs/` and are **resumable** — re-running the same config skips completed games.

## Reproducing paper results

### Step 0 — Get the data

Every result in the paper is derived from the released runs, published as a
HuggingFace dataset. Download them instead of re-running the benchmark:

```bash
huggingface-cli download <DATASET_ID> --repo-type dataset --local-dir outputs
# each experiment ships a conversations.zip — unpack them in place
find outputs/models -name conversations.zip -execdir unzip -q -o {} \;
```

> The dataset identifier is withheld here for anonymous review.

This gives you `outputs/models/<triple>/<experiment>/` with `runs.csv`,
`summary.json`, `variance.json` and the full `conversations/` tree
(`turns.jsonl`, `seeker.json`, `seeker_traces.json`, …).

It also ships `outputs/aggregates/`, the four LLM-derived artifacts the paper's
analyses are built on (all extracted with `gemma-4-31B-it`):

| File | Produced by | Used for |
|---|---|---|
| `seeker_traces.jsonl` | `synthesize_traces.py` | structured CoT reasoning per turn |
| `question_evaluations.jsonl` | `evaluate_all_seeker_choices.py` | counterfactual IG of considered questions |
| `question_classifications.jsonl` | `classify_questions.py` | question taxonomy + redundancy |
| `belief_states.jsonl` | `extract_belief_states.py` | recovered belief state vs. true Ω_t |

Move them to `outputs/` (`mv outputs/aggregates/*.jsonl outputs/`) and Steps 3–5
below become pure post-processing — **no GPU required**. Re-run those steps only
if you want to regenerate the artifacts from scratch.

To regenerate the runs from scratch instead (requires vLLM serving the seeker
and the Qwen3-8B oracle/pruner):

```bash
python benchmark_runner.py --config configs/full/<model>/<config>.yaml
```

### Step 1 — Metrics tables

Reads only the released runs; no LLM needed.

Run these from the repository root (paths resolve relative to it, not to `$PWD`):

```bash
python scripts/analysis/analyze_results.py --all        # per-experiment summary.json + variance.json
python scripts/analysis/generate_unified_csv.py         # outputs/unified_experiments.csv
python scripts/analysis/generate_model_summary_csv.py   # outputs/model_summary.csv
```

`unified_experiments.csv` (one row per experiment: win rate, total IG, IG/turn,
turns, compliance) is the raw table behind the main results.

### Step 2 — Figures

The two Ω/IG facet figures. No LLM needed.

```bash
python scripts/analysis/plot_omega_ig_dual_axis.py --facet --canonical --no-cot \
  --ylim-omega 160 --ylim-ig 1.52 --out outputs/plots/omega_ig_facet_no_cot.pdf

python scripts/analysis/plot_omega_ig_dual_axis.py --facet --canonical --cot \
  --ylim-omega 160 --ylim-ig 1.52 --out outputs/plots/omega_ig_facet_cot.pdf
```

### Step 3 — Decision quality (CoT models only)

Counterfactual IG of every question the seeker *considered*.

```bash
python scripts/reasoning_traces/generate_question_evaluations_csv.py  # -> outputs/question_evaluations_unified.csv
python scripts/reasoning_traces/summary_table.py                      # decision-quality table
```

<details><summary>Regenerating <code>seeker_traces.jsonl</code> / <code>question_evaluations.jsonl</code> from scratch (needs GPU)</summary>

`synthesize_traces.py` calls an LLM to structure each `<think>` block;
`evaluate_all_seeker_choices.py` re-simulates the real Oracle/Pruner (Qwen3-8B)
against the true game state for every candidate question, so it needs a live
vLLM endpoint.

```bash
python scripts/reasoning_traces/synthesize_traces.py            # -> outputs/seeker_traces.jsonl
python scripts/reasoning_traces/evaluate_all_seeker_choices.py  # -> question_evaluation.json per conversation
```
</details>

### Step 4 — Question classification

```bash
python scripts/question_classification/flatten_question_classifications.py # -> question_classifications.csv
python scripts/question_classification/analyze_question_classifications.py
python scripts/analysis/question_diversity_embeddings.py                   # sentence embeddings, no LLM
```

<details><summary>Regenerating <code>question_classifications.jsonl</code> from scratch (needs GPU)</summary>

```bash
python scripts/question_classification/classify_questions.py
```
</details>

### Step 5 — Belief-state analysis (optional)

Scores the belief state recovered from the seeker's CoT against the Pruner's
true active set Ω_t.

```bash
python scripts/reasoning_traces/analyze_belief_states.py
```

<details><summary>Regenerating <code>belief_states.jsonl</code> from scratch (needs GPU)</summary>

```bash
python scripts/reasoning_traces/extract_belief_states.py --all \
  --base-url http://<NODE_IP>:<PORT>/v1 --model google/gemma-4-31B-it
```
</details>

**Determinism.** Steps 1–4 run entirely on the released data. The optional
regeneration paths call an LLM, so they reproduce the artifacts but not
bit-identically.

## Repository structure

```
benchmark_runner.py        # main entrypoint
src/                       # core benchmark code (agents, orchestrator, entropy)
configs/full/              # experiment configs (one per model × domain × mode)
data/                      # candidate sets (geo, objects, diseases)
scripts/
  analysis/                # metrics, tables, figures
  reasoning_traces/        # CoT trace synthesis and EIG evaluation
  question_classification/ # question type annotation
```

## Citation

```bibtex
@article{...,
  title   = {...},
  author  = {...},
  journal = {...},
  year    = {2025},
}
```
