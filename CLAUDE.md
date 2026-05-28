# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DomainBenchmark is a Nextflow DSL2 pipeline for benchmarking domain-domain interaction (DDI) prediction methods. Built from the `nf-core/tools 4.0.2` template. For each database split it runs feature extraction → ML classifiers (RF, NN) → graph-based models (KGIDDI, DDIParsimony) → MultiQC evaluation, then aggregates across splits.

## Common commands

```bash
# full run across all database splits in the samplesheet
nextflow run . --input assets/samplesheet.csv -profile slurm,singularity -resume

# stub run (smoke test)
nextflow run . -profile test,singularity -stub-run

# single-database run via direct param
nextflow run . --input assets/samplesheet.csv -profile slurm,singularity --skip kgiddi,ddiparsimony

# lint
nf-core pipelines lint --dir .

# nf-test
nf-test test tests/default.nf.test
```

Samplesheet schema (`assets/schema_input.json`): array of `{id, db_path}` rows. `db_path` must be a directory containing `train.sqlite3`, `test.sqlite3`, `optimization.sqlite3`. Skip stages via `--skip aacomp,kgiddi` (comma-separated, matches feature or graph model names).

Python deps managed via conda — `environments/general.yml` (extraction/RF/graph/eval) and `environments/ml.yml` (PyTorch CU128 + cuML for NN training). No `pyproject.toml` / `requirements.txt`.

## Architecture

### Top-level layout
- `main.nf` — entry. Defines `DOMAINBENCHMARK` workflow (MultiQC + versions/methods boilerplate) and `DAISYBIO_DOMAINBENCHMARK` (the science workflow).
- `workflows/domainbenchmark.nf` — wires sample channel → `PER_DB_BENCHMARK` (scattered per DB) → `AGGREGATE_EVAL`.
- `subworkflows/local/per_db_benchmark/main.nf` — scatter: `DDI_EXTRACTION` → `FEATURE_EXTRACTION` (fan-out feature × split) → `NEURAL_NETWORK` + `RANDOM_FOREST` (per-feature singletons + one all-feature concat run, gated by `params.machine_learning_models`) + `GRAPH_MODEL` → `EVAL_ONE` (per-prediction) → `EVALUATION` (per-DB MultiQC reduce).
- `subworkflows/local/aggregate_eval/main.nf` — runs `COMBINE_EVAL` across per-DB reports to produce `results/evaluation/ddi_report.html`.
- `subworkflows/local/utils_nfcore_domainbenchmark_pipeline/main.nf` — nf-core boilerplate (initialise, completion, citations).
- `nextflow.config` — single source of truth for `db_list` (legacy), `graph_models`, `machine_learning_models`, `machine_learning_features`, `large_features`, `max_protein_combinations_per_ddi`, `skip`, `out_dir`, profiles.
- `conf/{base,slurm,test,test_full,modules}.config` — layered config. `conf/base.config` carries retry strategy and per-label resources.
- `assets/<ModelName>.json` — per-model hyperparameter grid + search config. Filename must match `model_name` and the Python script in `bin/`.
- `modules/local/<stage>/main.nf` — Nextflow process defs (`ddi_extraction`, `feature_extraction`, `neural_network`, `random_forest`, `graph_model`, `evaluation`).
- `bin/` — Python entrypoints invoked by modules (`run_models.py`, `random_forest.py`, `run_graph_models.py`, `kgiddi.py`, `ddiparsimony.py`, `extract_features.py`, `eval_one.py`, `eval_multiqc.py`, `combine_eval.py`, `load_data_gm.py`). Auto on `PATH` from Nextflow.
- `bin/features/` — feature encoders (`aacomp`, `aaencode`, `dummy`, `embeddings`, `protdcal`, `esm3_*`, `esmc_*`, `prott5_*`). New feature = new file here + entry in `params.machine_learning_features`. Heavy ones go in `params.large_features` → routed to `process_gpu_large`.
- `docker/`, `containers_{docker,singularity,conda_lock}_{amd64,arm64}.config` — container/lock matrices.

### Data flow
1. Input: samplesheet of `{id, db_path}`. Each `db_path` contains `train/test/optimization.sqlite3` (tables: DDI, DGO, PD, DomSeq, PPI, PGO, Embeddings).
2. `DDI_EXTRACTION` → SQL → CSV per split.
3. `FEATURE_EXTRACTION` (fan-out per feature × split) → per-feature `train/test/optimization.h5` under `results/<db>/data/<feature>/`.
4. `NEURAL_NETWORK` / `RANDOM_FOREST` consume `.h5`, grid-search via model JSON, predictions to `results/<db>/nn_output/` and `results/<db>/rf_output/`.
5. `GRAPH_MODEL` (KGIDDI, DDIParsimony, KGIDDI_RANDOM) runs independently against sqlite splits → `results/<db>/graph_models/<model>/`.
6. `EVAL_ONE` per-prediction → `EVALUATION` per-DB MultiQC → `results/<db>/evaluation/evaluation.html`.
7. `AGGREGATE_EVAL` / `COMBINE_EVAL` → `results/evaluation/ddi_report.html`.

The scatter design (`EVAL_ONE` → `EVALUATION` reduce) replaced a monolithic evaluation that hit 300 GB OOM. See comment in `modules/local/evaluation/main.nf`.

### Adding things
- **New ML model:** add `assets/<Name>.json` (must include `model_name`, `data`, `search_parameters`, `model_parameters`) + matching Python file in `bin/`. Picked up automatically.
- **New feature encoding:** add `bin/features/<name>.py` and append `<name>` to `params.machine_learning_features` in `nextflow.config`. Append to `params.large_features` if it needs GPU/big memory.
- **Skip stages:** `--skip aacomp,kgiddi` (comma-separated; matches feature or graph model names).

### Profiles
- `standard`: local executor, conda enabled.
- `slurm`: slurm executor, per-label resources via `conf/slurm.config`, singularity cache at `/nfs/scratch/singularity_cache`.
- `test` / `test_full`: minimal SQLite triplet under `tests/data/`, single feature.
- `daisybio`: site-specific defaults.

Default DB paths in `nextflow.config` point at `/nfs/data/CoBiNet_Masterpraktikum/databases/...` — override via samplesheet for local runs.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
