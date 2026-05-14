# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DomainBenchmark is a Nextflow pipeline for benchmarking domain-domain interaction (DDIs) methods with protein data. It runs ML classifiers and graph-based models across multiple database splits, then produces a combined MultiQC evaluation report.

- Root (`main.nf` / `wrapper.nf`) — training, graph models, evaluation.

## Common commands

```bash
# full run across all db splits in nextflow.config, then combined eval
bash wrapper.sh

# single-database run (results in results/<db_name>)
nextflow run main.nf

# combined evaluation only (after multiple main.nf runs)
nextflow run wrapper.nf --report_list <comma-sep dirs> --out_dir results

# profiles: standard (local, default), slurm
nextflow run main.nf -profile slurm -resume

```

`wrapper.sh` reads `params.db_list` and `params.out_dir` from `nextflow.config`, calls `main.nf` for each database, then runs `wrapper.nf` to combine reports. Supported CLI overrides: `-profile`, `-c`, `-resume`, `--skip`, `--out_dir`.

No test suite, no linter config, no `pyproject.toml` / `requirements.txt`. Python deps managed via conda (`fopra.yml` top-level, per-module `environment.yml` files).

## Architecture

### Top-level layout
- `main.nf` — orchestrates per-database workflow. Includes modules for feature extraction, ML training, random forest, graph models (KGIDDI, DDI parsimony), DDI extraction, data loading, evaluation. Parses model JSON configs from `assets/` at runtime.
- `wrapper.nf` / `wrapper.sh` — iterate database splits, then aggregate evaluation across them.
- `nextflow.config` — single source of truth for `db_list`, `graph_models`, `machine_learning_features`, `skip`, `out_dir`, and executor profiles.
- `assets/<ModelName>.json` — per-model hyperparameter grid and search config. Filename **must** match `model_name` field and the Python script in `bin/`.
- `modules/local/<stage>/main.nf` — Nextflow process definitions. Each stage may ship its own `environment.yml`.
- `bin/` — Python scripts invoked by modules (`run_models.py`, `random_forest.py`, `run_graph_models.py`, `kgiddi.py`, `ddiparsimony.py`, `extract_features.py`, `eval_multiqc.py`, `combine_eval.py`, `load_data_gm.py`, etc.). Must be executable and on `PATH` (Nextflow handles this from `bin/`).
- `bin/features/` — feature encoding implementations (`aacomp`, `aaencode`, `protdcal`, `embeddings`, `esm3_*`, `esmc_*`, `prott5_*`). New feature = new file here + entry in `params.machine_learning_features`.
- `environments/general.yml`, `fopra.yml`, `tower.yml` — conda / Tower configs.
- `docker/` — container definitions.

### Data flow
1. Input: database split directory with `train.sqlite3`, `test.sqlite3`, `optimization.sqlite3` (tables: DDI, DGO, PD, DomSeq, PPI, PGO, Embeddings).
2. `feature_extraction` → writes per-feature `train/test/optimization.h5` under `results/<db>/data/<feature>/`.
3. `machine_learning` / `random_forest` consume `.h5` features, grid-search via the model JSON, emit predictions to `results/<db>/ml_output/`.
4. `graph_model` stages (KGIDDI, DDI parsimony) run independently against the sqlite splits, output under `results/<db>/graph_models/<model>/`.
5. `evaluation` (MultiQC) combines everything into `results/<db>/evaluation/evaluation.html`; `wrapper.nf` merges across DBs into `results/evaluation/ddi_report.html`.

### Adding things
- **New ML model:** add `assets/<Name>.json` (must include `model_name`, `data`, `search_parameters`, `model_parameters`) and matching logic in the ML module. Name is auto-picked up by `main.nf`.
- **New feature encoding:** add `bin/features/<name>.py` and append `<name>` to `params.machine_learning_features` in `nextflow.config`.
- **Skip stages:** set `--skip aacomp,kgiddi` (comma-sep, matches feature or graph model names).

### Profiles
- `standard`: local executor, conda enabled.
- `slurm`: slurm executor, 8 cpus / 200 GB / 48h per process, singularity cache at `/nfs/scratch/singularity_cache`.

Default paths in `nextflow.config` point at `/nfs/data/CoBiNet_Masterpraktikum/databases/...` — override with `--db` / `--db_list` for local runs.

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
