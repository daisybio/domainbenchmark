# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DomainBenchmark is a Nextflow DSL2 pipeline for benchmarking domain-domain interaction (DDI) prediction methods. Built from the `nf-core/tools 4.0.2` template. For each database it runs feature extraction → ML classifiers (RF, NN) → graph-based models (KGIDDI, DDIParsimony) → MultiQC evaluation, then aggregates across databases. A database with an internal test set ships `test_balanced` + `test_realistic`; both are scored by the same trained models and reported as separate datasets.

## Common commands

```bash
# full run across every database in the samplesheet (cluster: executor/queues/
# workDir/GPU clusterOptions all come from the -c config, not from this repo)
nextflow run . --input assets/samplesheet.csv -c daisybio.config -profile apptainer,gpu,keep_work -resume

# same, straight off a domainsplit output directory
nextflow run . --input /path/to/domainsplit/results/databases -c daisybio.config -profile apptainer,gpu,keep_work -resume

# stub run (smoke test)
nextflow run . -profile test,singularity -stub-run

# single-database run via direct param
nextflow run . --input assets/samplesheet.csv -c daisybio.config -profile apptainer,gpu,keep_work --skip kgiddi,ddiparsimony

# lint
nf-core pipelines lint --dir .

# nf-test
nf-test test tests/default.nf.test
```

`--input` takes either a samplesheet CSV or a directory. Samplesheet schema (`assets/schema_input.json`): array of `{id, db_path}` rows. Directory form: every immediate subdirectory holding a `train.sqlite3` is a dataset named after the directory (what domainsplit publishes under `databases/`). Either way each database directory must contain `train.sqlite3`, `validation.sqlite3`, and one or more `test*.sqlite3`. Skip stages via `--skip aacomp,kgiddi` (comma-separated, matches feature or graph model names).

Python deps managed via conda — `environments/general.yml` (extraction/RF/graph/eval) and `environments/ml.yml` (PyTorch CU128 + cuML for NN training). No `pyproject.toml` / `requirements.txt`.

## Architecture

### Top-level layout
- `main.nf` — entry. `PIPELINE_INITIALISATION` → `DAISYBIO_DOMAINBENCHMARK` (the science workflow: `PER_DB_BENCHMARK` scattered per DB → `AGGREGATE_EVAL`) → `PIPELINE_COMPLETION`. There is no pipeline-level MULTIQC process and no software-version collection: the report is built by `bin/eval_multiqc.py` / `bin/combine_eval.py`, which write their own MultiQC configs.
- `subworkflows/local/per_db_benchmark/main.nf` — scatter: `DDI_EXTRACTION` → `FEATURE_EXTRACTION` (fan-out feature × split) + `VERIFY_EMBEDDINGS` (per db, for `params.published_features`) → `NEURAL_NETWORK` + `RANDOM_FOREST` (per-feature singletons + one all-feature concat run, gated by `params.machine_learning_models`) + `GRAPH_MODEL` → `EVAL_ONE` (per-prediction) → `EVALUATION` (per (db, test variant) MultiQC reduce). `runLabel()` there defines the `<db>_<variant>` run name used downstream.
- `subworkflows/local/aggregate_eval/main.nf` — runs `COMBINE_EVAL` across per-DB reports to produce `results/evaluation/ddi_report.html`.
- `subworkflows/local/utils_nfcore_domainbenchmark_pipeline/main.nf` — nf-core boilerplate (initialise, completion) plus `discoverSplits()` / `datasetTuple()`, which turn a database directory into `meta.splits` / `meta.tests`.
- `nextflow.config` — single source of truth for `db_list` (legacy), `graph_models`, `machine_learning_models`, `machine_learning_features`, `published_features`, `large_features`, `embeddings`, `domainsplit_run`, `min_embedding_coverage`, `skip`, `out_dir`, profiles.
- `conf/{base,test,test_full,modules}.config` — layered config. `conf/base.config` carries retry strategy and per-label resources. There is deliberately no executor profile: executor, queue, `workDir` and GPU `clusterOptions` come from the institutional config passed with `-c` (e.g. `daisybio.config`), which keys on the `process_gpu` label. `conf/base.config` therefore must never mention a queue. NN/RF memory/time are sized per task from `params.large_features` in `conf/modules.config` (`withName` beats `withLabel`), which is what replaced the old `process_gpu_small`/`process_gpu_large` labels.
- `assets/<ModelName>.json` — per-model hyperparameter grid + search config. Filename must match `model_name` and the Python script in `bin/`.
- `modules/local/<stage>/main.nf` — Nextflow process defs (`ddi_extraction`, `feature_extraction`, `verify_embeddings`, `neural_network`, `random_forest`, `graph_model`, `evaluation`).
- `bin/` — Python entrypoints invoked by modules (`run_models.py`, `random_forest.py`, `run_graph_models.py`, `kgiddi.py`, `ddiparsimony.py`, `extract_features.py`, `verify_embeddings.py`, `eval_one.py`, `eval_multiqc.py`, `combine_eval.py`, `load_data_gm.py`). Auto on `PATH` from Nextflow.
- `bin/determinism.py` — `seed_everything(seed)` (seeds `random`/`numpy`/`torch`, kills cuDNN autotuning, asks torch for deterministic kernels) and `derive_seed(seed, *tokens)` (stable child seed for a worker, keyed on its own identity rather than completion order). Every entrypoint calls it. `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG` are set in the `env` scope of `nextflow.config` — they must exist before the interpreter starts.
- `bin/features/` — feature encoders (`aacomp`, `aaencode`, `dummy`, `embeddings`, `protdcal`). New *extracted* feature = new file here + entry in `params.machine_learning_features`; key the HDF5 by instance via `embeddings.INSTANCE_KEY_SQL` / `embeddings.write_instance()`. Heavy ones go in `params.large_features`. There are no embedding encoders any more: domainsplit stopped storing BLOBs in SQLite, so `protein.*_per_residue` and `domain_protein_map.*_per_domain` are gone from the schema and the `esm3_*`/`esmc_*`/`prott5_*` extractors that read them were deleted with the columns.
- `docker/` — image definitions. One image per environment, referenced directly in each module's `container` directive.

### Data flow
1. Input: samplesheet of `{id, db_path}` or a `databases/` directory. Each database dir contains `train.sqlite3`, `validation.sqlite3`, and one or more `test*.sqlite3` (tables: `domain`, `domain_go_terms`, `domain_domain_interaction`, `protein`, `protein_go_terms`, `protein_protein_interaction`, `domain_protein_map`, `ddi_split_membership`).
2. `DDI_EXTRACTION` → SQL → `<split>.csv` plus `<split>_instances.csv` (the domain-instance pairs `ddi_split_membership` assigns to that split — what the ML loader instantiates, instead of a cross-product).
3. `FEATURE_EXTRACTION` (fan-out per feature × split) → one `<feature>__<split>.h5` per task under `results/<db>/data/`. The flat name carries the layout; NN/RF stage the whole set into `features/` and `machine_learning.py:resolve_feature_file` resolves it. HDF5 layout is `h5[domain_id][instance_key]` — instance-level, because one protein can carry two instances of the same family. There is deliberately no staging process in between: the old `STAGE_FEATURE_DIR` cost a scheduled job per (db, feature) to run `cp`, and on a node that cannot loop-mount the SIF, singularity unpacked the whole image first and blew through the label's walltime (exit 140). (`resolve_feature_file` used to accept a `features/<feature>/<split>.h5` tree as well; that branch went when the process that built it did.)
3b. `VERIFY_EMBEDDINGS` (per db) for every `params.published_features` entry. These are not extracted at all: domainsplit publishes one `<model>_domain_embeddings.h5` per **run** under `--embeddings`, and the process gates it against the split databases before renaming it to `features/<feature>.h5` for `resolve_feature_file`'s second layout. See the reproducibility note below for why the gate exists.
4. `NEURAL_NETWORK` / `RANDOM_FOREST` consume `.h5`, grid-search via model JSON on train+validation, then score every test split → `predictions_<variant>.parquet` under `results/<db>/nn_output/` and `results/<db>/rf_output/`.
5. `GRAPH_MODEL` (KGIDDI, DDIParsimony, KGIDDI_RANDOM) trains once on the train split and scores every test split → `results/<db>/graph_models/<model>/predictions_<variant>.parquet`.
6. `EVAL_ONE` per-prediction → `EVALUATION` per (db, test variant) MultiQC → `results/<db>/evaluation/<variant>/`.
7. `AGGREGATE_EVAL` / `COMBINE_EVAL` → `results/evaluation/ddi_report.html`, with each (db, variant) as its own dataset entry.

The scatter design (`EVAL_ONE` → `EVALUATION` reduce) replaced a monolithic evaluation that hit 300 GB OOM. See comment in `modules/local/evaluation/main.nf`.

### Published embeddings and the cross-run hazard

`domain.id` is a **surrogate integer**. domainsplit's `SUBSET_SPLIT_DB` copies it verbatim and `PRUNE_UNREPRESENTED_DDIS` deletes without renumbering, so a published `<model>_domain_embeddings.h5` is valid across every split database *of the same run* and silently wrong across runs — the ids still exist, they just name different domains.

Nothing raises when that happens. `bin/features/embeddings.py` keys on `COALESCE(instance_id, 'r' || rowid)`, and `load_embedding_data` **skips** every pair and instance combination it cannot resolve, so a drifted key layout produces zero training rows, which looks exactly like a database that holds no data. Assert the join resolves; never assume it. Two places do:

- `bin/verify_embeddings.py` (run by `VERIFY_EMBEDDINGS`, before any model trains) counts how many of each split database's `(domain_id, instance_key)` pairs the HDF5 actually carries and fails under `params.min_embedding_coverage`. A same-run file resolves ~100%, a foreign one ~0%. It also enforces `domainsplit_run` agreement between files, and against `params.domainsplit_run` when set. Because NN/RF consume its output, an unverified file cannot reach a GPU.
- `machine_learning.load_embedding_data` tallies per-feature hits and raises naming the feature when one resolves no domain pair at all, or when every domain pair resolves but no instance pair does.

The `domainsplit_run` root attribute exists only in the HDF5; the split databases carry no run marker, so the structural check is the load-bearing one and the declared id is a cheap extra.

### Reproducibility

The pipeline is bit-reproducible for a given commit + input + `--seed`, and `nf-test` asserts it by snapshotting model and prediction contents. Anything that draws from an RNG must take a `--seed`, call `determinism.seed_everything`, and use `derive_seed` for work inside a process pool. Two traps worth knowing:

- Never iterate a `set`/`dict` of strings where the order affects output, even with `PYTHONHASHSEED` fixed — sort it. The fixed salt makes it reproducible, but the sort makes it obvious.
- joblib/`ProcessPoolExecutor` workers inherit no RNG state. Reseed inside the worker (skorch's `on_train_begin`, or a `seed` argument), and never key a result's position on completion order — `compute_random_x_matrix_parallel` used to number rows by `as_completed`. skorch is worse than it looks: `initialize()` draws the weights *before* `on_train_begin` fires, so the callback alone is not enough (hence `SeededNeuralNetBinaryClassifier`).
- A model can be bit-identical while something derived from it is not. sklearn's `RandomForest.predict_proba` parallelises the per-tree sum, so with `n_jobs=-1` the `.pkl` matched run to run but the MCC-tuned threshold in `model_parameters.json` did not. Fitting is deterministic; prediction is where the reduction order leaks. The CPU path is therefore `n_jobs=1`.
- **Nextflow's own ordering is a reproducibility surface.** `groupTuple()` and `collect()` emit in task-completion order. If that list becomes a command-line argument order, it reaches the output: MultiQC writes its JSON in the order it was fed, so `--per_model_metrics` had to be sorted. Sort every grouped list before it reaches a script — it also stabilises the task hash for `-resume`.

Do not re-add entries to `tests/.nftignore` to make a snapshot pass: that hides real regressions. Regenerate the snapshot instead, and only for a change you can explain.

### Adding things
- **New ML model:** add `assets/<Name>.json` (must include `model_name`, `data`, `search_parameters`, `model_parameters`) + matching Python file in `bin/`. Picked up automatically.
- **New feature encoding:** add `bin/features/<name>.py` and append `<name>` to `params.machine_learning_features` in `nextflow.config`. Append to `params.large_features` if it needs GPU/big memory.
- **New published feature:** no Python file. Append `<name>` to `params.machine_learning_features` *and* `params.published_features`, and drop `<name>.h5` (or `<model>_domain_embeddings.h5`) into `--embeddings`.
- **Skip stages:** `--skip aacomp,kgiddi` (comma-separated; matches feature or graph model names).

### Profiles
- `standard`: local executor, conda enabled.
- `apptainer` / `singularity` / `docker`: container engine, with a 2 h pull timeout (the GPU image is large).
- `gpu`: adds `--nv` / `--gpus all`; on a cluster it is also the profile the institutional config keys on to send `process_gpu` tasks to the GPU queue.
- `test` / `test_full`: minimal SQLite triplet under `tests/data/`, single feature.
- `daisybio`: site-specific defaults from nf-core/configs. `daisybio.config` sets `cleanup = true`, so pair it with `keep_work` if you want `-resume` to work.

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
