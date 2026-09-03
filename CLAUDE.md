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
- `main.nf` — a typed `params { }` declaration block, then the entry. `PIPELINE_INITIALISATION` → `DAISYBIO_DOMAINBENCHMARK` (the science workflow: `PER_DB_BENCHMARK` scattered per DB → `AGGREGATE_EVAL`) → `PIPELINE_COMPLETION`. There is no pipeline-level MULTIQC process and no software-version collection: the report is built by `bin/eval_multiqc.py` / `bin/combine_eval.py`, which write their own MultiQC configs.
- `subworkflows/local/per_db_benchmark/main.nf` — scatter: `DDI_EXTRACTION` → `FEATURE_EXTRACTION` (fan-out feature × split) + `VERIFY_EMBEDDINGS` (per db, for `params.published_features`) → `NEURAL_NETWORK` + `RANDOM_FOREST` (per-feature singletons + one all-feature concat run, gated by `params.machine_learning_models`) + `GRAPH_MODEL` → `EVAL_ONE` (per-prediction) → `EVALUATION` (per (db, test variant) MultiQC reduce). `runLabel()` there defines the `<db>_<variant>` run name used downstream.
- `subworkflows/local/aggregate_eval/main.nf` — runs `COMBINE_EVAL` across per-DB reports to produce `results/evaluation/ddi_report.html`.
- `subworkflows/local/utils_nfcore_domainbenchmark_pipeline/main.nf` — nf-core boilerplate (initialise, completion) plus `discoverSplits()` / `datasetTuple()`, which turn a database directory into `meta.splits` / `meta.tests`.
- `nextflow.config` — single source of truth for `db_list` (legacy), `graph_models`, `machine_learning_models`, `machine_learning_features`, `published_features`, `large_features`, `embeddings`, `min_embedding_coverage`, `ppi_score_cutoff`, `mqc_order`, `skip`, `out_dir`, profiles.
- `conf/{base,test,test_full,modules}.config` — layered config. `conf/base.config` carries retry strategy and per-label resources. There is deliberately no executor profile: executor, queue, `workDir` and GPU `clusterOptions` come from the institutional config passed with `-c` (e.g. `daisybio.config`), which keys on the `process_gpu` label. `conf/base.config` therefore must never mention a queue. NN/RF memory/time are sized per task from `params.large_features` in `conf/modules.config` (`withName` beats `withLabel`), which is what replaced the old `process_gpu_small`/`process_gpu_large` labels. Graph models carry `process_graph` (12 cpus, 160 GB × attempt, 24 h). That label sat dead for a while — it was defined here while `modules/local/graph_model/main.nf` still declared `process_high`, so every graph model silently ran at 72 GB and a cluster run lost five kgiddi/kgiddi_random tasks to exit 137 and two ddiparsimony tasks to a `BrokenProcessPool`. Its cpu count is deliberately not raised: both graph models fan `--threads task.cpus` into a `ProcessPoolExecutor`, so each extra cpu is another worker holding its own copy of the interactome, and memory demand rises with cpus.
- `assets/<ModelName>.json` — per-model hyperparameter grid + search config. Filename must match `model_name` and the Python script in `bin/`.
- `modules/local/<stage>/main.nf` — Nextflow process defs (`ddi_extraction`, `feature_extraction`, `verify_embeddings`, `neural_network`, `random_forest`, `graph_model`, `evaluation`).
- `bin/` — Python entrypoints invoked by modules (`run_models.py`, `random_forest.py`, `run_graph_models.py`, `kgiddi.py`, `ddiparsimony.py`, `extract_features.py`, `verify_embeddings.py`, `eval_one.py`, `eval_multiqc.py`, `combine_eval.py`, `load_data_gm.py`). Auto on `PATH` from Nextflow.
- `bin/determinism.py` — `seed_everything(seed)` (seeds `random`/`numpy`/`torch`, kills cuDNN autotuning, asks torch for deterministic kernels) and `derive_seed(seed, *tokens)` (stable child seed for a worker, keyed on its own identity rather than completion order). Every entrypoint calls it. `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG` are set in the `env` scope of `nextflow.config` — they must exist before the interpreter starts.
- `bin/features/` — feature encoders (`aacomp`, `aaencode`, `dummy`, `embeddings`, `protdcal`). Every encoder is `extract_features(conn, out_file, seed)` — the seed is part of the ABI whether or not the encoder draws from it, so a sampling encoder cannot be added without one. New *extracted* feature = new file here + entry in `params.machine_learning_features`; key the HDF5 group by Pfam accession via `embeddings.DOMAIN_KEY_SQL` + `embeddings.DOMAIN_JOIN_SQL` and the dataset by instance via `embeddings.INSTANCE_KEY_SQL` / `embeddings.write_instance()`. Heavy ones go in `params.large_features`. There are no embedding encoders any more: domainsplit stopped storing BLOBs in SQLite, so `protein.*_per_residue` and `domain_protein_map.*_per_domain` are gone from the schema and the `esm3_*`/`esmc_*`/`prott5_*` extractors that read them were deleted with the columns.
- `docker/` — image definitions. One image per environment, referenced directly in each module's `container` directive.

### Data flow
1. Input: samplesheet of `{id, db_path}` or a `databases/` directory. Each database dir contains `train.sqlite3`, `validation.sqlite3`, and one or more `test*.sqlite3` (tables: `domain`, `domain_go_terms`, `domain_domain_interaction`, `protein`, `protein_go_terms`, `protein_protein_interaction`, `domain_protein_map`, `ddi_split_membership`).
2. `DDI_EXTRACTION` → SQL → `<split>.csv`, `<split>_sources.csv` and `<split>_instances.csv` (the domain-instance pairs `ddi_split_membership` assigns to that split — what the ML loader instantiates, instead of a cross-product). Every one of them joins `domain` and reports **`pfam_id`**, never `domain.id`; see "The domain key" below.
3. `FEATURE_EXTRACTION` (fan-out per feature × split) → one `<feature>__<split>.h5` per task under `results/<db>/data/`. The flat name carries the layout; NN/RF stage the whole set into `features/` and `machine_learning.py:resolve_feature_file` resolves it. HDF5 layout is `h5[pfam_id][instance_key]` — keyed by Pfam accession (see "The domain key") and instance-level, because one protein can carry two instances of the same family. There is deliberately no staging process in between: the old `STAGE_FEATURE_DIR` cost a scheduled job per (db, feature) to run `cp`, and on a node that cannot loop-mount the SIF, singularity unpacked the whole image first and blew through the label's walltime (exit 140). (`resolve_feature_file` used to accept a `features/<feature>/<split>.h5` tree as well; that branch went when the process that built it did.)
3b. `VERIFY_EMBEDDINGS` (per db) for every `params.published_features` entry. These are not extracted at all: domainsplit publishes one `<model>_domain_embeddings.h5` per **run** under `--embeddings`, and the process gates it against the split databases before renaming it to `features/<feature>.h5` for `resolve_feature_file`'s second layout. See the reproducibility note below for why the gate exists.
4. `NEURAL_NETWORK` / `RANDOM_FOREST` consume `.h5`, grid-search via model JSON on train+validation, then score every test split → `predictions_<variant>.parquet` under `results/<db>/nn_output/` and `results/<db>/rf_output/`.
5. `GRAPH_MODEL` (KGIDDI, DDIParsimony, KGIDDI_RANDOM) trains once on the train split and scores every test split → `results/<db>/graph_models/<model>/predictions_<variant>.parquet`, pair-canonicalised through `load_data_gm.canonical_pair` like every other prediction file. All three build their interactome from PPIs at or above `params.ppi_score_cutoff` (STRING `combined_score`, default 400 = STRING's "medium confidence"). It is one pipeline-level param, not a per-model JSON key, because the models are compared against each other and must see the same network; `run_graph_models.py --ppi_score_cutoff` overrides a model JSON's own value, and `DEFAULT_PPI_SCORE_CUTOFF` in `bin/load_data_gm.py` is the last resort.

   DDIParsimony's LP relaxation (`ddiparsimony_functions.compute_lp_score`) poses its constraint matrix **sparse**. It used to build one dense `np.zeros(len(domain_pairs))` row per kept PPI and stack them — an `(n_constraints × n_domain_pairs)` float64 matrix whose rows hold at most `|domains(p1)| × |domains(p2)|` nonzeros. Measured at 8k constraints × 20k pairs that is 1.2 GiB of matrix and 2.5 GiB of process peak, versus 0.6 MiB / 62 MiB sparse — and `compute_random_x_matrix_parallel` has every pool worker build one concurrently, which is what OOM-killed the workers and surfaced as `BrokenProcessPool`. HiGHS converts its input to sparse CSC internally either way, so the LP is identical; verified bitwise-equal `res.fun` and `res.x` against the dense form. That pool also runs under **spawn** with an `initializer`, so the adjacency matrix and domain maps are pickled once per worker instead of once per each of the 1000 submissions, and `random_x_matrix` is float32 — the workers only ever returned float32. Raising it is not free: a split database holds only its own proteins' interactome, so at 900 `minimal_leakage/test_balanced` kept 2 of 254 PPIs, which collapsed KGIDDI's union-find into one cluster and left chi2 with no outside stratum to contrast against. `load_data_gm.load_ppi` **coerces the score to numeric and fails if any row will not coerce** (a NULL included). It was not numeric in the databases of 2026-09-02: domainsplit declared `protein_protein_interaction.score` with no type (BLOB affinity, stores whatever class it is handed) and its `insert_ppi.py` bound the raw string it split out of STRING's links file, so pandas read an `object` column and `score >= cutoff` raised `'>=' not supported between instances of 'str' and 'int'`. domainsplit types the column `REAL` and parses the score now, so a fresh database needs nothing from the check — it stays for databases already on disk, which do not get retyped. Failing rather than dropping is deliberate: an unparseable score cannot pass any cutoff, so tolerating it would score a quietly smaller interactome.
6. `EVAL_ONE` per-prediction → `EVALUATION` per (db, test variant) MultiQC → `results/<db>/evaluation/<variant>/`. `EVAL_ONE` **fails** when a scored pair has no row in `<split>_sources.csv`: both sides come from the same `domain_domain_interaction` and the orientation is canonicalised, so a miss is a key-space mismatch and a report built on it is meaningless.
7. `AGGREGATE_EVAL` / `COMBINE_EVAL` → `results/evaluation/ddi_report.html`, with each (db, variant) as its own dataset entry.

The scatter design (`EVAL_ONE` → `EVALUATION` reduce) replaced a monolithic evaluation that hit 300 GB OOM. See comment in `modules/local/evaluation/main.nf`.

### The domain key

Every domain is named by its **Pfam accession** (`domain.pfam_id`) throughout: the DDI CSVs, the feature HDF5 group names, the `domain_a`/`domain_b` columns of every `predictions_*.parquet`, and `eval_one.py`'s per-source join.

Every `predictions_*.parquet` is also **canonically ordered** — `domain_a <= domain_b`, one row per DDI. A DDI is undirected, so the emitted orientation should not depend on which one a model's internal iteration produced. `bin/load_data_gm.canonical_pair` is the rule; `machine_learning._aggregate_to_ddi_level` applies the same comparison inline. Plain string comparison is correct here only because a Pfam accession is `PF` + a zero-padded five-digit number, so lexicographic and numeric order coincide — it would not hold for a bare integer id.

On the ML side that canonicalisation also **merges the two orientations**. `load_embedding_data` adds both `(A, B)` and `(B, A)` to `labeled_domain_pairs`, deliberately: the feature vector is `concat(emb_a, emb_b)`, so without it the model learns an order-dependent function of an undirected relation. Grouping on the raw orientation carried that augmentation into the output — two rows per DDI, so every metric counted each one twice and `n_scored` came out at twice the split's `n`. Merging makes the prediction the mean of the two orientations, i.e. symmetric, and leaves one row per DDI.

Not `domain.id`. That is a **per-run surrogate integer**: domainsplit's `SUBSET_SPLIT_DB` copies it verbatim and `PRUNE_UNREPRESENTED_DDIS` deletes without renumbering, so the same integer names a different domain in the next run and a report keyed on it cannot be compared between runs — which is the entire point of the report. `domain` is `UNIQUE(pfam_id)` with `id INTEGER PRIMARY KEY`, so the two are in bijection *within* a database and the swap loses nothing. `DDI_EXTRACTION` fails if any `pfam_id` is NULL, because that would merge unrelated domains into one row rather than error.

This closed a total join failure. The graph models never went through the CSVs — `bin/load_data_gm.py` reads the split database directly and has always joined `domain` for `pfam_id` — while `DDI_EXTRACTION` emitted surrogate ids, so *every* graph-model pair missed `eval_one.py`'s join against `<split>_sources.csv` and the report showed one quiet `unknown` row holding the whole test set. A non-empty unmatched set is now fatal there, naming examples; the surviving `unknown` row means only NULL provenance in the database.

### Published embeddings

domainsplit publishes one `<model>_domain_embeddings.h5` per **run** under `--embeddings`, keyed `h5[pfam_id][instance_id]`. It is per-run, not per-split: it holds every domain the run saw and each split database is a subset of that, so one file serves train, validation and every test split. `--embeddings` may be omitted when `--input` is a directory — domainsplit publishes `embeddings/` as a sibling of `databases/`, so the pipeline derives it one level up and only errors if that directory does not exist (`embeddingsDir()` in the subworkflow). A samplesheet input still requires it explicitly: its rows can point at several runs.

Nothing raises when a file does not fit. `machine_learning.load_embedding_data` **skips** every pair and instance combination it cannot resolve, so an ill-fitting file produces zero training rows, which looks exactly like a database that holds no data. Assert the join resolves; never assume it. Two places do:

- `bin/verify_embeddings.py` (run by `VERIFY_EMBEDDINGS`, before any model trains) counts how many of each split database's `(pfam_id, instance_key)` pairs the HDF5 carries and fails under `params.min_embedding_coverage`, and rejects a file still declaring the retired `key_layout = {domain_id}/{instance_id}`. Because NN/RF consume its output, an unverified file cannot reach a GPU.
- `machine_learning.load_embedding_data` tallies per-feature hits and raises naming the feature when one resolves no domain pair at all, or when every domain pair resolves but no instance pair does.

Note what the coverage check cannot do — this inverted when the key changed. Pfam accessions are **stable across runs**, so it catches a wrong or stale *export* (wrong dataset, made before domains were added, truncated) but **not a foreign run** over the same domain universe: that resolves ~100%. There is deliberately no guard for it. `params.domainsplit_run` used to be one and was removed: with `--embeddings` derived from `--input`'s own directory the right pairing happens by construction, and for a samplesheet run — or an explicit `--embeddings` — pointing at the matching run is the caller's job. Getting it wrong shows up as a coverage failure only if the two runs also disagree on domains.

### Numeric params on the command line

`main.nf` opens with a typed `params { }` block declaring `seed: Integer`, `ppi_score_cutoff: Integer`, `min_embedding_coverage: Float`, `allow_cpu_ml: Boolean` — **types only**; the defaults stay in `nextflow.config`.

It is there because Nextflow's v2 (strict) parser, the default since 26.x, stopped inferring types for command-line parameters. `--seed 7` arrives as the String `"7"` and nf-schema rejects it against `"type": "integer"` with `Value is [string] but should be [integer]`. A typed declaration makes Nextflow coerce before validation. Add any new numeric or boolean param here, or it cannot be overridden from the CLI.

Two things that look like the fix and are not, both tested:

- `validation.lenientMode = true` only widens the *other* direction — it lets an integer satisfy a `string` type. The option is read (`Set 'validation.lenientMode' to true` in the log) and the run still fails.
- `NXF_SYNTAX_PARSER=v1` does restore type inference, by reverting to the retired parser. Not a fix, a postponement.

Use `Float`, not `Double`/`BigDecimal`/`Number`: those reject the String outright (`Parameter 'f' with type Double cannot be assigned to 0.8 [String]`) instead of coercing it. A `-params-file` (YAML/JSON) preserves types on its own and never needed any of this.

**Only the entry script's declarations count.** A `params { }` block inside an included script is ignored, so a pipeline that embeds this one as a subworkflow would have to repeat the declarations in its own `main.nf`. The same fix is in `daisybio/domainsplit` and `ppi-splitting-pipeline`; domainsplit has to declare the ppi-splitting params it inherits through `includeConfig` for exactly this reason.

### Reproducibility

The pipeline is bit-reproducible for a given commit + input + `--seed`, and `nf-test` asserts it by snapshotting model and prediction contents. Anything that draws from an RNG must take a `--seed`, call `determinism.seed_everything`, and use `derive_seed` for work inside a process pool. Two traps worth knowing:

- Never iterate a `set`/`dict` of strings where the order affects output, even with `PYTHONHASHSEED` fixed — sort it. The fixed salt makes it reproducible, but the sort makes it obvious.
- joblib/`ProcessPoolExecutor` workers inherit no RNG state. Reseed inside the worker (skorch's `on_train_begin`, or a `seed` argument), and never key a result's position on completion order — `compute_random_x_matrix_parallel` used to number rows by `as_completed`. skorch is worse than it looks: `initialize()` draws the weights *before* `on_train_begin` fires, so the callback alone is not enough (hence `SeededNeuralNetBinaryClassifier`).
- **A function that draws must take the seed, not reach for the global RNG.** `compute_lp_score` drew its reliability mask from `np.random.rand()` while running in a pool, and was reproducible only by accident: under fork every worker inherited one shared RNG state, and the 6-point reliability grid happened to be smaller than the worker count, so each worker ran at most one task. `seed` and `rng_token` are now keyword-only *and* have no defaults, so a caller that forgets raises `TypeError` instead of quietly randomising a benchmark. `rng_token` names what is being scored (`train_grid:<r>`, `test_eval:<variant>:<r>`, `random_x:<i>`): two calls meant to reproduce each other pass the same token — which is how the `avg_x_vec_*.npy` written beside `pw_scores_*.json` now actually corresponds to those scores, where before the rerun drew a fresh mask and saved a vector the scores did not come from.
- **Deriving per item beats drawing in a loop.** The `dummy` encoder drew from the seeded global RNG once per row, so each vector's value depended on its *position* in a query with no `ORDER BY`. It now derives `derive_seed(seed, "dummy", domain_key, instance_key)` per instance: same vector for the same pair regardless of row order or how many rows precede it. Same rule for any future sampling encoder.
- **One RNG parameter, one spelling, and it has to be *wired*.** `params.seed` (default 42, typed `Integer` in `main.nf`) is the only seed the pipeline has, and every entrypoint takes it as `--seed`. `eval_one.py` used to call its own `--bootstrap_seed`, defaulted to 42, and no module passed it — so `--seed` never reached the bootstrap CIs. Renamed and wired. A second seed parameter, or a flag spelled differently, is the bug.
- A model can be bit-identical while something derived from it is not. sklearn's `RandomForest.predict_proba` parallelises the per-tree sum, so with `n_jobs=-1` the `.pkl` matched run to run but the MCC-tuned threshold in `model_parameters.json` did not. Fitting is deterministic; prediction is where the reduction order leaks. The CPU path is therefore `n_jobs=1`.
- **Nextflow's own ordering is a reproducibility surface.** `groupTuple()` and `collect()` emit in task-completion order. If that list becomes a command-line argument order, it reaches the output: MultiQC writes its JSON in the order it was fed, so `--per_model_metrics` had to be sorted. Sort every grouped list before it reaches a script — it also stabilises the task hash for `-resume`.

Do not re-add entries to `tests/.nftignore` to make a snapshot pass: that hides real regressions. Regenerate the snapshot instead, and only for a change you can explain.

### Report ordering

`params.mqc_order` is a comma-separated list of dataset (run label) names — `random_balanced,minimal_leakage_hcni_realistic,external_test`. A *dataset* is one (database, test variant), the same label `runLabel()` builds. It drives three things, all via the helpers in `bin/eval_multiqc_functions.py` (`parse_dataset_order`, `resolve_dataset_order`, `sort_by_dataset`, `order_dataset_keys`):

- the MultiQC section order of the blocks that exist once per dataset (`db_database_analysis_<ds>`, `model_eval_metrics_<ds>`);
- the tab/series order inside blocks that carry every dataset (combined ROC/PR line graphs, the by-source bar graph);
- the column order of the AUC/AP heatmaps and their CI tables.

A requested name that exists in no dataset is a **hard failure**, raised in `combine_eval.py` — the only stage that sees every dataset. `eval_multiqc.py` orders but does not validate, because a per-dataset report legitimately holds only one of the names. A dataset in the data but missing from the list **warns** and is appended alphabetically after the ordered ones; dropping it silently would hide a whole database. Left unset, everything is alphabetical (never input order — the report has to be a function of the names alone to stay reproducible).

The degree, betweenness and clustering-coefficient box plots are gone, in both the per-dataset and combined reports. Betweenness and clustering were never computed (both returned a hardcoded `[1..10]`, the real `nx` calls commented out because they do not finish on a PPI graph this size), and the degree distribution answered a question the node/edge counts in `database_analysis` already cover. `analyse_interaction_network` now returns node and edge counts only.

### Adding things
- **New ML model:** add `assets/<Name>.json` (must include `model_name`, `data`, `search_parameters`, `model_parameters`) + matching Python file in `bin/`. Picked up automatically.
- **New feature encoding:** add `bin/features/<name>.py` and append `<name>` to `params.machine_learning_features` in `nextflow.config`. Append to `params.large_features` if it needs GPU/big memory.
- **New published feature:** no Python file. Append `<name>` to `params.machine_learning_features` *and* `params.published_features`, and drop `<name>.h5` (or `<model>_domain_embeddings.h5`) into `--embeddings`.
- **Skip stages:** `--skip aacomp,kgiddi` (comma-separated; matches feature or graph model names).

### Profiles
- `standard`: local executor, conda enabled.
- `apptainer` / `singularity` / `docker`: container engine, with a 2 h pull timeout (the GPU image is large).
- `gpu`: adds `--nv` / `--gpus all`; on a cluster it is also the profile the institutional config keys on to send `process_gpu` tasks to the GPU queue.
- `test` / `test_full`: minimal SQLite triplet under `tests/data/`. Features are `aacomp,dummy,esm3_embeddings` — one cheap extracted encoder, the one encoder that draws from the RNG, and one published embedding, so the singleton and all-concat combos each mix extracted and published sources. `graph_models = ''`: the graph models are deliberately **not** covered, so changes to `ddiparsimony.py` / `kgiddi.py` / `load_data_gm.py` carry no snapshot signal and have to be argued for by hand.
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
