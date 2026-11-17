# daisybio/domainbenchmark: Output

## Introduction

This document describes the output produced by the pipeline. All paths are
relative to the top-level results directory (`--outdir`).

Outputs are namespaced per database. Everything up to and including model
fitting is shared across a database's test variants, so it sits directly under
`<db_name>/`; only the evaluation fans out, into
`<db_name>/evaluation/<variant>/`.

A `<variant>` is the name of a test split without its `test_` prefix: a
database may ship any number of `test_<variant>.sqlite3`, and each is reported
as its own dataset `<db_name>_<variant>`. The current domainsplit layout has
`test_1_1` (in distribution, 1:1) and `test_1_10` (out of distribution, 1:10),
both scored by the same trained model. A database with a single `test.sqlite3`
(e.g. `external_test`) has the variant `test` and reports under its bare name.

## Pipeline overview

- [Per-database outputs](#per-database-outputs) — DDI tables, features, trained models, predictions, per-database report
- [Combined report](#combined-report) — the cross-database `ddi_report.html`
- [Pipeline information](#pipeline-information) — Nextflow execution reports

### Per-database outputs

<details markdown="1">
<summary>Output files</summary>

- `<db_name>/ddi/DDI/`
  - `<split>.csv`: `(domain_1, domain_2, interaction)` rows for one split.
    `domain_1`/`domain_2` are **Pfam accessions**, not `domain.id`: that is a
    per-run surrogate integer, so a report keyed on it could not be compared
    between domainsplit runs. Everything downstream uses the same key.
  - `<split>_sources.csv`: the same rows plus each DDI's comma-joined
    provenance list, which drives the per-source accuracy section of the report.
  - `<split>_instances.csv`: the domain-instance pairs `ddi_split_membership`
    assigns to the split. This is what the ML loader instantiates, instead of a
    cross-product over every instance of each family.
- `<db_name>/data/`
  - `<feature>__<split>.h5`: one feature encoding for one split, keyed
    `h5[pfam_id][instance_key]`. The flat name carries the layout; each ML
    task stages only its own combo's files into `features/` and resolves them
    by name, so adding a feature does not rerun models that do not use it.
  - Structure features (`--machine_learning_features_structure`) are computed
    on domainsplit's AlphaFold 3 complexes. Positives come from PDB co-crystals
    that are in AF3's training data, so interface features (buried surface,
    contacts, clashes, patches) partly measure AF3's recall of known complexes
    rather than predict an interaction. Read any structure-assisted model --
    a structure singleton or `all` -- with that caveat. Such a model also scores
    only the DDIs with at least one structure-covered instance pair.
  - Features listed in `--published_features` produce nothing here. domainsplit
    already exported them, once per run, and `--embeddings` points at that
    directory; `VERIFY_EMBEDDINGS` checks each file against the split databases
    and stages it as `features/<feature>.h5` without copying it into `results/`.
    One file serves `train`, `validation` and every `test*` split of the
    databases the same run produced.
- `<db_name>/nn_output/neural_network_<combo>/` and
  `<db_name>/rf_output/random_forest_<combo>/`
  - `model/`: the fitted model plus `model_parameters.json` (chosen
    hyperparameters, balance method, tuned decision threshold).
  - `predictions_<variant>.parquet`: one file per test split, scored by the one
    trained model. `<combo>` is a single feature name, or `all` for the
    all-feature concatenation run. One row per DDI, with `domain_a <= domain_b`:
    the loader instantiates each pair in both orientations (the feature vector is
    `concat(emb_a, emb_b)`, so the model needs both to stay symmetric) and the
    two are averaged back into a single row here.
- `<db_name>/graph_models/<model>/`
  - `model/`: graph-topology exports and the run record.
  - `predictions_<variant>.parquet`: one file per test split, same
    `domain_a <= domain_b` convention as the ML ones.
- `<db_name>/evaluation/<variant>/`
  - The per-(database, test variant) MultiQC report: `ddi_report.html`, its
    `ddi_report_data/`, and the JSON inputs under `multiqc_json/`.

</details>

A run is seeded end to end from `--seed` (see `bin/determinism.py`), and
`PYTHONHASHSEED` is fixed for every task in `nextflow.config`. On the tested
setup -- the CPU backend with the pinned containers, as the nf-test suite runs
it (`-profile test,docker --allow_cpu_ml true`) -- two runs of the same commit on
the same data produce byte-identical models, predictions and metrics; the
nf-test snapshot asserts it. The GPU backends (PyTorch CUDA, cuML) and other
hardware or library versions are seeded the same way but not tested for
bit-identity. In either case MultiQC's own report metadata differs between
runs: it embeds a timestamp, absolute work-dir paths and the MultiQC version.

### Combined report

<details markdown="1">
<summary>Output files</summary>

- `evaluation/`
  - `ddi_report.html`: every (database, test variant) as its own dataset entry,
    with metrics, ROC/PR curves, per-source accuracy and pairwise significance
    across all models. The per-source accuracy section has one tab per dataset
    plus pooled `Combined_<variant>` tabs, derived from the data: one per
    `test_<variant>` variant present in at least two datasets, each the
    unweighted mean of the per-dataset accuracies. A single-`test` dataset such
    as `external_test` joins no pooled tab. The `Mean over shown models` bar is
    the mean of the bars beside it, not an ensemble.

</details>

### Pipeline information

<details markdown="1">
<summary>Output files</summary>

- `pipeline_info/`
  - Reports generated by Nextflow: `execution_report.html`,
    `execution_timeline.html`, `execution_trace.txt` and
    `pipeline_dag.dot`/`pipeline_dag.svg`.
  - `pipeline_report.html` / `pipeline_report.txt`: only present when `--email`
    / `--email_on_fail` is used.
  - `params.json`: the parameters this run was launched with.

</details>

The pipeline does not collect software versions. Provenance comes from the
container tags (one image per environment, named in the module definitions,
e.g. `domainbenchmark-general:1.0.2`) and the Nextflow execution trace. Tags are
mutable: the image behind a tag is not pinned by digest, so record the digest
your engine pulled (`docker image inspect` / the SIF file) when it matters.

[Nextflow](https://www.nextflow.io/docs/latest/tracing.html) provides excellent
functionality for generating various reports relevant to the running and
execution of the pipeline. This will allow you to troubleshoot errors with the
running of the pipeline, and also provide you with other information such as
launch commands, run times and resource usage.
