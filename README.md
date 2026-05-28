# daisybio/domainbenchmark

[![Open in GitHub Codespaces](https://img.shields.io/badge/Open_In_GitHub_Codespaces-black?labelColor=grey&logo=github)](https://github.com/codespaces/new/daisybio/domainbenchmark)
[![GitHub Actions CI Status](https://github.com/daisybio/domainbenchmark/actions/workflows/nf-test.yml/badge.svg)](https://github.com/daisybio/domainbenchmark/actions/workflows/nf-test.yml)
[![GitHub Actions Linting Status](https://github.com/daisybio/domainbenchmark/actions/workflows/linting.yml/badge.svg)](https://github.com/daisybio/domainbenchmark/actions/workflows/linting.yml)
[![nf-test](https://img.shields.io/badge/unit_tests-nf--test-337ab7.svg)](https://www.nf-test.com)

<!-- Zenodo DOI badge will be added after the v1.0.0 release. -->


[![Nextflow](https://img.shields.io/badge/version-%E2%89%A525.10.2-green?style=flat&logo=nextflow&logoColor=white&color=%230DC09D&link=https%3A%2F%2Fnextflow.io)](https://www.nextflow.io/)
[![nf-core template version](https://img.shields.io/badge/nf--core_template-4.0.2-green?style=flat&logo=nfcore&logoColor=white&color=%2324B064&link=https%3A%2F%2Fnf-co.re)](https://github.com/nf-core/tools/releases/tag/4.0.2)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?labelColor=000000&logo=docker)](https://www.docker.com/)
[![run with singularity](https://img.shields.io/badge/run%20with-singularity-1d355c.svg?labelColor=000000)](https://sylabs.io/docs/)
[![Launch on Seqera Platform](https://img.shields.io/badge/Launch%20%F0%9F%9A%80-Seqera%20Platform-%234256e7)](https://cloud.seqera.io/launch?pipeline=https://github.com/daisybio/domainbenchmark)

## Introduction

**daisybio/domainbenchmark** is a bioinformatics benchmarking pipeline for protein **domain-domain
interaction (DDI)** prediction. Given one or more pre-split DDI databases
(`train.sqlite3`, `test.sqlite3`, `optimization.sqlite3` plus matching
embeddings), the pipeline trains a panel of machine-learning and
graph-based predictors, evaluates each one against a held-out test split,
and produces a unified MultiQC report comparing them.

The pipeline runs the following stages:

1. **DDI extraction** from the database split (`DDI_EXTRACTION`).
2. **Feature extraction** for every requested encoding (`aacomp`,
   `aaencode`, ProtT5 / ESM-3 / ESM-C protein and domain embeddings) —
   parallelized per `(feature × split)`.
3. **ML classifiers** trained on each feature individually plus one
   all-feature concatenation run (gated by `--machine_learning_models`):
   - `NEURAL_NETWORK` — neural network (PyTorch + skorch).
   - `RANDOM_FOREST` — RAPIDS cuML random forest on GPU.
4. **Graph models** (`GRAPH_MODEL`): KGIDDI, DDI parsimony, KGIDDI-random.
5. **Per-prediction evaluation** (`EVAL_ONE`) → tiny per-model JSONs.
6. **Per-database aggregation** (`EVALUATION`) → MultiQC report.
7. **Cross-database aggregation** (`COMBINE_EVAL`) when `--db_list` is set.

```mermaid
flowchart LR
    subgraph "per database"
        ddi[DDI_EXTRACTION]
        feat[FEATURE_EXTRACTION]
        nn[NEURAL_NETWORK]
        rf[RANDOM_FOREST]
        gm[GRAPH_MODEL]
        eo[EVAL_ONE]
        ev[EVALUATION]
    end
    db[(db_list)] --> ddi
    db --> feat --> nn & rf
    db --> gm
    ddi --> nn & rf
    nn & rf & gm --> eo --> ev
    ev --> agg[COMBINE_EVAL] --> report[ddi_report.html]
```



## Usage

> [!NOTE]
> If you are new to Nextflow and nf-core, please refer to [this page](https://nf-co.re/docs/get_started/environment_setup/overview) on how to set-up Nextflow. Make sure to [test your setup](https://nf-co.re/docs/get_started/run-your-first-pipeline) with `-profile test` before running the workflow on actual data.

### Samplesheet (recommended entry)

```bash
nextflow run . \
    -profile <docker/singularity/conda>,slurm \
    --input assets/samplesheet.csv \
    --outdir results
```

The samplesheet is a CSV with one row per database split:

```csv
id,db_path
random_denoise,/path/to/random_denoise
random_ddi,/path/to/random_ddi
```

Schema in `assets/schema_input.json`. Each `db_path` must contain
`train.sqlite3`, `test.sqlite3`, `optimization.sqlite3`.

### Cluster (Slurm + Singularity)

```bash
nextflow run . \
    -profile slurm,singularity \
    --input assets/samplesheet.csv \
    --outdir /nfs/scratch/cobinet/results \
    -resume
```

### Skipping stages

`--skip` accepts a comma-separated list of feature names or graph-model
names. For example, to skip the heavy embedding-based features and the
two parsimony graph models:

```bash
nextflow run . \
    -profile slurm,singularity \
    --input assets/samplesheet.csv \
    --skip "kgiddi,ddiparsimony,prott5_protein_domain_embeddings,esm3_protein_domain_embeddings"
```

### Test profile (in-repo fixture)

```bash
nextflow run . -profile test,singularity --outdir results-test
```

The `test` profile points at a tiny in-repo SQLite triple under
`tests/data/` and disables every heavy feature except `aacomp`. Used by
CI and by `nf-test`.

## Pipeline parameters

| Parameter | Default | Description |
|---|---|---|
| `--input` | `null` | **Required.** Samplesheet CSV (one row per database split). |
| `--outdir` | `./results` | Output directory. |
| `--modeljson` | `${projectDir}/assets` | Directory of model hyperparameter JSONs. |
| `--skip` | `''` | Comma-separated feature/model names to skip. |
| `--graph_models` | `kgiddi,ddiparsimony,kgiddi_random` | Graph models to run. |
| `--machine_learning_features` | `aacomp,aaencode,prott5_*,esm3_*,esmc_*` | Feature encodings to compute. |
| `--large_features` | `prott5_*,esm3_*,esmc_*` | Features routed to `process_gpu_large`. |
| `--machine_learning_models` | `neural_network,random_forest` | ML models to run. |
| `--max_protein_combinations_per_ddi` | `null` | Cap on protein-pair instantiations per DDI pair (sampled without replacement). Null = use all. |
| `--seed` | `42` | Global RNG seed. |
| `--publish_dir_mode` | `'copy'` | Nextflow `publishDir` mode. |

Full schema with defaults, types, and descriptions: `nextflow_schema.json`.
Run `nextflow run . --help` for a CLI summary.

## Pipeline output

For each database split processed, a subdirectory under `--outdir/<db_name>/`:

```
<outdir>/<db_name>/
├── data/
│   └── <feature>/
│       ├── train.h5
│       ├── test.h5
│       └── optimization.h5
├── nn_output/
│   └── neural_network_<feature_combo>/
│       ├── predictions.parquet
│       └── model/
├── rf_output/
│   └── random_forest_<feature_combo>/
│       ├── predictions.parquet
│       └── model/
├── graph_models/
│   └── <model_name>/
│       ├── predictions.parquet
│       └── model/
└── evaluation/
    └── multiqc_report.html
```

When the samplesheet contains more than one database, a top-level
cross-database report is also written: `<outdir>/evaluation/ddi_report.html`.

A full description of each output is in [`docs/output.md`](docs/output.md).

## Profiles

| Profile | Effect |
|---|---|
| `standard` | Local executor + conda. Default. |
| `docker` | Local executor + docker. |
| `singularity` | Local executor + singularity / apptainer. |
| `conda` | Forces conda; disables docker / singularity. |
| `slurm` | Slurm executor with GPU labels and retry-on-OOM. Pairs with `singularity` on the cluster. |
| `test` | Tiny in-repo fixture for smoke tests and `nf-test`. |
| `test_full` | Full-data CI run (large fixtures). |

## Adding new components

### New feature encoding

Copy the template and implement your feature computation:

1. Copy `bin/features/new_feature.py` to `bin/features/<your_feature>.py`
2. Implement `extract_features(conn, out_file)` — read from SQLite, write feature vectors to HDF5
3. Append `<your_feature>` to `params.machine_learning_features` in `nextflow.config`
4. If it needs GPU or large memory, also add it to `params.large_features`

See `bin/features/new_feature.py` for the full contract and examples.

### New ML model

Copy the template and implement your training/prediction logic:

1. Copy `bin/new_model.py` to `bin/<your_model>.py`
2. Subclass `DDIModelTrainer` and implement the required methods
3. Create `assets/<YourModel>.json` with hyperparameter grid (must include `model_name`, `data`, `search_parameters`, `model_parameters`)
4. Add a Nextflow process in `modules/local/<your_model>/main.nf` and wire it into `subworkflows/local/per_db_benchmark/main.nf`

See `bin/new_model.py` for the full API and optional overrides.

## Credits

daisybio/domainbenchmark was originally written by Konstantin Pelz, Chiara Thomas, Christian Romberg.

We thank the following people for their extensive assistance in the development of this pipeline:

- Amelie Hilbig

## Contributions and Support

If you would like to contribute to this pipeline, please see the [contributing guidelines](docs/CONTRIBUTING.md).

## Citations

A Zenodo DOI for daisybio/domainbenchmark will be issued at the v1.0.0
release; this section will be updated then. An extensive list of
references for the tools used by the pipeline can be found in the
[`CITATIONS.md`](CITATIONS.md) file.

This pipeline uses code and infrastructure developed and maintained by the [nf-core](https://nf-co.re) community, reused here under the [MIT license](https://github.com/nf-core/tools/blob/main/LICENSE).

> **The nf-core framework for community-curated bioinformatics pipelines.**
>
> Philip Ewels, Alexander Peltzer, Sven Fillinger, Harshil Patel, Johannes Alneberg, Andreas Wilm, Maxime Ulysse Garcia, Paolo Di Tommaso & Sven Nahnsen.
>
> _Nat Biotechnol._ 2020 Feb 13. doi: [10.1038/s41587-020-0439-x](https://dx.doi.org/10.1038/s41587-020-0439-x).
