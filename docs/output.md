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
an in-distribution 1:1 set (`test_matched_1_1`, or `test_uniform_1_1` for
`random`) and a 1:10 set with the same positives and uniform negatives
(`test_uniform_1_10`), both
scored by the same trained model. A database with a single `test.sqlite3`
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
  - Structure features (`--machine_learning_features_structure`, off by default)
    are computed on domainsplit's AlphaFold 3 complexes. They are off because
    current domainsplit runs model complexes for most positives but almost no
    negatives, so structure coverage itself predicts the label. Positives come from PDB co-crystals
    that may have been in AF3's training data (PDB cutoff 2021-09-30), i.e.
    with AF3 that may have seen the positives, so interface features (buried surface,
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
- `<db_name>/nn_output/neural_network_<combo>/`,
  `<db_name>/rf_output/random_forest_<combo>/` and
  `<db_name>/svm_output/svm_<combo>/` -- with several `--seeds`, one copy per
  seed under `<x>_output/seed<N>/<model>_<combo>/` (`model_parameters.json`
  records its `seed`; the thresholds are per model per seed)
  - `model/`: the fitted model plus `model_parameters.json` (chosen
    hyperparameters, balance method, a `selection` block describing how they
    were chosen, and a `thresholds` block with the two validation operating
    points: `fpr0.01` — the smallest threshold with validation FPR ≤ 0.01, the
    primary point — and `universe_prior` — the MCC optimum at the scenario
    prevalence π = train positives / possible pairs of train domains, i.e.
    catalogue density. Each carries `resolvable`, by a different rule:
    `fpr0.01` is resolvable when validation has ≥ 100 negatives (so the budget
    allows at least one false positive); `universe_prior` is resolvable only
    when its optimum has ≥ 1 validation false positive — on any split size it is
    not when the optimum sits at validation FPR 0, where the true FPR is only
    bounded).
  - `predictions_<variant>.parquet`: one file per test split, scored by the one
    trained model. `<combo>` is a single feature name, `all` for the
    all-feature concatenation run (everything but `identity`), or `all_seq` for the
    sequence-only one (`all` minus structure features and
    `--structure_informed_features`). One row per DDI, with `domain_a <= domain_b`:
    the loader instantiates each pair in both orientations (the feature vector is
    `concat(emb_a, emb_b)`, so the model needs both to stay symmetric) and the
    two are averaged back into a single row here. Columns: `domain_a`,
    `domain_b`, `true_interaction`, `predicted_interaction` (at `fpr0.01`),
    `predicted_interaction_prior` (at `universe_prior`; ML models only) and
    `predicted_probability`. Both thresholds come from validation and are
    applied unchanged to every test split.
  - A model using `identity` also records `feature_digests` (the vocabulary
    sha1) in `model_parameters.json`; scoring fails on a test file built from
    another vocabulary. `identity__<split>.h5` carries `vocabulary_size`,
    `vocabulary_sha1` and the pair counts `n_pairs`, `n_pairs_both_known`,
    `n_pairs_one_unseen`, `n_pairs_zero` as attributes.
- `<db_name>/baseline_output/degree_baseline/` (with several `--seeds`, under
  `baseline_output/seed<N>/`)
  - `predictions_<variant>.parquet`: deg(A) × deg(B) per test DDI, deg = the
    family's positive DDIs in `train.csv`; columns `domain_a`, `domain_b`,
    `true_interaction`, `predicted_probability` (the integer product). No label
    columns and no model directory: the baseline is ranking-only and appears in
    the AUC/AP rows of the reports only.
- `<db_name>/graph_models/<model>/`
  - `model/`: graph-topology exports and the run record.
  - `predictions_<variant>.parquet`: one file per test split, same
    `domain_a <= domain_b` convention as the ML ones.
- `<db_name>/evaluation/<variant>/`
  - The per-(database, test variant) MultiQC report: `ddi_report.html`, its
    `ddi_report_data/`, and the JSON inputs under `multiqc_json/`.
  - Report sections, in order: "Model performance (AUC / AP)", then the two
    operating-point tables, "Operating point: validation FPR ≤ 0.01" (status,
    validation negatives, validation FPR/TPR, scored test positives/negatives,
    realised test FPR/TPR/precision, TP/FP/TN/FN, accuracy, balanced accuracy,
    F1, MCC) and "Scenario: universe-prior threshold" (status, π catalogue
    density, validation and test FPR/TPR, test precision, its own
    TP/FP/TN/FN, and π-precision with a note saying why a cell is blank).
    Every thresholded column names its point. 'Indicative' marks a point
    resolved on too few validation negatives (`fpr0.01`) or without a
    validation false positive (`universe_prior`). Each model's JSON under
    `per_model/` carries the same numbers as an `operating_points` object.
  - After the PR curves, three threshold-free source sections, built from the
    primary seed (each model's JSON carries them as `cross_source_auc` and
    `score_by_source`, for every seed):
    - "Score distribution by source": one box per (source, class), one tab per
      model (sample `<source> (+, n=…)` / `<source> (−, n=…)`). Box =
      quartiles, whiskers = minimum and maximum; a box hides the shape of the
      distribution (e.g. two modes). Compare sources within one tab, never
      across tabs: scores are model-specific rankings.
    - "ROC AUC by source pair" (heatmap, models × `<P> vs <N>`): the ROC AUC of
      positive source P's scored DDIs against negative source N's only, then
      the unweighted mean over each source's eligible cells (not a pooled AUC).
      A cell with fewer than 10 DDIs of either class is suppressed; a cell for
      a source the model did not score is blank too.
    - "ROC AUC by source pair — 95 % CI": per cell `AUC [lo, hi] · n+/n−`
      from a class-stratified bootstrap, "suppressed · …", "not scored", or
      the mean's `k` (eligible cells). DDIs sharing a Pfam family are not
      independent, so the intervals are narrow for new-family claims.

    A DDI with several sources counts in each of them. The two AUC sections
    are omitted when the dataset has one positive and one negative source
    (the one cell would equal the ROC AUC of each model that scored both
    classes; e.g. the default `external_test`, PPIDM_Gold vs hub-depleted
    INVERSE_PPIDM_Gold) or a
    single class; the box section's description then says so.
  - With several `--seeds` only: `seed_metrics.json` (one row per (model,
    seed): AUC, AP and both points' status, rates and counts) and two more
    report sections, "Per-seed results" and "ROC AUC order agreement across
    seeds". The metric tables then hold means over seeds (counts too) plus
    `Seeds`, `ROC AUC SD`, `PR AP SD`, `ROC AUC min–max` and `<rate> SD` for
    every thresholded rate; a status reads "resolved in k/n seeds", and
    π-precision is shown only if it is defined in every seed. Curves,
    coverage, by-source rates, the source sections, pairwise tests and
    enrichment are the primary (first) seed's.
  - With `--metadata` only: `enrichment.json`, the enrichment inputs of this
    variant (`observed` reference fit plus one entry per model), which the
    combined report reads, and the "Metadata enrichment" report sections. Each
    model gets `observed` (true label), refitted on the DDIs that model scored
    with complete metadata, so it shares input rows and candidate columns with
    its `predicted` target. The row "test DDIs with complete metadata
    (reference)" is `observed` on every complete test DDI. Slopes are per SD of
    the variant's metadata table. Full-model log2(odds ratio) heatmap cells are
    blank unless q is finite and below 0.05; the bars carry 95 % Wald intervals; the agreement
    table reads "k of m" (of the m `observed` slopes with q < 0.05, k have the
    same sign in `predicted`) or "n/a (<reason>)".

</details>

A run is seeded end to end from `--seeds` (see `bin/determinism.py`), and
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
    with metrics, ROC/PR curves, per-source rates and pairwise significance
    across all models. After the AUC/AP heatmaps comes "Test TPR at validation
    FPR ≤ 0.01" (model × dataset) with a details table per cell (realised test
    FPR, validation FPR, scored positives/negatives, test prevalence,
    'indicative'). A database with a 1:1 and a 1:k test set (e.g.
    `uniform_1_1` and `uniform_1_10`; a 1:k variant needs exactly one matching
    1:1 variant in its database) also gets two tables before it, built from
    each 1:k report's `test_set_shift.json`. "Same trained model: 1:1 test set
    vs 1:k test set": per model, AUC, TPR and FPR on both sets (TPR/FPR at the
    validation-selected threshold), Δ = 1:k − 1:1 with a 95% bootstrap CI that
    draws DDIs present in both sets jointly, counts, shared DDIs, and whether
    the 1:1 positives and negatives are subsets of the 1:k ones. "Model
    ranking: 1:1 vs 1:k": Spearman ρ of the AUC ranking. Points, CIs and ρ are
    the primary seed's; with several seeds, `Δ… SD (k/n)` is the sample SD of
    Δ across seeds (training variation, not test-set sampling). On `random` the
    negative sampler is uniform at both ratios and only prevalence changes; on
    matched → uniform splits the sampler changes too. Prevalence alone moves
    precision and AP, not FPR or AUC. Neither measures unseen proteins or
    families. The per-dataset operating-point tables follow, primary and
    prior together. The per-source section ("Per-source rate at validation
    FPR ≤ 0.01") names each bar group's rate: TPR on a positive-only source, TNR
    on a negative-only one, accuracy on a mixed one. It has one tab per dataset
    plus pooled `Combined_<ratio>` tabs, derived from the data: one per class
    ratio (the trailing `<pos>_<neg>` of the variant, so `uniform_1_1` and
    `matched_1_1` pool together) present in at least two datasets, each the
    unweighted mean (macro-average) of the per-dataset rates. A single-`test` dataset such
    as `external_test` joins no pooled tab. The `Mean over shown models` bar is
    the mean of the bars beside it, not an ensemble. Last, each dataset's
    "ROC AUC by source pair" heatmap and its CI table are copied through,
    heatmap then CI per dataset in `--mqc_order`; the per-source score box
    plots stay in the per-dataset reports.
  - `enrichment_data.csv` (with `--metadata` only): the enrichment report's
    tidy export, one row per (dataset, model, target, quantity, feature), with
    the fit's `n`, `status` and `reason`; skipped targets keep a `status` row.
    Quantities are `status`, `r2_adj`, `delta_r2`, `group_delta_r2`,
    `single_feature_r2`, `coef`, `se`, `p`, `q`, `log2_odds_ratio` and
    `single_feature_log2_odds`: the raw full-fit `r2` is not in it, and
    single-feature log-odds appear only as log2 odds ratios. Per-equation
    numbers carry an equation-qualified target, e.g. `combined: FP vs TN`,
    `combined: FN vs TN`, `combined: TP vs TN` (MNLogit, against TN). The model
    `test DDIs with complete metadata (reference)` is the reference `observed`
    fit; `coef`/`se` are per SD of that dataset's metadata table.

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
