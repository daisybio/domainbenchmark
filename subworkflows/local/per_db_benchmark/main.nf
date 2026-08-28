/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PER_DB_BENCHMARK -- run the full per-database benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Given a channel of (db_meta, db_path) tuples, scatter:
        DDI_EXTRACTION                                  (per db)
            ↓
        FEATURE_EXTRACTION  (scatter (db × feature × split))
            ↓
        NEURAL_NETWORK + RANDOM_FOREST + GRAPH_MODEL    (per db × combo / model)
            ↓                 one predictions_<variant>.parquet per test split
        EVAL_ONE  (scatter — one tiny JSON per prediction)
            ↓
        EVALUATION  (per (db, test variant) MultiQC reduce)
    and emit one (meta, evaluation_dir) pair per (db, test variant).

    Training data (train + validation) is shared across a database's test
    variants: everything up to and including model fitting runs once per
    database, and only scoring and evaluation fan out per test set.
----------------------------------------------------------------------------*/

include { DDI_EXTRACTION                                } from '../../../modules/local/ddi_extraction/main.nf'
include { FEATURE_EXTRACTION                            } from '../../../modules/local/feature_extraction/main.nf'
include { NEURAL_NETWORK                                } from '../../../modules/local/neural_network/main.nf'
include { RANDOM_FOREST                                 } from '../../../modules/local/random_forest/main.nf'
include { GRAPH_MODEL                                   } from '../../../modules/local/graph_model/main.nf'
include { EVAL_ONE; EVALUATION                          } from '../../../modules/local/evaluation/main.nf'
include { VERIFY_EMBEDDINGS                             } from '../../../modules/local/verify_embeddings/main.nf'


def csvToList(s) {
    // Accept either a List or a comma-separated String. Filter out blanks
    // and the sentinel 'none' so callers can disable a list cleanly with
    // `--graph_models none` (or `--graph_models ''` once nf-schema allows it).
    def items = s instanceof List ? s : (s ? s.tokenize(',') : [])
    items*.trim().findAll { it && it.toLowerCase() != 'none' }
}

def publishedEmbeddingFile(feature) {
    // Locate the file backing a published feature under `--embeddings`.
    //
    // Two names are accepted. `<feature>.h5` is the direct one. domainsplit
    // currently publishes `<model>_domain_embeddings.h5` while the feature is
    // called `<model>_embeddings`, so that spelling is tried too -- the
    // "domain" in the middle is redundant now that the protein-level encoders
    // are gone, and this pipeline should not have to be re-released the day
    // domainsplit drops it.
    if (!params.embeddings) {
        error(
            "Feature '${feature}' is published by domainsplit rather than " +
            "extracted from the split databases, but --embeddings is not set. " +
            "Point it at domainsplit's results/embeddings/ directory, or drop " +
            "the feature with --skip ${feature}."
        )
    }
    def dir = file(params.embeddings, checkIfExists: true)
    def candidates = [
        dir / "${feature}.h5",
        dir / "${feature - '_embeddings'}_domain_embeddings.h5",
    ]
    def found = candidates.find { it.exists() }
    if (!found) {
        error(
            "No published embedding file for feature '${feature}' under " +
            "${params.embeddings}. Looked for: " +
            candidates.collect { it.getName() }.join(', ') + '.'
        )
    }
    found
}

def runLabel(db_id, variant) {
    // A database with an internal test set produces two runs
    // (`random_balanced`, `random_realistic`); one with a single `test` keeps
    // its bare name. The label identifies the run everywhere downstream --
    // process tags, the evaluation directory, and the entry in the combined
    // cross-database report.
    variant == 'test' ? "${db_id}".toString() : "${db_id}_${variant}".toString()
}

workflow PER_DB_BENCHMARK {

    take:
        db_ch   // channel: tuple(meta, db_path)

    main:
        // ---------------------------------------------------------------
        // Feature / model filtering (constant per pipeline run)
        // ---------------------------------------------------------------

        def skip_features        = csvToList(params.skip)
        def all_features         = csvToList(params.machine_learning_features)
        def ml_features          = all_features.findAll { !skip_features.contains(it) }

        // Split the feature list by where the h5 comes from.
        //
        // A *published* feature is not extracted from the split databases at
        // all: domainsplit embeds the cut domain sequence once per run and
        // publishes one mean-pooled vector per domain instance. That file is
        // per-RUN, not per-split -- `domain.id` is a surrogate integer that
        // SUBSET_SPLIT_DB copies verbatim and PRUNE_UNREPRESENTED_DDIS never
        // renumbers -- so one file serves train, validation and every test
        // split of the databases that run produced. Fanning it out over
        // (db x split) would schedule jobs to duplicate multi-gigabyte files
        // that are byte-identical, which is what STAGE_FEATURE_DIR used to do
        // and was deleted for.
        def all_published        = csvToList(params.published_features)
        def published_features   = ml_features.findAll { all_published.contains(it) }
        def extracted_features   = ml_features.findAll { !all_published.contains(it) }
        def published_files      = published_features.collect { publishedEmbeddingFile(it) }

        def all_graph_models     = csvToList(params.graph_models)
        def graph_model_names    = all_graph_models.findAll { !skip_features.contains(it) }

        def all_ml_models        = csvToList(params.machine_learning_models)
        def ml_model_names       = all_ml_models.findAll { !skip_features.contains(it) }
        def nn_enabled           = ml_model_names.contains('neural_network')
        def rf_enabled           = ml_model_names.contains('random_forest')

        // One run per feature (singleton) plus one all-feature concatenation
        // run when more than one feature is available.
        def feature_combos = ml_features.collect { [it] }
        if (ml_features.size() >= 2) feature_combos << ml_features

        def ml_config = file(params.modeljson) / 'NeuralNetwork.json'
        def rf_config = file(params.modeljson) / 'RandomForest.json'

        // ---------------------------------------------------------------
        // DDI + Feature extraction (per-DB)
        // ---------------------------------------------------------------
        DDI_EXTRACTION(db_ch)
        ddi_ch = DDI_EXTRACTION.out.ddi   // tuple(meta, ddi_dir)

        FEATURE_EXTRACTION(Channel.from(extracted_features), db_ch)

        // Gate the published files against the databases they will be paired
        // with, and hand them on renamed to `<feature>.h5`. NN/RF consume this
        // output, so a foreign-run embedding file cannot reach a GPU: see the
        // header of modules/local/verify_embeddings/main.nf for why nothing
        // downstream would notice on its own.
        verified_per_db = published_features
            ? VERIFY_EMBEDDINGS(
                  db_ch.map { meta, db_path ->
                      tuple(meta, db_path, published_features, published_files)
                  }
              ).h5.map { meta, files ->
                  tuple(meta.id, files instanceof java.util.List ? files : [files])
              }
            : Channel.empty()

        extracted_per_db = FEATURE_EXTRACTION.out.h5.groupTuple()

        // Group every h5 back to one entry per DB. The files are staged flat
        // into `features/` by NN/RF; their names carry the layout
        // (`<feature>__<split>.h5` per split, `<feature>.h5` per run), so no
        // process in between has to build a per-feature directory tree.
        //
        // `remainder: true` on both joins: a database can legitimately have no
        // extracted features at all (published embeddings only) or, because
        // FEATURE_EXTRACTION_ONE's output is optional, produce no file for a
        // split it does not carry. A plain join would drop that database from
        // the run without a word.
        //
        // Sorted for the same reason as the EVAL_ONE group below: the list is
        // staged as NN/RF's `features/*` input, and a task's hash covers its
        // input file order -- an unsorted group makes -resume re-run every
        // model on the next launch even when nothing changed.
        feature_files_per_db = db_ch
            .map { meta, _db_path -> tuple(meta.id, meta.id) }
            .join(extracted_per_db, remainder: true)
            .join(verified_per_db, remainder: true)
            .map { db_id, _marker, extracted, published ->
                def files = (extracted ?: []) + (published ?: [])
                tuple(db_id, files.toSorted { a, b -> a.name <=> b.name })
            }
        // emits: tuple(db_id, [aacomp__train.h5, ..., esm3_embeddings.h5, ...])

        // Key DDI outputs by db_id so we can join with feature_files_per_db.
        ddi_keyed = ddi_ch.map { meta, ddi_dir -> tuple(meta.id, meta, ddi_dir) }

        // ---------------------------------------------------------------
        // NN / RF (per-feature singletons + one all-concat run, gated by
        // params.machine_learning_models and --skip)
        // ---------------------------------------------------------------
        nn_input_ch = nn_enabled ? ddi_keyed
            .join(feature_files_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.value(ml_config))
            .map { _db_id, meta, ddi_dir, feature_files, combo, cfg ->
                def combo_id = combo.size() == 1 ? combo[0] : 'all'
                def m = [
                    id      : "${meta.id}_neural_network_${combo_id}",
                    db      : meta.db,
                    model   : 'neural_network',
                    features: combo,
                    combo_id: combo_id,
                    tests   : meta.tests
                ]
                tuple(m, ddi_dir, feature_files, cfg)
            } : Channel.empty()
        NEURAL_NETWORK(nn_input_ch)

        rf_input_ch = rf_enabled ? ddi_keyed
            .join(feature_files_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.value(rf_config))
            .map { _db_id, meta, ddi_dir, feature_files, combo, cfg ->
                def combo_id = combo.size() == 1 ? combo[0] : 'all'
                def m = [
                    id      : "${meta.id}_random_forest_${combo_id}",
                    db      : meta.db,
                    model   : 'random_forest',
                    features: combo,
                    combo_id: combo_id,
                    tests   : meta.tests
                ]
                tuple(m, ddi_dir, feature_files, cfg)
            } : Channel.empty()
        RANDOM_FOREST(rf_input_ch)

        // ---------------------------------------------------------------
        // Graph models
        // ---------------------------------------------------------------
        graph_input_ch = db_ch
            .combine(Channel.from(graph_model_names))
            .combine(Channel.value(file(params.modeljson)))
            .map { meta, db_path, model_name, modeljson ->
                def m = [
                    id   : "${meta.id}_${model_name}",
                    db   : meta.id,
                    model: model_name,
                    tests: meta.tests
                ]
                tuple(m, db_path, modeljson)
            }
        GRAPH_MODEL(graph_input_ch)

        // ---------------------------------------------------------------
        // Per-prediction evaluation (scatter)
        //
        // Each model emits one `predictions_<variant>.parquet` per test split
        // of its database. The variant is the fan-out axis from here on: every
        // test set is evaluated and reported as if it were its own dataset,
        // under the run label `<db>_<variant>` (bare `<db>` when the database
        // ships a single `test`).
        // ---------------------------------------------------------------
        all_predictions_ch = NEURAL_NETWORK.out.predictions
            .mix(RANDOM_FOREST.out.predictions)
            .mix(GRAPH_MODEL.out.predictions)
            .flatMap { meta, pred ->
                def files = pred instanceof java.util.List ? pred : [pred]
                files.collect { f ->
                    def pf         = file(f)
                    def model_name = pf.getParent().getName()
                    def variant    = pf.getSimpleName() - 'predictions_'
                    def run_label  = runLabel(meta.db, variant)
                    def m_eval     = [
                        id       : "${run_label}_${model_name}",
                        db       : meta.db,
                        variant  : variant,
                        // Split name behind the variant (`balanced` ->
                        // `test_balanced`), which is how EVAL_ONE finds the
                        // right `<split>_sources.csv`.
                        split    : (meta.tests ? meta.tests[variant] : null),
                        run_label: run_label,
                        model    : model_name
                    ]
                    tuple(m_eval, pf)
                }
            }

        // Per-source accuracy needs each test DDI's provenance list, so the
        // scatter task gets its database's DDI directory alongside the
        // predictions. `combine(by: 0)` rather than `join`: one DDI dir fans out
        // to every (model x variant) prediction of that database.
        eval_one_input_ch = all_predictions_ch
            .map { m, pf -> tuple(m.db, m, pf) }
            .combine(ddi_keyed.map { id, _meta, ddi_dir -> tuple(id, ddi_dir) }, by: 0)
            .map { _db_id, m, pf, ddi_dir -> tuple(m, pf, ddi_dir) }
        EVAL_ONE(eval_one_input_ch)

        // ---------------------------------------------------------------
        // Per-(DB, variant) MultiQC reduce. Group EVAL_ONE outputs by
        // (db, variant), then join back to the expanded db channel to recover
        // (meta, db_path) for the EVALUATION call.
        // ---------------------------------------------------------------
        per_model_jsons_ch = EVAL_ONE.out.metrics
            .map { meta, j -> tuple("${meta.db}::${meta.variant}".toString(), j) }
            .groupTuple()
            // groupTuple() emits in task-completion order. That list becomes
            // eval_multiqc.py's --per_model_metrics argument order, and MultiQC
            // writes its JSON in the order it was fed -- so without this sort
            // the report bytes flip between runs depending on which model
            // finished first. It also stabilises the task hash for -resume.
            .map { key, jsons -> tuple(key, jsons.toSorted { a, b -> a.name <=> b.name }) }

        // One entry per (database, test variant). Train/validation splits are
        // shared, so only the test split differs between an entry pair.
        db_variant_ch = db_ch
            .flatMap { meta, db_path ->
                meta.tests.collect { variant, split ->
                    def run_label = runLabel(meta.id, variant)
                    def m = [
                        id       : run_label,
                        db       : meta.id,
                        variant  : variant,
                        split    : split,
                        run_label: run_label
                    ]
                    tuple("${meta.id}::${variant}".toString(), m, db_path)
                }
            }

        evaluation_input_ch = db_variant_ch
            .join(per_model_jsons_ch)
            .map { _key, meta, db_path, jsons ->
                tuple(
                    meta,
                    db_path,
                    jsons,
                    "${workflow.projectDir}/${params.outdir}/${params.old_report}"
                )
            }
        EVALUATION(evaluation_input_ch)

    emit:
        report = EVALUATION.out.report   // tuple(meta, evaluation/)
}
