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
    and emit one (meta, evaluation_dir) pair per (db, test variant) plus a
    merged versions channel.

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


def csvToList(s) {
    // Accept either a List or a comma-separated String. Filter out blanks
    // and the sentinel 'none' so callers can disable a list cleanly with
    // `--graph_models none` (or `--graph_models ''` once nf-schema allows it).
    def items = s instanceof List ? s : (s ? s.tokenize(',') : [])
    items*.trim().findAll { it && it.toLowerCase() != 'none' }
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

        FEATURE_EXTRACTION(Channel.from(ml_features), db_ch)

        // Group staged feature dirs back to one entry per DB.
        feature_dirs_per_db = FEATURE_EXTRACTION.out.feature_dir
            .map { meta, dir -> tuple(meta.db, dir) }
            .groupTuple()
        // emits: tuple(db_id, [feature_dir_1, feature_dir_2, ...])

        // Key DDI outputs by db_id so we can join with feature_dirs_per_db.
        ddi_keyed = ddi_ch.map { meta, ddi_dir -> tuple(meta.id, meta, ddi_dir) }

        // ---------------------------------------------------------------
        // NN / RF (per-feature singletons + one all-concat run, gated by
        // params.machine_learning_models and --skip)
        // ---------------------------------------------------------------
        nn_input_ch = nn_enabled ? ddi_keyed
            .join(feature_dirs_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.value(ml_config))
            .map { _db_id, meta, ddi_dir, feature_dirs, combo, cfg ->
                def combo_id = combo.size() == 1 ? combo[0] : 'all'
                def m = [
                    id      : "${meta.id}_neural_network_${combo_id}",
                    db      : meta.db,
                    model   : 'neural_network',
                    features: combo,
                    combo_id: combo_id,
                    tests   : meta.tests
                ]
                tuple(m, ddi_dir, feature_dirs, cfg)
            } : Channel.empty()
        NEURAL_NETWORK(nn_input_ch)

        rf_input_ch = rf_enabled ? ddi_keyed
            .join(feature_dirs_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.value(rf_config))
            .map { _db_id, meta, ddi_dir, feature_dirs, combo, cfg ->
                def combo_id = combo.size() == 1 ? combo[0] : 'all'
                def m = [
                    id      : "${meta.id}_random_forest_${combo_id}",
                    db      : meta.db,
                    model   : 'random_forest',
                    features: combo,
                    combo_id: combo_id,
                    tests   : meta.tests
                ]
                tuple(m, ddi_dir, feature_dirs, cfg)
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
                        run_label: run_label,
                        model    : model_name
                    ]
                    tuple(m_eval, pf)
                }
            }
        EVAL_ONE(all_predictions_ch)

        // ---------------------------------------------------------------
        // Per-(DB, variant) MultiQC reduce. Group EVAL_ONE outputs by
        // (db, variant), then join back to the expanded db channel to recover
        // (meta, db_path) for the EVALUATION call.
        // ---------------------------------------------------------------
        per_model_jsons_ch = EVAL_ONE.out.metrics
            .map { meta, j -> tuple("${meta.db}::${meta.variant}".toString(), j) }
            .groupTuple()

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

        // ---------------------------------------------------------------
        // Versions roll-up
        // ---------------------------------------------------------------
        ch_versions = DDI_EXTRACTION.out.versions
            .mix(FEATURE_EXTRACTION.out.versions)
            .mix(NEURAL_NETWORK.out.versions)
            .mix(RANDOM_FOREST.out.versions)
            .mix(GRAPH_MODEL.out.versions)
            .mix(EVAL_ONE.out.versions)
            .mix(EVALUATION.out.versions)

    emit:
        report   = EVALUATION.out.report   // tuple(meta, evaluation/)
        versions = ch_versions             // path versions.yml
}
