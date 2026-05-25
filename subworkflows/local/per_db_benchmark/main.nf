/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PER_DB_BENCHMARK -- run the full per-database benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Given a channel of (db_meta, db_path) tuples, scatter:
        DDI_EXTRACTION
            ↓
        FEATURE_EXTRACTION  (scatter (db × feature × split))
            ↓
        NEURAL_NETWORK + RANDOM_FOREST + GRAPH_MODEL  (per db × combo / model)
            ↓
        EVAL_ONE  (scatter — one tiny JSON per prediction)
            ↓
        EVALUATION  (per-DB MultiQC reduce)
    and emit one (meta, evaluation_dir) pair per DB plus a merged versions
    channel.
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
                    combo_id: combo_id
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
                    combo_id: combo_id
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
                    model: model_name
                ]
                tuple(m, db_path, modeljson)
            }
        GRAPH_MODEL(graph_input_ch)

        // ---------------------------------------------------------------
        // Per-prediction evaluation (scatter)
        // ---------------------------------------------------------------
        all_predictions_ch = NEURAL_NETWORK.out.predictions
            .mix(RANDOM_FOREST.out.predictions)
            .mix(GRAPH_MODEL.out.predictions)
            .map { meta, pred ->
                def f          = pred instanceof java.util.List ? pred[0] : pred
                def model_name = file(f).getParent().getName()
                def m_eval     = [
                    id   : "${meta.db}_${model_name}",
                    db   : meta.db,
                    model: model_name
                ]
                tuple(m_eval, f)
            }
        EVAL_ONE(all_predictions_ch)

        // ---------------------------------------------------------------
        // Per-DB MultiQC reduce. Group EVAL_ONE outputs by DB, then join
        // back to db_ch to recover (meta, db_path) for the EVALUATION call.
        // ---------------------------------------------------------------
        per_model_jsons_ch = EVAL_ONE.out.metrics
            .map { meta, j -> tuple(meta.db, j) }
            .groupTuple()

        evaluation_input_ch = db_ch
            .map { meta, db_path -> tuple(meta.id, meta, db_path) }
            .join(per_model_jsons_ch)
            .map { _id, meta, db_path, jsons ->
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
