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
include { PREFLIGHT                                     } from '../../../modules/local/preflight/main.nf'
include { FEATURE_EXTRACTION                            } from '../../../modules/local/feature_extraction/main.nf'
include { NEURAL_NETWORK                                } from '../../../modules/local/neural_network/main.nf'
include { RANDOM_FOREST                                 } from '../../../modules/local/random_forest/main.nf'
include { SVM                                           } from '../../../modules/local/svm/main.nf'
include { GRAPH_MODEL                                   } from '../../../modules/local/graph_model/main.nf'
include { DEGREE_BASELINE                               } from '../../../modules/local/degree_baseline/main.nf'
include { EVAL_ONE; EVALUATION                          } from '../../../modules/local/evaluation/main.nf'
include { VERIFY_EMBEDDINGS                             } from '../../../modules/local/verify_embeddings/main.nf'
include { METADATA; ENRICHMENT                          } from '../../../modules/local/enrichment/main.nf'
include { csvToList; runLabel; referenceVariant; featureVariants; parseSeeds } from '../utils_nfcore_domainbenchmark_pipeline'


def embeddingsDir(features) {
    // The directory holding the published embedding files.
    //
    // `--embeddings` names it outright. Left unset it is derived, because
    // domainsplit publishes both under one results directory:
    //
    //     <run>/results/databases/<db>/{train,validation,test*}.sqlite3
    //     <run>/results/embeddings/<model>_domain_embeddings.h5
    //
    // so `--input <run>/results/databases` already says where the embeddings
    // are. Only a directory input can be resolved this way: a samplesheet's
    // `db_path` rows may point anywhere, including at several runs, and guessing
    // from one of them would silently pair a database with a foreign run's
    // embeddings -- exactly the failure VERIFY_EMBEDDINGS exists to catch.
    if (params.embeddings) {
        return file(params.embeddings, checkIfExists: true)
    }

    def named = features.join(', ')
    def explain =
        "Feature(s) ${named} are published by domainsplit rather than extracted " +
        "from the split databases, so the pipeline needs its embeddings directory."
    def remedy =
        "Set --embeddings to domainsplit's results/embeddings/, or drop the " +
        "feature(s) with --skip ${features.join(',')}."

    def input_path = params.input ? file(params.input) : null
    if (!input_path || !input_path.isDirectory()) {
        error(
            "${explain} --embeddings is not set, and it can only be derived from " +
            "a directory --input (a samplesheet's rows can point at several " +
            "runs). ${remedy}"
        )
    }

    // Sibling of the databases/ directory, i.e. one level up from --input.
    def candidate = input_path.getParent() / 'embeddings'
    if (!candidate.exists() || !candidate.isDirectory()) {
        error(
            "${explain} --embeddings is not set and no 'embeddings' directory " +
            "exists next to --input (looked for ${candidate}). ${remedy}"
        )
    }
    log.info("[embeddings] --embeddings not set; using ${candidate} (sibling of --input)")
    return candidate
}

def publishedEmbeddingFile(dir, feature) {
    // Locate the file backing a published feature under the embeddings dir.
    //
    // Two names are accepted. `<feature>.h5` is the direct one. domainsplit
    // currently publishes `<model>_domain_embeddings.h5` while the feature is
    // called `<model>_embeddings`, so that spelling is tried too -- the
    // "domain" in the middle is redundant now that the protein-level encoders
    // are gone, and this pipeline should not have to be re-released the day
    // domainsplit drops it.
    def candidates = [
        dir / "${feature}.h5",
        dir / "${feature - '_embeddings'}_domain_embeddings.h5",
    ]
    def found = candidates.find { it.exists() }
    if (!found) {
        error(
            "No published embedding file for feature '${feature}' under " +
            "${dir}. Looked for: " +
            candidates.collect { it.getName() }.join(', ') + '.'
        )
    }
    found
}

def structuresFile(features) {
    // domainsplit's structures.h5, found like embeddingsDir(): `--structures`
    // names it, otherwise it is `structures/structures.h5` beside --input's
    // databases/ directory, which is where domainsplit publishes it.
    if (params.structures) {
        return file(params.structures, checkIfExists: true)
    }
    def remedy =
        "Set --structures to domainsplit's results/structures/structures.h5, or " +
        "drop the structure feature(s) with --machine_learning_features_structure none " +
        "or --skip ${features.join(',')}."
    def input_path = params.input ? file(params.input) : null
    if (!input_path || !input_path.isDirectory()) {
        error(
            "Structure feature(s) ${features.join(', ')} need domainsplit's structures.h5. " +
            "--structures is not set, and it can only be derived from a directory --input. ${remedy}"
        )
    }
    def candidate = input_path.getParent() / 'structures' / 'structures.h5'
    if (!candidate.exists()) {
        error(
            "Structure feature(s) ${features.join(', ')} need domainsplit's structures.h5, " +
            "and none exists next to --input (looked for ${candidate}). ${remedy}"
        )
    }
    log.info("[structures] --structures not set; using ${candidate} (sibling of --input)")
    return candidate
}

def comboFiles(files, combo, splits, seed) {
    // The feature files one model task stages: only those of its own combo,
    // and of a seeded feature only the copy drawn with this task's seed.
    // `files` holds [seed, file] pairs; a seed-free file carries seed null.
    // Staging the whole database's set put every feature into every model's
    // task hash, so adding one feature re-ran every model on -resume.
    // Names follow resolve_feature_file (bin/machine_learning.py):
    // `<feature>__<split>.h5` (extracted, structure variants included) or
    // `<feature>.h5` (published). Matched exactly, so `k_closest_distance`
    // does not pick up `k_closest_distance_5k`.
    def wanted = combo.collectMany { feat -> ["${feat}.h5".toString()] + splits.collect { "${feat}__${it}.h5".toString() } }
    files.findAll { s, f -> wanted.contains(f.name) && (s == null || s == seed) }.collect { pair -> pair[1] }
}

workflow PER_DB_BENCHMARK {

    take:
        db_ch   // channel: tuple(meta, db_path)

    main:
        // ---------------------------------------------------------------
        // Feature / model filtering (constant per pipeline run)
        // ---------------------------------------------------------------

        def skip_features        = csvToList(params.skip)

        // Training seeds; the first is the primary seed (graph models,
        // enrichment, seed-free feature encoders). Validated at launch.
        def seeds                = parseSeeds(params.seeds)
        def primary_seed         = seeds[0]
        // Encoders that draw from the RNG, i.e. whose output depends on the
        // seed: they are extracted once per seed, every other encoder once
        // with the primary seed. A new sampling encoder must be added here, or
        // every seed silently reuses the primary seed's vectors (CLAUDE.md,
        // "Deriving per item beats drawing in a loop"). None draws today
        // (`identity` replaced the random `dummy`); the mechanism stays.
        def seeded_features      = []
        // Encoders built against a vocabulary of the database's train +
        // validation families: FEATURE_EXTRACTION_ONE stages both sqlite files
        // for them as inputs of their own (extract_features.py --vocabulary_db).
        def vocabulary_features  = ['identity']

        def all_ml_models        = csvToList(params.machine_learning_models)
        def ml_model_names       = all_ml_models.findAll { !skip_features.contains(it) }
        def nn_enabled           = ml_model_names.contains('neural_network')
        def rf_enabled           = ml_model_names.contains('random_forest')
        def svm_enabled          = ml_model_names.contains('svm')
        def any_ml               = nn_enabled || rf_enabled || svm_enabled

        // Features exist only to feed the ML models. With none enabled (a graph-only
        // run) every FEATURE_EXTRACTION and VERIFY_EMBEDDINGS task would be
        // wasted, and a samplesheet run without --embeddings would even fail in
        // embeddingsDir() asking for a directory nothing reads.
        def all_features         = any_ml ? csvToList(params.machine_learning_features) : []
        // Structure features read domainsplit's AF3 complexes. They take part
        // in every combo like any other feature, `_all` included -- so `_all`
        // scores only the DDIs with at least one structure-covered instance pair.
        // `_all_seq` is the one combo without them.
        def all_structure        = any_ml ? csvToList(params.machine_learning_features_structure) : []
        def structure_features   = all_structure.findAll { !skip_features.contains(it) }
        def ml_features          = all_features.findAll { !skip_features.contains(it) } + structure_features

        // Split the feature list by where the h5 comes from.
        //
        // A *published* feature is not extracted from the split databases at
        // all: domainsplit embeds the cut domain sequence once per run and
        // publishes one mean-pooled vector per domain instance, keyed
        // `h5[pfam_id][instance_id]`. That file is per-RUN, not per-split -- it
        // holds every domain the run saw and each split database is a subset of
        // that -- so one file serves train, validation and every test split of
        // the databases that run produced. Fanning it out over (db x split)
        // would schedule jobs to duplicate multi-gigabyte files that are
        // byte-identical, which is what STAGE_FEATURE_DIR used to do and was
        // deleted for.
        def all_published        = csvToList(params.published_features)
        def published_features   = ml_features.findAll { all_published.contains(it) }
        def extracted_features   = ml_features.findAll { !all_published.contains(it) }
        def embeddings_dir       = published_features ? embeddingsDir(published_features) : null
        def published_files      = published_features.collect {
            publishedEmbeddingFile(embeddings_dir, it)
        }

        def all_graph_models     = csvToList(params.graph_models)
        def graph_model_names    = all_graph_models.findAll { !skip_features.contains(it) }

        // One run per feature (singleton) plus two concatenations, each built
        // when it holds >= 2 features. `identity` is a reference (what family
        // identity alone explains), so it gets its singleton run but joins neither.
        //   all     -- every other feature, structure ones included.
        //   all_seq -- sequence features only: no structure feature and no
        //              structure-informed embedding (params.structure_informed_features),
        //              so it scores the full population. Skipped when it
        //              would be the same set as `all`.
        // Each combo is [id: ..., features: [...]]; the id names the output
        // directory, task tag and report entry, so it must be unique.
        def structure_informed = csvToList(params.structure_informed_features)
        def feature_combos = ml_features.collect { [id: it, features: [it]] }
        def all_combo      = ml_features.findAll { it != 'identity' }
        def all_seq_combo  = all_combo.findAll { !structure_features.contains(it) && !structure_informed.contains(it) }
        if (all_combo.size() >= 2) feature_combos << [id: 'all', features: all_combo]
        if (all_seq_combo.size() >= 2 && all_seq_combo != all_combo) feature_combos << [id: 'all_seq', features: all_seq_combo]
        def dup_combo_ids = feature_combos*.id.countBy { it }.findAll { _id, n -> n > 1 }.keySet()
        if (dup_combo_ids) {
            error("Feature combo id(s) ${dup_combo_ids.join(', ')} used twice: a feature may not be called 'all' or 'all_seq'.")
        }

        // Every model scores every test split, so each (db, variant) receives
        // exactly this many predictions. Known up front, which lets the
        // EVALUATION group below emit as soon as its own models are done
        // instead of waiting for the slowest model of every database.
        // ENRICHMENT sees the primary seed's predictions only, so its group
        // has one prediction per model; EVAL_ONE's has one per (ML model, seed)
        // plus one per seed for the degree baseline, which ENRICHMENT never
        // sees (ranking-only, W-15): (M + 1) x S and M.
        def ml_models_per_variant = feature_combos.size() * ((nn_enabled ? 1 : 0) + (rf_enabled ? 1 : 0) + (svm_enabled ? 1 : 0))
        def models_per_variant = ml_models_per_variant + graph_model_names.size()
        def evals_per_variant  = (ml_models_per_variant + 1) * seeds.size() + graph_model_names.size()
        // The guard counts real models only: the baseline alone is no benchmark.
        if (models_per_variant == 0) {
            error(
                "Nothing to benchmark: no ML model, feature or graph model is left after " +
                "--machine_learning_models / --graph_models / --skip."
            )
        }

        def ml_config = file(params.modeljson) / 'NeuralNetwork.json'
        def rf_config = file(params.modeljson) / 'RandomForest.json'
        def svm_config = file(params.modeljson) / 'SVM.json'

        // ---------------------------------------------------------------
        // DDI + Feature extraction (per-DB)
        // ---------------------------------------------------------------
        DDI_EXTRACTION(db_ch)
        ddi_ch = DDI_EXTRACTION.out.ddi   // tuple(meta, ddi_dir)

        // A feature name may be a parameterised variant of an encoder module
        // (assets/feature_variants.json); anything not listed there is its own
        // module with no parameters.
        def variants = featureVariants()
        def feature_specs = extracted_features.collect { name ->
            def v = variants[name]
            def seeded = seeded_features.contains(name)
            [name: name, module: v ? v.module : name, params: v ? v.params : [:],
             structure: structure_features.contains(name),
             seeded: seeded, seeds: seeded ? seeds : [primary_seed],
             vocabulary: vocabulary_features.contains(name)]
        }
        def structures_file = structure_features ? structuresFile(structure_features) : null

        // Everything that would otherwise only fail after training (see
        // bin/preflight.py). Its output gates the ML inputs (through ddi_keyed
        // below) and SASA_CACHE_SHARD, so a broken database costs minutes.
        PREFLIGHT(
            db_ch.map { meta, db_path -> tuple(meta, db_path, published_features, published_files) },
            structures_file ?: [],
            params.metadata ? file(params.metadata, checkIfExists: true) : [],
            [
                combos   : feature_combos*.features,
                extracted: feature_specs.findAll { !it.structure }.collectEntries { [(it.name): it.module] },
                structure: feature_specs.findAll { it.structure }.collectEntries { [(it.name): it.module] },
            ]
        )
        preflight_ok = PREFLIGHT.out.report.map { meta, report -> tuple(meta.id, report) }

        FEATURE_EXTRACTION(
            Channel.fromList(feature_specs),
            db_ch,
            structures_file ? Channel.value(structures_file) : Channel.empty(),
            db_ch.map { meta, db_path -> tuple(meta.id, meta, db_path) }
                .join(preflight_ok)
                .map { _id, meta, db_path, _report -> tuple(meta, db_path) }
        )

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

        // Sized per database (one file per extracted feature x split, the same
        // product FEATURE_EXTRACTION fans out over), so each database's models
        // start when its own features are done rather than when the last
        // feature task of any database is. `remainder: true` because the
        // output is optional: a short group still emits when the channel closes.
        def n_feature_runs = feature_specs.sum(0) { spec -> spec.seeds.size() }
        extracted_per_db = FEATURE_EXTRACTION.out.h5
            .combine(db_ch.map { meta, _db_path -> tuple(meta.id, (meta.splits ?: ['test']).size()) }, by: 0)
            .map { db_id, seeded_h5, n_splits -> tuple(groupKey(db_id, n_splits * n_feature_runs), seeded_h5) }
            .groupTuple(remainder: true)
            // GroupKey -> plain String again, or the joins below never match.
            .map { key, files -> tuple(key.toString(), files) }

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
                // [seed, file] pairs: seed null for a seed-free file.
                def files = (extracted ?: []) + (published ?: []).collect { [null, it] }
                tuple(db_id, files.toSorted { a, b -> (a[1].name <=> b[1].name) ?: ((a[0] == null ? -1 : a[0]) <=> (b[0] == null ? -1 : b[0])) })
            }
        // emits: tuple(db_id, [[null, aacomp__train.h5], [null, identity__train.h5], ..., [null, esm3_embeddings.h5], ...])
        // (a seeded feature's files would carry their seed instead of null)

        // Key DDI outputs by db_id so we can join with feature_files_per_db.
        // Joined on PREFLIGHT: every ML input is built from this channel, so
        // no model trains on a database that failed it.
        ddi_keyed = ddi_ch.map { meta, ddi_dir -> tuple(meta.id, meta, ddi_dir) }
            .join(preflight_ok)
            .map { id, meta, ddi_dir, _report -> tuple(id, meta, ddi_dir) }

        // ---------------------------------------------------------------
        // NN / RF (per-feature singletons + one all-concat run, gated by
        // params.machine_learning_models and --skip)
        // ---------------------------------------------------------------
        nn_input_ch = nn_enabled ? ddi_keyed
            .join(feature_files_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.fromList(seeds))
            .combine(Channel.value(ml_config))
            .map { _db_id, meta, ddi_dir, feature_files, combo, seed, cfg ->
                def combo_id = combo.id
                def m = [
                    id      : "${meta.id}_neural_network_${combo_id}",
                    db      : meta.db,
                    model   : 'neural_network',
                    features: combo.features,
                    combo_id: combo_id,
                    tests   : meta.tests,
                    seed    : seed
                ]
                tuple(m, ddi_dir, comboFiles(feature_files, combo.features, meta.splits ?: ['test'], seed), cfg)
            } : Channel.empty()
        NEURAL_NETWORK(nn_input_ch)

        rf_input_ch = rf_enabled ? ddi_keyed
            .join(feature_files_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.fromList(seeds))
            .combine(Channel.value(rf_config))
            .map { _db_id, meta, ddi_dir, feature_files, combo, seed, cfg ->
                def combo_id = combo.id
                def m = [
                    id      : "${meta.id}_random_forest_${combo_id}",
                    db      : meta.db,
                    model   : 'random_forest',
                    features: combo.features,
                    combo_id: combo_id,
                    tests   : meta.tests,
                    seed    : seed
                ]
                tuple(m, ddi_dir, comboFiles(feature_files, combo.features, meta.splits ?: ['test'], seed), cfg)
            } : Channel.empty()
        RANDOM_FOREST(rf_input_ch)

        svm_input_ch = svm_enabled ? ddi_keyed
            .join(feature_files_per_db)
            .combine(Channel.fromList(feature_combos.collect { [it] }))
            .combine(Channel.fromList(seeds))
            .combine(Channel.value(svm_config))
            .map { _db_id, meta, ddi_dir, feature_files, combo, seed, cfg ->
                def combo_id = combo.id
                def m = [
                    id      : "${meta.id}_svm_${combo_id}",
                    db      : meta.db,
                    model   : 'svm',
                    features: combo.features,
                    combo_id: combo_id,
                    tests   : meta.tests,
                    seed    : seed
                ]
                tuple(m, ddi_dir, comboFiles(feature_files, combo.features, meta.splits ?: ['test'], seed), cfg)
            } : Channel.empty()
        SVM(svm_input_ch)

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
                    tests: meta.tests,
                    // Deprecated and off: primary seed only.
                    seed : primary_seed
                ]
                tuple(m, db_path, modeljson)
            }
        GRAPH_MODEL(graph_input_ch)

        // ---------------------------------------------------------------
        // Degree baseline (W-15): deg(A) x deg(B) from train's positives, a
        // model-free ranking reference on every test split. Per seed (D-W15-2)
        // so it sits in every seed's EVAL_ONE group like a model; the scores
        // are the same, only the bootstrap differs.
        // ---------------------------------------------------------------
        DEGREE_BASELINE(
            ddi_keyed
                .combine(Channel.fromList(seeds))
                .map { _id, meta, ddi_dir, seed ->
                    def m = [
                        id          : "${meta.id}_degree_baseline",
                        db          : meta.db,
                        model       : 'degree_baseline',
                        tests       : meta.tests,
                        seed        : seed,
                        ranking_only: true
                    ]
                    tuple(m, ddi_dir)
                }
        )

        // ---------------------------------------------------------------
        // Per-prediction evaluation (scatter)
        //
        // Each model emits one `predictions_<variant>.parquet` per test split
        // of its database. The variant is the fan-out axis from here on: every
        // test set is evaluated and reported as if it were its own dataset,
        // under the run label `<db>_<variant>` (bare `<db>` when the database
        // ships a single `test`).
        // ---------------------------------------------------------------
        // Each prediction carries its model's model_parameters.json: the
        // validation operating points EVAL_ONE reports exist only there. Both
        // emits come from one task with one meta, so the join is exact;
        // failOnMismatch makes a dropped prediction an error rather than a
        // short EVALUATION group. Graph models write no such file and are not
        // supported from W-7 on (they are being removed): their 2-element
        // tuples fail in the flatMap below.
        def params_file = { m, dir -> tuple(m, dir / 'model_parameters.json') }
        scored_ch = NEURAL_NETWORK.out.predictions
            .join(NEURAL_NETWORK.out.model.map(params_file), failOnMismatch: true, failOnDuplicate: true)
            .mix(RANDOM_FOREST.out.predictions
                .join(RANDOM_FOREST.out.model.map(params_file), failOnMismatch: true, failOnDuplicate: true))
            .mix(SVM.out.predictions
                .join(SVM.out.model.map(params_file), failOnMismatch: true, failOnDuplicate: true))
            .mix(GRAPH_MODEL.out.predictions)
            // Ranking-only: no model_parameters.json to stage.
            .mix(DEGREE_BASELINE.out.predictions.map { m, pred -> tuple(m, pred, []) })
            .flatMap { meta, pred, model_params ->
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
                        // Split name behind the variant (`<variant>` ->
                        // `test_<variant>`, e.g. `1_10` -> `test_1_10`;
                        // `test` -> `test`), which is how EVAL_ONE finds the
                        // right `<split>_sources.csv`.
                        split    : (meta.tests ? meta.tests[variant] : null),
                        run_label: run_label,
                        model    : model_name,
                        // The model's seed: EVAL_ONE bootstraps with it.
                        seed     : meta.seed
                    ] + (meta.ranking_only ? [ranking_only: true] : [:])
                    // (only when set, so every model's meta and task hash stays as it was)
                    // W-10: the 1:1 variant this 1:k one is compared against
                    // (null otherwise). Kept out of m_eval so ENRICHMENT's
                    // meta, and with it its task hash, stays as it was.
                    def ref_variant = meta.tests ? referenceVariant(variant, meta.tests) : null
                    def ref = [
                        reference_variant: ref_variant,
                        reference_split  : ref_variant ? meta.tests[ref_variant] : null
                    ]
                    tuple(m_eval, pf, model_params, ref)
                }
            }
        // ENRICHMENT's input: models only. The baseline's scores are products,
        // not probabilities, and `error = 1 - p` would be meaningless (W-15).
        all_predictions_ch = scored_ch
            .filter { m, _pf, _p, _r -> !m.ranking_only }
            .map { m, pf, _p, _r -> tuple(m, pf) }

        // W-10: a 1:k target also gets its 1:1 reference parquet -- from the
        // same model task (one task writes every variant), so same model and
        // seed. Non-targets get [] (nothing staged).
        def ref_preds_ch = scored_ch.map { m, pf, _p, _r -> tuple([m.db, m.model, m.seed, m.variant], pf) }
        def with_ref_ch = scored_ch
            .filter { _m, _pf, _p, r -> r.reference_variant }
            .map { m, pf, mp, r -> tuple([m.db, m.model, m.seed, r.reference_variant], m + r, pf, mp) }
            .combine(ref_preds_ch, by: 0)
            .map { _key, m, pf, mp, ref_pf -> tuple(m, pf, mp, ref_pf) }
            .mix(scored_ch
                .filter { _m, _pf, _p, r -> !r.reference_variant }
                .map { m, pf, mp, r -> tuple(m + r, pf, mp, []) })

        // Per-source accuracy needs each test DDI's provenance list, so the
        // scatter task gets its database's DDI directory alongside the
        // predictions. `combine(by: 0)` rather than `join`: one DDI dir fans out
        // to every (model x variant) prediction of that database.
        eval_one_input_ch = with_ref_ch
            .map { m, pf, mp, ref_pf -> tuple(m.db, m, pf, mp, ref_pf) }
            .combine(ddi_keyed.map { id, _meta, ddi_dir -> tuple(id, ddi_dir) }, by: 0)
            .map { _db_id, m, pf, mp, ref_pf, ddi_dir -> tuple(m, pf, ddi_dir, mp, ref_pf) }
        EVAL_ONE(eval_one_input_ch)

        // ---------------------------------------------------------------
        // Per-(DB, variant) MultiQC reduce. Group EVAL_ONE outputs by
        // (db, variant), then join back to the expanded db channel to recover
        // (meta, db_path) for the EVALUATION call.
        // ---------------------------------------------------------------
        per_model_jsons_ch = EVAL_ONE.out.metrics
            .map { meta, j -> tuple("${meta.db}::${meta.variant}".toString(), j) }
            // `remainder: true` so a group a failed-and-ignored task left short
            // still reaches EVALUATION rather than vanishing.
            .groupTuple(size: evals_per_variant, remainder: true)
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

        // ---------------------------------------------------------------
        // Enrichment (only with --metadata): regress each model's errors on
        // DDI metadata. METADATA builds one table per (db, variant) and fits
        // the model-independent `observed` target once; ENRICHMENT runs per
        // prediction. Everything lands in that variant's EVALUATION report.
        // ---------------------------------------------------------------
        if (params.metadata) {
            def metadata_dir = file(params.metadata, checkIfExists: true)
            METADATA(
                db_variant_ch
                    .map { _key, m, _db_path -> tuple(m.db, m) }
                    .combine(ddi_keyed.map { id, _meta, ddi_dir -> tuple(id, ddi_dir) }, by: 0)
                    .map { _db_id, m, ddi_dir -> tuple(m, metadata_dir, ddi_dir) }
            )
            def by_variant = { m, f -> tuple("${m.db}::${m.variant}".toString(), f) }
            // Primary seed only (W6-D1): one fit per model, labelled as such
            // in the report.
            ENRICHMENT(
                all_predictions_ch
                    .filter { m, _pf -> m.seed == primary_seed }
                    .map { m, pf -> tuple("${m.db}::${m.variant}".toString(), m, pf) }
                    .combine(METADATA.out.table.map(by_variant), by: 0)
                    .map { _key, m, pf, table -> tuple(m, table, pf) }
            )
            // Sorted like the EVAL_ONE group, for the same reasons.
            enrichment_ch = ENRICHMENT.out.json
                .map(by_variant)
                .groupTuple(size: models_per_variant, remainder: true)
                .join(METADATA.out.observed.map(by_variant))
                .map { key, jsons, observed ->
                    tuple(key, (jsons + [observed]).toSorted { a, b -> a.name <=> b.name })
                }
        } else {
            enrichment_ch = Channel.empty()
        }

        // `remainder: true` and an explicit error: a plain join dropped a
        // (db, variant) that received no predictions without a word, and with
        // none at all COMBINE_EVAL never ran and the run "succeeded" with no
        // report.
        evaluation_input_ch = db_variant_ch
            .join(per_model_jsons_ch, remainder: true)
            .map { key, meta, db_path, jsons ->
                if (meta == null || jsons == null) {
                    error("No predictions reached evaluation for ${key}: every model for this test split failed or was skipped.")
                }
                if (jsons.every { it.name.startsWith('degree_baseline.seed') }) {
                    error("Only the ranking-only baseline reached evaluation for ${key}: every model failed.")
                }
                tuple(key, meta, db_path, jsons)
            }
            // Without --metadata there is no enrichment entry: stage nothing.
            .join(enrichment_ch, remainder: true)
            // structure_features / structure_informed: the report marks
            // `<model>_<feature>` / `_all` models built on either as
            // structure-assisted (the AF3 caveat applies to the former only).
            .map { _key, meta, db_path, jsons, enrichment ->
                tuple(meta + [structure_features: structure_features.join(','),
                              structure_informed: ml_features.findAll { structure_informed.contains(it) }.join(',')],
                      db_path, jsons, enrichment ?: [])
            }
        EVALUATION(evaluation_input_ch)

    emit:
        report = EVALUATION.out.report   // tuple(meta, evaluation/)
}
