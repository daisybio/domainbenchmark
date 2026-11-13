// Parallelise feature extraction across a database's sqlite splits.
//
// The `FEATURE_EXTRACTION` (lower-case) inner workflow fans out
// (feature × split) into independent jobs. Each job writes exactly one
// `<feature>__<split>.h5`, and NN / RF stage the whole set flat into
// `features/` — the filename carries the layout, so nothing in between has to
// materialise a `features/<feature>/<split>.h5` tree.
//
// There used to be a STAGE_FEATURE_DIR process building that tree with `cp`. It
// cost one scheduled job per (db, feature) to copy a handful of files, and on a
// node that cannot loop-mount the SIF, singularity unpacked the whole image to
// a temporary sandbox first — which outlived the label's walltime and killed the
// run with exit 140 after three attempts. `bin/machine_learning.py`'s
// `resolve_feature_file` reads `<feature>__<split>.h5` from here and
// `<feature>.h5` (a published per-run file, via VERIFY_EMBEDDINGS); the
// per-feature directory tree is no longer accepted.
//
// The split set comes from `meta.splits` — `train`, `validation`, and one or
// more `test*` — so a database shipping both `test_balanced` and
// `test_realistic` produces an h5 for each while training data is extracted
// once.
//
// Structure features (params.machine_learning_features_structure) read the
// AF3 complexes in domainsplit's structures.h5. Most of them need per-residue
// SASA, so SASA_CACHE_SHARD / MERGE_SASA_CACHE run Shrake-Rupley once per
// complex over every database's splits and every structure feature reads it
// from the merged cache. Only structure features wait for it; the others start
// at once.

process SASA_CACHE_SHARD {
    tag "sasa_cache_${shard_id}"
    label 'sasa_cache'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        path(db_dirs, stageAs: 'db?/*')
        path(structures, stageAs: 'structures.h5')
        each shard_id

    output:
        path("sasa_cache.shard_${shard_id}.h5"), emit: shard

    script:
        """
        build_sasa_cache.py \\
            --db db*/*/*.sqlite3 \\
            --struct structures.h5 \\
            --out sasa_cache.shard_${shard_id}.h5 \\
            --procs ${task.cpus} \\
            --shard_id ${shard_id} \\
            --n_shards ${params.sasa_cache_shards}
        """

    stub:
        """
        touch sasa_cache.shard_${shard_id}.h5
        """
}

process MERGE_SASA_CACHE {
    tag "sasa_cache_merge"
    label 'process_low'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        path(shards, stageAs: 'shards/*')

    output:
        path('sasa_cache.h5'), emit: cache

    script:
        """
        merge_sasa_cache.py --shards shards/* --out sasa_cache.h5
        """

    stub:
        """
        touch sasa_cache.h5
        """
}

process FEATURE_EXTRACTION_ONE {
    tag "${meta.id}"
    label 'feature_extraction'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        // `structures` / `sasa_cache` are [] for a non-structure feature.
        tuple val(meta), path(database_dir), path(structures, stageAs: 'structures.h5'), path(sasa_cache, stageAs: 'sasa_cache.h5')

    output:
        // optional: a split absent from this database produces no file at all.
        // The previous zero-byte placeholder existed only so the stager could
        // skip it; with no stager, an empty .h5 would just break h5py.
        tuple val(meta), path("${meta.feature}__${meta.dataset}.h5"), emit: h5, optional: true

    script:
        def out          = "${meta.feature}__${meta.dataset}.h5"
        def feature_name = meta.feature
        def dataset      = meta.dataset
        def module_arg   = meta.module != feature_name ? "--module ${meta.module}" : ''
        def params_arg   = meta.params ? "--params '${groovy.json.JsonOutput.toJson(meta.params)}'" : ''
        def struct_arg   = structures ? '--structures structures.h5 --sasa_cache sasa_cache.h5' : ''

        // `database_dir` is always a directory -- datasetTuple() lists it via
        // discoverSplits() -- so there is no single-file branch.
        """
        if [ -f "${database_dir}/${dataset}.sqlite3" ]; then
            extract_features.py \\
                --db ${database_dir}/${dataset}.sqlite3 \\
                --feature ${feature_name} ${module_arg} ${params_arg} ${struct_arg} \\
                --out ${out} \\
                --seed ${params.seed}
        else
            echo "Database ${database_dir} has no ${dataset}.sqlite3 — nothing to extract."
        fi
        """

    stub:
        """
        touch ${meta.feature}__${meta.dataset}.h5
        """
}


workflow FEATURE_EXTRACTION {
    take:
        feature_ch     // queue of feature specs: [name, module, params (Map), structure (bool)]
        db_ch          // channel: tuple(meta, db_path)  — multi-DB capable
        structures     // value channel: structures.h5, or Channel.empty() without structure features

    main:
        // Build per-(db, feature, split) tasks. The split list is per-database
        // (`meta.splits`), not a pipeline constant.
        per_task = db_ch
            .combine(feature_ch)
            .flatMap { db_meta, db_path, feat ->
                (db_meta.splits ?: ['test']).collect { ds ->
                    def m = [
                        id       : "${db_meta.id}_${feat.name}_${ds}",
                        db       : db_meta.id,
                        feature  : feat.name,
                        module   : feat.module,
                        params   : feat.params,
                        structure: feat.structure,
                        dataset  : ds
                    ]
                    tuple(m, db_path)
                }
            }

        // One SASA cache over every database that runs a structure feature.
        // `collect` waits for the whole db channel, so the cache is a single
        // value every structure task combines with.
        sasa_cache = MERGE_SASA_CACHE(
            SASA_CACHE_SHARD(
                db_ch.map { _meta, db_path -> db_path }.collect(sort: true),
                structures,
                (0..<params.sasa_cache_shards).toList()
            ).shard.collect(sort: true)
        ).cache

        structure_tasks = per_task
            .filter { m, _db -> m.structure }
            .combine(structures)
            .combine(sasa_cache)
        plain_tasks = per_task
            .filter { m, _db -> !m.structure }
            .map { m, db_path -> tuple(m, db_path, [], []) }

        per_split = FEATURE_EXTRACTION_ONE(plain_tasks.mix(structure_tasks))

    emit:
        // tuple(db_id, h5) — one item per (db, feature, split). The caller
        // groupTuple()s by db_id to get every feature file of one database.
        h5 = per_split.h5.map { meta, h5 -> tuple(meta.db, h5) }
}
