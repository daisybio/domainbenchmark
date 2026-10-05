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
// `resolve_feature_file` reads both layouts, so a pre-existing per-feature
// directory tree still works.
//
// The split set comes from `meta.splits` — `train`, `validation`, and one or
// more `test*` — so a database shipping both `test_balanced` and
// `test_realistic` produces an h5 for each while training data is extracted
// once.
//
// SASA cache: when a structures file is given (params.structures) AND
// params.machine_learning_features_structure is non-empty, SASA_CACHE runs ONCE
// over every split of every database and writes `sasa_cache.h5`. Structure
// features then read per-residue SASA from it (via $SASA_CACHE_H5) instead of
// each re-parsing the PDB and re-running Shrake-Rupley. Only the structure
// features wait for the cache; all other features start immediately.

// Splits to extract for one database. With --train_only only `train` is kept
// (e.g. for mRMR, which never looks at validation/test), so everything keyed on
// the other splits -- feature tasks and the SASA cache -- is skipped. Task
// metadata for the train splits is unchanged, so -resume still hits the cache
// of an earlier full run.
def splitsFor(db_meta) {
    def splits = db_meta.splits ?: ['test']
    params.train_only ? splits.findAll { it == 'train' } : splits
}

process SASA_CACHE {
    tag "sasa_cache"
    label 'sasa_cache'

    conda "${projectDir}/environments/general.yml"
    // Same image the structure features run in (Bio.PDB + the `features` package).
    container 'docker.io/konstantinpelz/domainbenchmark-gpu:1.0.1'

    input:
        path(dbs, stageAs: 'db?/*')   // every split sqlite3 file, each in its own dbN/ (avoids name clashes)
        path(structure_h5, stageAs: 'structures.h5')

    output:
        path('sasa_cache.h5'), emit: cache

    script:
        """
        build_sasa_cache.py \\
            --db db*/* \\
            --struct structures.h5 \\
            --out sasa_cache.h5 \\
            --procs ${task.cpus}
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
    container { def gpu_features = params.machine_learning_features_structure.split(',')*.trim()
        meta.module in gpu_features
        ? 'docker.io/konstantinpelz/domainbenchmark-gpu:1.0.1'
        : 'docker.io/konstantinpelz/domainbenchmark-general:1.0.0' 
    }

    input:
        tuple val(meta), path(database_dir)
        path(structure_h5, stageAs: 'structures.h5')  // may be [] -- see FEATURE_EXTRACTION below
        path(sasa_cache, stageAs: 'sasa_cache.h5')    // [] for non-structure features / when no cache is built

    output:
        tuple val(meta), path("${meta.feature}__${meta.dataset}.h5"), emit: h5, optional: true

    script:
        def out          = "${meta.feature}__${meta.dataset}.h5"
        def feature_name = meta.feature
        def module_name = meta.module
        def feature_params = meta.params ?: [:]
        def dataset      = meta.dataset
        // structure_h5 is [] (empty list) when params.structures is unset --
        // Nextflow stages nothing and the input list is empty, so we only add
        // the flag when there is actually a file present.
        def struct_arg = (structure_h5.size() > 0) ? "--struct_file ${structure_h5}" : ""
        // Same trick for the SASA cache: only export the path when it was staged.
        // File locking is disabled because many tasks open the same file read-only
        // on a shared filesystem.
        def cache_env = (sasa_cache.size() > 0)
            ? "export SASA_CACHE_H5=\"\$PWD/sasa_cache.h5\"; export HDF5_USE_FILE_LOCKING=FALSE"
            : ""

        if (database_dir.isFile()) {
            if (dataset != 'test') {
                """
                echo "Single-file database ${database_dir} has no '${dataset}' split — nothing to extract."
                """
            } else {
                """
                ${cache_env}
                extract_features.py \\
                    --db ${database_dir} \\
                    --feature ${feature_name} \\
                    --module ${module_name} \\
                    --params '${groovy.json.JsonOutput.toJson(feature_params)}' \\
                    --out ${out} \\
                    ${struct_arg} \\
                    --seed ${params.seed}
                """
            }
        } else {
            """
            ${cache_env}
            if [ -f "${database_dir}/${dataset}.sqlite3" ]; then
                extract_features.py \\
                    --db ${database_dir}/${dataset}.sqlite3 \\
                    --feature ${feature_name} \\
                    --module ${module_name} \\
                    --params '${groovy.json.JsonOutput.toJson(feature_params)}' \\
                    --out ${out} \\
                    ${struct_arg} \\
                    --seed ${params.seed}
            else
                echo "Database ${database_dir} has no ${dataset}.sqlite3 — nothing to extract."
            fi
            """
        }

    stub:
        """
        touch ${meta.feature}__${meta.dataset}.h5
        """
}


workflow FEATURE_EXTRACTION {
    take:
        feature_ch     // queue of feature names (String)
        db_ch          // channel: tuple(meta, db_path)  — multi-DB capable
        struct_file_ch // value channel: single structures.h5 File, or Channel.value([]) if params.structures unset

    main:
        // Structure features = the ones the container selection already treats
        // as structure features.
        def struct_modules = (params.machine_learning_features_structure ?: '')
            .toString().split(',')*.trim().findAll { it }
        def build_cache = params.structures && struct_modules

        per_task = db_ch
            .combine(feature_ch)
            .flatMap { db_meta, db_path, feat ->
                splitsFor(db_meta).collect { ds ->
                    def m = [
                        id     : "${db_meta.id}_${feat.name}_${ds}",
                        db     : db_meta.id,
                        feature: feat.name,
                    module : feat.module,
                    params : groovy.json.JsonOutput.toJson(feat.params ?: [:]),
                        dataset: ds
                    ]
                    tuple(m, db_path)
                }
            }

        // struct_file_ch is a single-value channel; .combine broadcasts it
        // onto every task without changing the fan-out shape.
        per_task_with_struct = per_task.combine(struct_file_ch) //.view { "PER_TASK_WITH_STRUCT: $it" }

        if (build_cache) {
            // Every split sqlite3 file of every database, collected into one list.
            // Single-file databases are passed as-is.
            sasa_dbs = db_ch
                .flatMap { db_meta, db_path ->
                    def p = file(db_path)
                    p.isFile()
                        ? [p]
                        : splitsFor(db_meta).collect { ds -> file("${p}/${ds}.sqlite3") }.findAll { it.exists() }
                }
                .toSortedList { a, b -> a.toString() <=> b.toString() }

            SASA_CACHE(sasa_dbs, struct_file_ch)
            sasa_cache_ch = SASA_CACHE.out.cache.first()

            // Structure features wait for the cache ...
            with_cache = per_task_with_struct
                .filter { it[0].module in struct_modules }
                .combine(sasa_cache_ch)
            // ... everything else carries an empty placeholder and starts right away.
            without_cache = per_task_with_struct
                .filter { !(it[0].module in struct_modules) }
                .map { m, db_path, struct -> tuple(m, db_path, struct, []) }

            per_task_full = with_cache.mix(without_cache)
        } else {
            per_task_full = per_task_with_struct
                .map { m, db_path, struct -> tuple(m, db_path, struct, []) }
        }

        per_split = FEATURE_EXTRACTION_ONE(
            per_task_full.map { m, db_path, _struct, _cache -> tuple(m, db_path) },
            per_task_full.map { _m, _db_path, struct, _cache -> struct },
            per_task_full.map { _m, _db_path, _struct, cache -> cache }
        )

    emit:
        h5 = per_split.h5.map { meta, h5 -> tuple(meta.db, h5) }
}
