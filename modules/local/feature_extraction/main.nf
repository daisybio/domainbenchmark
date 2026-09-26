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
        def struct_arg   = (structure_h5 && !(structure_h5 instanceof List && structure_h5.isEmpty()))
            ? "--struct-file ${structure_h5}"
            : ""

        if (database_dir.isFile()) {
            if (dataset != 'test') {
                """
                echo "Single-file database ${database_dir} has no '${dataset}' split — nothing to extract."
                """
            } else {
                """
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
        per_task = db_ch
            .combine(feature_ch)
            .flatMap { db_meta, db_path, feat ->
                (db_meta.splits ?: ['test']).collect { ds ->
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
        per_task_with_struct = per_task.combine(struct_file_ch)

        per_split = FEATURE_EXTRACTION_ONE(
            per_task_with_struct.map { m, db_path, _struct -> tuple(m, db_path) },
            per_task_with_struct.map { _m, _db_path, struct -> struct }
        )

    emit:
        h5 = per_split.h5.map { meta, h5 -> tuple(meta.db, h5) }
}