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
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database_dir)

    output:
        // optional: a split absent from this database produces no file at all.
        // The previous zero-byte placeholder existed only so the stager could
        // skip it; with no stager, an empty .h5 would just break h5py.
        tuple val(meta), path("${meta.feature}__${meta.dataset}.h5"), emit: h5, optional: true

    script:
        def out          = "${meta.feature}__${meta.dataset}.h5"
        def feature_name = meta.feature
        def dataset      = meta.dataset

        if (database_dir.isFile()) {
            // Single-file db inputs only support a 'test' split.
            if (dataset != 'test') {
                """
                echo "Single-file database ${database_dir} has no '${dataset}' split — nothing to extract."
                """
            } else {
                """
                extract_features.py \\
                    --db ${database_dir} \\
                    --feature ${feature_name} \\
                    --out ${out} \\
                    --seed ${params.seed}
                """
            }
        } else {
            """
            if [ -f "${database_dir}/${dataset}.sqlite3" ]; then
                extract_features.py \\
                    --db ${database_dir}/${dataset}.sqlite3 \\
                    --feature ${feature_name} \\
                    --out ${out} \\
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

    main:
        // Build per-(db, feature, split) tasks. The split list is per-database
        // (`meta.splits`), not a pipeline constant.
        per_task = db_ch
            .combine(feature_ch)
            .flatMap { db_meta, db_path, feat ->
                (db_meta.splits ?: ['test']).collect { ds ->
                    def m = [
                        id     : "${db_meta.id}_${feat}_${ds}",
                        db     : db_meta.id,
                        feature: feat,
                        dataset: ds
                    ]
                    tuple(m, db_path)
                }
            }

        per_split = FEATURE_EXTRACTION_ONE(per_task)

    emit:
        // tuple(db_id, h5) — one item per (db, feature, split). The caller
        // groupTuple()s by db_id to get every feature file of one database.
        h5 = per_split.h5.map { meta, h5 -> tuple(meta.db, h5) }
}
