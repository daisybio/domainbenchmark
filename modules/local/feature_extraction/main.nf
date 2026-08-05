// Parallelise feature extraction across train/test/optimization splits.
//
// The `FEATURE_EXTRACTION` (lower-case) inner workflow fans out
// (feature × split) into independent jobs and stages the resulting h5
// files into the per-feature directory layout that ML / RF / graph_model
// already expect (`features/<feature>/<train|test|optimization>.h5`).
// Downstream channel signatures are unchanged — only the granularity
// of work changes.

process FEATURE_EXTRACTION_ONE {
    tag "${meta.id}"
    label 'feature_extraction'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database_dir)

    output:
        tuple val(meta), path("${meta.dataset}.h5"), emit: h5
        path "versions.yml",                         emit: versions

    script:
        def out = "${meta.dataset}.h5"
        def feature_name = meta.feature
        def dataset      = meta.dataset

        def script_body
        if (database_dir.isFile()) {
            // Single-file db inputs only support a 'test' split.
            if (dataset != 'test') {
                script_body = ": > ${out}"
            } else {
                script_body = """extract_features.py \\
                    --db ${database_dir} \\
                    --feature ${feature_name} \\
                    --out ${out}"""
            }
        } else {
            script_body = """if [ -f "${database_dir}/${dataset}.sqlite3" ]; then
                extract_features.py \\
                    --db ${database_dir}/${dataset}.sqlite3 \\
                    --feature ${feature_name} \\
                    --out ${out}
            else
                # Split missing in this database — emit zero-byte placeholder
                # so the stager can skip it without breaking groupTuple sizing.
                : > ${out}
            fi"""
        }

        """
        ${script_body}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
        END_VERSIONS
        """
}


process STAGE_FEATURE_DIR {
    tag "${meta.id}"
    label 'feature_stage'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(h5_files)

    output:
        tuple val(meta), path("${meta.db}/${meta.feature}"), emit: feature_dir
        path "versions.yml",                                 emit: versions

    script:
        def h5_list = h5_files instanceof java.util.List ? h5_files.join(' ') : h5_files
        """
        mkdir -p ${meta.db}/${meta.feature}
        for f in ${h5_list}; do
            # Skip empty placeholders for splits that didn't exist in this db.
            if [ -s "\$f" ]; then
                cp -L "\$f" ${meta.db}/${meta.feature}/\$(basename "\$f")
            fi
        done

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            posix_cp: \$(cp --version 2>&1 | head -n1 | awk '{print \$NF}')
        END_VERSIONS
        """
}


workflow FEATURE_EXTRACTION {
    take:
        feature_ch     // queue of feature names (String)
        db_ch          // channel: tuple(meta, db_path)  — multi-DB capable

    main:
        ds_ch = Channel.from('train', 'test', 'optimization')

        // Build per-(db, feature, split) tasks. Each task gets a unique meta
        // so groupTuple can re-cluster downstream by (db, feature).
        per_task = db_ch
            .combine(feature_ch)
            .combine(ds_ch)
            .map { db_meta, db_path, feat, ds ->
                def m = [
                    id     : "${db_meta.id}_${feat}_${ds}",
                    db     : db_meta.id,
                    feature: feat,
                    dataset: ds
                ]
                tuple(m, db_path)
            }

        per_split = FEATURE_EXTRACTION_ONE(per_task)

        // Group per-split outputs back to (db, feature). The stripped meta
        // (no dataset key) is what flows downstream into ML/RF.
        grouped = per_split.h5
            .map { meta, h5 ->
                def m = [
                    id     : "${meta.db}_${meta.feature}",
                    db     : meta.db,
                    feature: meta.feature
                ]
                tuple(m.id, m, h5)
            }
            .groupTuple()
            .map { _id, metas, h5s -> tuple(metas[0], h5s) }

        staged = STAGE_FEATURE_DIR(grouped)

    emit:
        feature_dir = staged.feature_dir
        versions    = staged.versions.mix(per_split.versions)
}
