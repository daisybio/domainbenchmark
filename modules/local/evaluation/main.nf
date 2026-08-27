// Scatter evaluation: one subtask per prediction file emits a tiny JSON
// (metrics + downsampled ROC/PR). The reducer consumes JSONs only — never
// holds all predictions in memory at once. Replaces the previous monolithic
// `evaluation` task that needed 300 GB and still SIGKILLed.

process EVAL_ONE {
    tag "${meta.id}"
    label 'process_eval_scatter'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(predictions), path(ddi_dir)

    output:
        tuple val(meta), path("per_model/${meta.model}.json"), emit: metrics
        path "versions.yml",                                    emit: versions

    script:
        // `<split>_sources.csv` carries each test DDI's comma-joined provenance
        // list, which is what turns the predictions into per-source accuracy.
        // Passed as a path, not a flag: a database whose splitter predates the
        // `source` column simply has no such file and the block is skipped.
        def sources_arg = meta.split ? "--sources ${ddi_dir}/${meta.split}_sources.csv" : ''
        """
        mkdir -p per_model

        eval_one.py \\
            --predictions ${predictions} \\
            --model_name ${meta.model} \\
            ${sources_arg} \\
            --out per_model/${meta.model}.json

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
        END_VERSIONS
        """

    stub:
        """
        mkdir -p per_model
        echo '{}' > per_model/${meta.model}.json

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            stub: true
        END_VERSIONS
        """
}


process EVALUATION {
    tag "${meta.id}"
    label 'process_eval'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database), path(per_model_jsons), val(old_report)

    output:
        tuple val(meta), path('evaluation/'), emit: report
        path "versions.yml",                  emit: versions

    script:
        def jsons_list     = per_model_jsons instanceof java.util.List ? per_model_jsons.join(' ') : per_model_jsons
        def old_report_arg = old_report ? "--report ${old_report}" : ''
        // One report per (database, test variant): the train and validation
        // splits are shared, only the test set differs.
        def test_split     = meta.split ?: 'test'
        """
        mkdir -p evaluation

        eval_multiqc.py \\
            --database ${database} \\
            --per_model_metrics ${jsons_list} \\
            --db_name ${meta.run_label ?: meta.id} \\
            --test_split ${test_split} \\
            --out_dir evaluation/ \\
            ${old_report_arg}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            multiqc: \$(multiqc --version 2>&1 | sed -e 's/.*version //' -e 's/[, ].*//')
        END_VERSIONS
        """

    stub:
        """
        mkdir -p evaluation
        echo "${meta.id} (${meta.split})" > evaluation/evaluation.html

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            stub: true
        END_VERSIONS
        """
}


process COMBINE_EVAL {
    tag "${meta.id}"
    label 'process_eval'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        // Every per-DB EVALUATION emits a dir literally named `evaluation/`, so
        // staging them with a shared target collides ("multiple input files for
        // each of the following file names: reports/evaluation"). `stageAs:
        // 'src/?/*'` gives each input its own numbered parent dir while
        // preserving the original `evaluation` name. `ids` runs in parallel with
        // `eval_reports` (same order), letting us symlink each staged dir into
        // `reports/<db_id>/evaluation` -- required by combine_eval.py, which
        // derives db_name from the report dir's parent.
        tuple val(meta), path(eval_reports, stageAs: 'src/?/*'), val(ids)

    output:
        tuple val(meta), path('evaluation/'), emit: combined_report
        path "versions.yml",                  emit: versions

    script:
        def ids_list = ids instanceof java.util.List ? ids : [ids]
        def ids_bash = ids_list.collect { "'${it}'" }.join(' ')
        """
        mkdir -p reports
        ids=(${ids_bash})
        n=\${#ids[@]}
        for (( i=1; i<=n; i++ )); do
            db_id="\${ids[\$((i-1))]}"
            mkdir -p "reports/\${db_id}"
            ln -s "../../src/\${i}/evaluation" "reports/\${db_id}/evaluation"
        done

        mkdir -p evaluation

        combine_eval.py \\
            --reports reports/*/evaluation \\
            --out_dir evaluation

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
        END_VERSIONS
        """

    stub:
        def ids_list = ids instanceof java.util.List ? ids : [ids]
        """
        mkdir -p evaluation
        printf '%s\\n' ${ids_list.collect { "'${it}'" }.join(' ')} > evaluation/ddi_report.html

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            stub: true
        END_VERSIONS
        """
}
