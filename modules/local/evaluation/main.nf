// Scatter evaluation: one subtask per prediction file emits a tiny JSON
// (metrics + downsampled ROC/PR). The reducer consumes JSONs only — never
// holds all predictions in memory at once. Replaces the previous monolithic
// `evaluation` task that needed 300 GB and still SIGKILLed.

process EVAL_ONE {
    tag "${meta.id}"
    label 'process_eval_scatter'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(predictions)

    output:
        tuple val(meta), path("per_model/${meta.model}.json"), emit: metrics
        path "versions.yml",                                    emit: versions

    script:
        """
        mkdir -p per_model

        eval_one.py \\
            --predictions ${predictions} \\
            --model_name ${meta.model} \\
            --out per_model/${meta.model}.json

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
        END_VERSIONS
        """
}


process EVALUATION {
    tag "${meta.id}"
    label 'process_eval'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database), path(per_model_jsons), val(old_report)

    output:
        tuple val(meta), path('evaluation/'), emit: report
        path "versions.yml",                  emit: versions

    script:
        def jsons_list     = per_model_jsons instanceof java.util.List ? per_model_jsons.join(' ') : per_model_jsons
        def old_report_arg = old_report ? "--report ${old_report}" : ''
        """
        mkdir -p evaluation

        eval_multiqc.py \\
            --database ${database} \\
            --per_model_metrics ${jsons_list} \\
            --out_dir evaluation/ \\
            ${old_report_arg}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            multiqc: \$(multiqc --version 2>&1 | sed -e 's/.*version //' -e 's/[, ].*//')
        END_VERSIONS
        """
}


process COMBINE_EVAL {
    tag "${meta.id}"
    label 'process_eval'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

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
}
