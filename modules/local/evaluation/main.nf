// Scatter evaluation: one subtask per prediction file emits a tiny JSON
// (metrics + downsampled ROC/PR). The reducer consumes JSONs only — never
// holds all predictions in memory at once. Replaces the previous monolithic
// `evaluation` task that needed 300 GB and still SIGKILLed.

process EVAL_ONE {
    tag "${meta.id}"
    label 'process_eval_scatter'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/cobinet-general:1.0.0"

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
    container "docker://konstantinpelz/cobinet-general:1.0.0"

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
    container "docker://konstantinpelz/cobinet-general:1.0.0"

    input:
        tuple val(meta), path(eval_reports, stageAs: 'reports/*')

    output:
        tuple val(meta), path('evaluation/'), emit: combined_report
        path "versions.yml",                  emit: versions

    script:
        def reports_list = eval_reports instanceof java.util.List ? eval_reports.join(' ') : eval_reports
        """
        mkdir -p evaluation

        combine_eval.py \\
            --reports ${reports_list} \\
            --out_dir evaluation

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
        END_VERSIONS
        """
}
