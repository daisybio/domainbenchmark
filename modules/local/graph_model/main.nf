process GRAPH_MODEL {
    tag "${meta.id}"
    label 'process_medium'

    // kgiddi_random is the permutation control. It runs the same pipeline as
    // kgiddi but on shuffled GO->protein assignments, which tends to leave more
    // biclusters surviving the chi-square filter and pushes network_expansion
    // well past `process_medium`'s 8h/6cpu/36GB envelope. On the last cluster
    // run both kgiddi_random tasks hit SLURM time limit (exit 140) on attempt
    // 1 and only finished after a retry. Start kgiddi_random at the envelope
    // that `process_medium` would have reached after two retries (3x baseline),
    // then keep the normal task.attempt scaling on top so a hard outlier can
    // still grow further.
    cpus   = { meta.model == 'kgiddi_random' ? 18                    : 6     * task.attempt }
    memory = { meta.model == 'kgiddi_random' ? 108.GB * task.attempt : 36.GB * task.attempt }
    time   = { meta.model == 'kgiddi_random' ? 24.h   * task.attempt : 8.h   * task.attempt }

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/cobinet-general:1.0.0"

    input:
        tuple val(meta), path(database), path(model_json)

    output:
        tuple val(meta), path("${meta.model}/predictions.parquet"), emit: predictions
        tuple val(meta), path("${meta.model}/model/"),               emit: model
        path "versions.yml",                                         emit: versions

    script:
        def output_model_dir   = "${meta.model}/model"
        def output_predictions = "${meta.model}/predictions.parquet"
        """
        mkdir -p ${output_model_dir}

        run_graph_models.py \\
            --database ${database} \\
            --model ${meta.model} \\
            --params ${model_json} \\
            --out_dir ${output_model_dir} \\
            --out_predictions ${output_predictions} \\
            --threads ${task.cpus}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            networkx: \$(python -c 'import networkx; print(networkx.__version__)')
        END_VERSIONS
        """
}
