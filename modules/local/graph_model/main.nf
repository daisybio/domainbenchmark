process GRAPH_MODEL {
    tag "${meta.id}"
    // Dedicated label `process_graph` (18cpu / 108GB / 24h, scaled by
    // task.attempt). The previous setup used `process_medium` plus inline
    // cpus/memory/time overrides in this process body, but `withLabel`
    // directives in conf/base.config take precedence over per-process
    // directives in Nextflow, so the inline overrides were silently dropped
    // and every graph task ran in `process_medium`'s 6cpu/36GB/8h slot —
    // kgiddi and kgiddi_random both hit SLURM exit 140 on the last run.
    label 'process_graph'

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
