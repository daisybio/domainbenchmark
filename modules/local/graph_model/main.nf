process GRAPH_MODEL {
    tag "${meta.id}"
    label 'process_medium'

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
            --out_predictions ${output_predictions}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            networkx: \$(python -c 'import networkx; print(networkx.__version__)')
        END_VERSIONS
        """
}
