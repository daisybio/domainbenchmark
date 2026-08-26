process GRAPH_MODEL {
    tag "${meta.id}"
    label 'process_high'

    conda "${projectDir}/environments/general.yml"
    container "docker://konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database), path(model_json)

    output:
        tuple val(meta), path("${meta.model}/predictions_*.parquet"), emit: predictions
        tuple val(meta), path("${meta.model}/model/"),                emit: model
        path "versions.yml",                                          emit: versions

    script:
        def output_model_dir = "${meta.model}/model"
        // Trained once on the train split, scored against every test split of
        // this database -- one predictions_<variant>.parquet per test set.
        def test_splits      = meta.tests.values().join(' ')
        """
        mkdir -p ${output_model_dir}

        run_graph_models.py \\
            --database ${database} \\
            --model ${meta.model} \\
            --params ${model_json} \\
            --out_dir ${output_model_dir} \\
            --out_predictions_dir ${meta.model} \\
            --test_splits ${test_splits} \\
            --threads ${task.cpus}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            networkx: \$(python -c 'import networkx; print(networkx.__version__)')
        END_VERSIONS
        """

    stub:
        def variants = meta.tests.keySet().join(' ')
        """
        mkdir -p ${meta.model}/model
        for v in ${variants}; do
            touch ${meta.model}/predictions_\${v}.parquet
        done
        touch ${meta.model}/model/model.txt

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            stub: true
        END_VERSIONS
        """
}
