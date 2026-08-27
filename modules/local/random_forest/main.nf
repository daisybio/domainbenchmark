process RANDOM_FOREST {
    tag "${meta.id}"
    label 'process_gpu'

    conda "${projectDir}/environments/ml.yml"
    container "docker.io/konstantinpelz/domainbenchmark-gpu:1.0.0"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json')

    output:
        tuple val(meta), path("random_forest_${meta.combo_id}/predictions_*.parquet"), emit: predictions
        tuple val(meta), path("random_forest_${meta.combo_id}/model/"),               emit: model

    script:
        def output_base        = "random_forest_${meta.combo_id}"
        def output_model_dir   = "${output_base}/model"
        // One trained model, scored against every test split of this database
        // (test_balanced + test_realistic, or a single test).
        def test_splits        = meta.tests.values().join(' ')
        // Without --allow_cpu the script exits 140 when its CUDA smoke test
        // fails, so Nextflow retries on a healthy GPU node.
        def allow_cpu          = params.allow_cpu_ml ? '--allow_cpu' : ''
        """
        mkdir -p ${output_model_dir}

        random_forest.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions_dir ${output_base} \\
            --test_splits ${test_splits} \\
            --out_model_dir ${output_model_dir} \\
            --seed ${params.seed} ${allow_cpu}
        """

    stub:
        def output_base = "random_forest_${meta.combo_id}"
        def variants    = meta.tests.keySet().join(' ')
        """
        mkdir -p ${output_base}/model
        for v in ${variants}; do
            touch ${output_base}/predictions_\${v}.parquet
        done
        touch ${output_base}/model/model_parameters.json
        """
}

process RANDOM_FOREST_EVALUATION {
    tag "${meta.id}"
    label 'process_high_memory'
    label 'process_long'

    conda "${projectDir}/environments/ml.yml"
    container "docker.io/konstantinpelz/domainbenchmark-gpu:1.0.0"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json'), path(prev_results)

    output:
        tuple val(meta), path("random_forest_${meta.combo_id}/predictions_*.parquet"), emit: predictions

    script:
        def output_base    = "random_forest_${meta.combo_id}"
        def model_dir      = "${prev_results}/rf_output/${output_base}/model"
        def test_splits    = meta.tests.values().join(' ')
        """
        mkdir -p ${output_base}

        random_forest.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions_dir ${output_base} \\
            --test_splits ${test_splits} \\
            --model_dir ${model_dir} \\
            --predict-only
        """

    stub:
        def output_base = "random_forest_${meta.combo_id}"
        def variants    = meta.tests.keySet().join(' ')
        """
        mkdir -p ${output_base}
        for v in ${variants}; do
            touch ${output_base}/predictions_\${v}.parquet
        done
        """
}
