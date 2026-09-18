process SVM {
    tag "${meta.id}"
    label 'process_gpu'  // TODO: Adapt as SVM is not GPU-accelerated -> first ask!
    // label 'process_high_memory'
    // label 'process_long'

    conda "${projectDir}/environments/ml.yml"
    container "docker.io/konstantinpelz/domainbenchmark-gpu:1.0.1"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json')

    output:
        tuple val(meta), path("svm_${meta.combo_id}/predictions_*.parquet"), emit: predictions
        tuple val(meta), path("svm_${meta.combo_id}/model/"),               emit: model

    script:
        def output_base        = "svm_${meta.combo_id}"
        def output_model_dir   = "${output_base}/model"
        def test_splits        = meta.tests.values().join(' ')

        """
        mkdir -p ${output_model_dir}

        svm.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions_dir ${output_base} \\
            --out_model_dir ${output_model_dir} \\
            --test_splits ${test_splits} \\
            --seed ${params.seed}
        """

    stub:
        def output_base = "svm_${meta.combo_id}"
        def variants    = meta.tests.keySet().join(' ')
        """
        mkdir -p ${output_base}/model
        for v in ${variants}; do
            touch ${output_base}/predictions_\${v}.parquet
        done
        touch ${output_base}/model/model_parameters.json
        """
}

process SVM_EVALUATION {
    tag "${meta.id}"
    label 'process_high_memory'
    label 'process_long'

    conda "${projectDir}/environments/ml.yml"
    container "docker.io/konstantinpelz/domainbenchmark-gpu:1.0.1"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json'), path(prev_results)

    output:
        tuple val(meta), path("svm_${meta.combo_id}/predictions_*.parquet"), emit: predictions

    script:
        def output_base    = "svm_${meta.combo_id}"
        def model_dir      = "${prev_results}/svm_output/${output_base}/model"
        def test_splits    = meta.tests.values().join(' ')
        """
        mkdir -p ${output_base}

        svm.py \\
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
        def output_base = "svm_${meta.combo_id}"
        def variants    = meta.tests.keySet().join(' ')
        """
        mkdir -p ${output_base}
        for v in ${variants}; do
            touch ${output_base}/predictions_\${v}.parquet
        done
        """
}