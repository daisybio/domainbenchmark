process MACHINE_LEARNING {
    tag "${meta.id}"
    label 'process_gpu'

    conda "${projectDir}/environments/ml.yml"
    container "docker://konstantinpelz/cobinet-ml:1.0.0"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json')

    output:
        tuple val(meta), path("machine_learning_${meta.features.join('_')}/predictions.parquet"), emit: predictions
        tuple val(meta), path("machine_learning_${meta.features.join('_')}/model/"),               emit: model
        path "versions.yml",                                                                        emit: versions

    script:
        def output_base        = "machine_learning_${meta.features.join('_')}"
        def output_predictions = "${output_base}/predictions.parquet"
        def output_model_dir   = "${output_base}/model"
        """
        mkdir -p ${output_model_dir}

        machine_learning.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions ${output_predictions} \\
            --out_model_dir ${output_model_dir} \\
            --seed ${params.seed}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            torch: \$(python -c 'import torch; print(torch.__version__)')
            scikit-learn: \$(python -c 'import sklearn; print(sklearn.__version__)')
        END_VERSIONS
        """
}

process MACHINE_LEARNING_EVALUATION {
    tag "${meta.id}"
    label 'process_high_memory'
    label 'process_long'

    errorStrategy 'ignore'

    conda "${projectDir}/environments/ml.yml"
    container "docker://konstantinpelz/cobinet-ml:1.0.0"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json'), path(prev_results)

    output:
        tuple val(meta), path("machine_learning_${meta.features.join('_')}/predictions.parquet"), emit: predictions
        path "versions.yml",                                                                      emit: versions

    script:
        def output_base = "machine_learning_${meta.features.join('_')}"
        def model_dir   = "${prev_results}/ml_output/${output_base}/model"
        """
        mkdir -p ${output_base}

        machine_learning.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions ${output_base}/predictions.parquet \\
            --model_dir ${model_dir} \\
            --predict-only

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            torch: \$(python -c 'import torch; print(torch.__version__)')
            scikit-learn: \$(python -c 'import sklearn; print(sklearn.__version__)')
        END_VERSIONS
        """
}
