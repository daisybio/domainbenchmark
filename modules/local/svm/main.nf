process SVM {
    tag "${meta.id}"
    label 'process_gpu'  // TODO: Adapt as SVM is not GPU-accelerated -> first ask!
    // label 'process_high_memory'
    // label 'process_long'

    conda "${projectDir}/environments/ml.yml"
    container "docker://konstantinpelz/cobinet-ml:1.0.0"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json')

    output:
        tuple val(meta), path("svm_${meta.combo_id}/predictions.parquet"), emit: predictions
        tuple val(meta), path("svm_${meta.combo_id}/model/"),               emit: model
        path "versions.yml",                                                          emit: versions

    script:
        def output_base        = "svm_${meta.combo_id}"
        def output_predictions = "${output_base}/predictions.parquet"
        def output_model_dir   = "${output_base}/model"
        def max_combos_arg     = params.max_protein_combinations_per_ddi ? "--max_protein_combinations_per_ddi ${params.max_protein_combinations_per_ddi}" : ''
        """
        mkdir -p ${output_model_dir}

        svm.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions ${output_predictions} \\
            --out_model_dir ${output_model_dir} \\
            ${max_combos_arg} \\
            --seed ${params.seed}

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            scikit-learn: \$(python -c 'import sklearn; print(sklearn.__version__)')
        END_VERSIONS
        """
}

process SVM_EVALUATION {
    tag "${meta.id}"
    label 'process_high_memory'
    label 'process_long'

    conda "${projectDir}/environments/ml.yml"
    container "docker://konstantinpelz/cobinet-ml:1.0.0"

    input:
        tuple val(meta), path('DDI'), path('features/*'), path('config.json'), path(prev_results)

    output:
        tuple val(meta), path("svm_${meta.combo_id}/predictions.parquet"), emit: predictions
        path "versions.yml",                                                         emit: versions

    script:
        def output_base    = "svm_${meta.combo_id}"
        def model_dir      = "${prev_results}/svm_output/${output_base}/model"
        def max_combos_arg = params.max_protein_combinations_per_ddi ? "--max_protein_combinations_per_ddi ${params.max_protein_combinations_per_ddi}" : ''
        """
        mkdir -p ${output_base}

        svm.py \\
            --features ${meta.features.join(' ')} \\
            --features_path features/ \\
            --ddi_path DDI/ \\
            --config config.json \\
            --out_predictions ${output_base}/predictions.parquet \\
            --model_dir ${model_dir} \\
            ${max_combos_arg} \\
            --predict-only

        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            python: \$(python --version 2>&1 | sed 's/Python //')
            scikit-learn: \$(python -c 'import sklearn; print(sklearn.__version__)')
        END_VERSIONS
        """
}
