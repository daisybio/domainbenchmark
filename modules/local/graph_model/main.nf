process GRAPH_MODEL {
    tag "${meta.id}"
    label 'process_medium'

    // Graph models all share one fat envelope. kgiddi spends ~5h in
    // build_ddi_network alone before network_expansion runs, and kgiddi_random
    // (permutation control) is heavier still because shuffled GO->protein
    // assignments leave more biclusters surviving the chi-square filter. The
    // prior conditional override on meta.model did not take effect at submit
    // time, so every graph task ran with `process_medium`'s 6cpu/36GB/8h slot
    // and kgiddi, kgiddi_random both hit SLURM exit 140 on the last run.
    // Size for the slowest model unconditionally; ddiparsimony finishes in
    // well under the cap so the over-allocation is cheap.
    cpus   = 18
    memory = { 108.GB * task.attempt }
    time   = { 24.h   * task.attempt }

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
