// Metadata enrichment: which DDI properties explain what a model predicts.
// METADATA runs once per (database, test variant) and also fits the
// model-independent `observed` target; ENRICHMENT runs once per prediction
// file. Both JSONs go to that variant's EVALUATION (--enrichment).

process METADATA {
    tag "${meta.id}"
    label 'process_eval_metadata_creation'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        tuple val(meta), path(metadata_dir), path(ddi_dir)

    output:
        tuple val(meta), path("metadata_table.csv"), emit: table
        tuple val(meta), path("observed.json"),      emit: observed

    script:
        """
        create_metadata.py \\
            --metadata ${metadata_dir} \\
            --instances ${ddi_dir}/${meta.split}_instances.csv \\
            --threads ${task.cpus} \\
            --out_table metadata_table.csv \\
            --out_observed observed.json
        """

    stub:
        """
        echo 'domain_a,domain_b,label' > metadata_table.csv
        echo '{"counts": {"n_ddis": 0, "n_with_metadata": 0, "n_complete": 0, "n_missing_by_column": {}}, "dropped_columns": {}, "targets": {}}' > observed.json
        """
}


process ENRICHMENT {
    tag "${meta.id}"
    label 'process_eval_enrichment'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        tuple val(meta), path(metadata_table), path(predictions)

    output:
        tuple val(meta), path("${meta.model}.enrichment.json"), emit: json

    script:
        """
        eval_enrichment.py \\
            --predictions ${predictions} \\
            --metadata_table ${metadata_table} \\
            --model_name ${meta.model} \\
            --threads ${task.cpus} \\
            --out ${meta.model}.enrichment.json
        """

    stub:
        """
        echo '{"model_name": "${meta.model}", "counts": {"n_predictions": 0, "n_with_metadata": 0, "n_complete": 0, "n_missing_by_column": {}}, "dropped_columns": {}, "targets": {}}' > ${meta.model}.enrichment.json
        """
}
