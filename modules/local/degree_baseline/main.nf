// Train-positive degree baseline (W-15): score(A, B) = deg(A) x deg(B), deg =
// positive train DDIs of the family. Model-free and ranking-only: no model
// directory, no thresholds; EVAL_ONE reads it with --ranking_only. One task per
// (database, seed): the scores are identical across seeds, the rows are
// evaluation replicates (the bootstrap CIs use the seed, D-W15-2).
process DEGREE_BASELINE {
    tag "${meta.id}"
    label 'process_single'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        tuple val(meta), path('DDI')

    output:
        tuple val(meta), path("degree_baseline/predictions_*.parquet"), emit: predictions

    script:
        def tests = meta.tests.collect { variant, split -> "--test ${variant}=${split}" }.sort().join(' ')
        """
        degree_baseline.py --ddi_path DDI/ ${tests} --out_dir degree_baseline
        """

    stub:
        def variants = meta.tests.keySet().join(' ')
        """
        mkdir -p degree_baseline
        for v in ${variants}; do
            touch degree_baseline/predictions_\${v}.parquet
        done
        """
}
