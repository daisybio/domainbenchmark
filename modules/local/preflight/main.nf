// Per-database checks for everything that would otherwise fail after ML
// training (bin/preflight.py lists them). PER_DB_BENCHMARK joins NN/RF/SVM and
// SASA_CACHE_SHARD on this task's output, so a database that would crash
// EVALUATION or ENRICHMENT fails in minutes instead of after the training.
//
// The general image on purpose: it is the one EVALUATION / ENRICHMENT /
// COMBINE_EVAL run in, so the toolchain check tests what those will load.

process PREFLIGHT {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.2"

    input:
        tuple val(meta), path(database_dir, stageAs: "input_db"), val(published_features), path(published_files, stageAs: 'published/*')
        path(structures, stageAs: 'structures.h5')
        path(metadata, stageAs: 'metadata')
        val(spec)   // [combos: [[feature, ...], ...], extracted: [name: module], structure: [name: module]]

    output:
        tuple val(meta), path("preflight.txt"), emit: report

    script:
        // Both lists arrive unwrapped when there is exactly one published feature.
        def feats = published_features instanceof List ? published_features : [published_features]
        def files = published_files instanceof List ? published_files : [published_files]
        def args = (
            meta.splits.collect { "--split ${it}" } +
            spec.combos.collect { "--combo ${it.join(',')}" } +
            [feats, files].transpose().collect { f, p -> "--published ${f}=${p}" } +
            spec.extracted.collect { n, m -> "--extracted ${n}=${m}" } +
            spec.structure.collect { n, m -> "--structure ${n}=${m}" } +
            (structures ? ['--structures structures.h5'] : []) +
            (metadata ? ['--metadata metadata'] : [])
        ).join(' \\\n            ')
        """
        preflight.py \\
            --db-dir ${database_dir} \\
            ${args}
        """

    stub:
        """
        touch preflight.txt
        """
}
