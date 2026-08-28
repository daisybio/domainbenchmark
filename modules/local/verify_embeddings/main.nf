// Gate the published domainsplit embedding files against the databases they
// will be paired with, and hand them on under their feature names.
//
// Two jobs in one process, deliberately:
//
//  1. Verification. `domain.id` is a surrogate integer that domainsplit copies
//     verbatim into every split database and never renumbers, so one published
//     `<model>_domain_embeddings.h5` is valid across every split of the run that
//     produced it and silently wrong across runs. Nothing downstream would
//     raise: `load_embedding_data` skips instance pairs it cannot resolve, so a
//     foreign file yields zero training rows rather than an error. Running the
//     check as a process makes the failure structural -- NN/RF consume this
//     output, so they cannot start on unverified files.
//
//  2. Renaming. domainsplit publishes `esm3_domain_embeddings.h5`; the feature
//     is called `esm3_embeddings`. `path('features/*')` stages a file under its
//     own basename, so the symlink here is what lets
//     `machine_learning.resolve_feature_file` find `features/<feature>.h5`.
//
// One task per database directory, not per (database x feature): the check is
// metadata-only and the published files are per-run, so there is nothing to
// gain from fanning out over multi-gigabyte inputs.

process VERIFY_EMBEDDINGS {
    tag "${meta.id}"
    label 'process_low'

    conda "${projectDir}/environments/general.yml"
    container "docker.io/konstantinpelz/domainbenchmark-general:1.0.0"

    input:
        tuple val(meta), path(database_dir), val(features), path(embedding_files)

    output:
        tuple val(meta), path("verified/*.h5"), emit: h5

    script:
        // features and embedding_files are index-aligned by the caller. Both
        // arrive unwrapped when there is exactly one published feature.
        def feats = features instanceof List ? features : [features]
        def files = embedding_files instanceof List ? embedding_files : [embedding_files]
        def pairs = [feats, files]
            .transpose()
            .collect { feature, f -> "--pair ${feature}=${f}" }
            .join(' \\\n            ')
        def links = [feats, files]
            .transpose()
            .collect { feature, f -> "ln -s ../${f} verified/${feature}.h5" }
            .join('\n        ')
        def expect_run = params.domainsplit_run ? "--expect-run '${params.domainsplit_run}'" : ''
        """
        verify_embeddings.py \\
            --db-dir ${database_dir} \\
            ${pairs} \\
            --min-coverage ${params.min_embedding_coverage} ${expect_run}

        mkdir -p verified
        ${links}
        """

    stub:
        def feats = features instanceof List ? features : [features]
        def links = feats.collect { "touch verified/${it}.h5" }.join('\n        ')
        """
        mkdir -p verified
        ${links}
        """
}
