// Gate the published domainsplit embedding files against the databases they
// will be paired with, and hand them on under their feature names.
//
// Two jobs in one process, deliberately:
//
//  1. Verification. A published `<model>_domain_embeddings.h5` is keyed
//     `h5[pfam_id][instance_id]`: one file covers every split database of the
//     run that produced it, because it holds every domain the run saw. Nothing
//     downstream would raise if it did not fit -- `load_embedding_data` skips
//     the (pfam_id, instance_id) pairs it cannot resolve, so an ill-fitting file
//     yields zero training rows rather than an error. Running the check as a
//     process makes the failure structural: NN/RF consume this output, so they
//     cannot start on unverified files.
//
//     What it cannot catch is a foreign *run*: Pfam accessions are stable across
//     runs, so an export made by another run over the same domains resolves like
//     a native one. Pairing the right export with the right databases is the
//     caller's job -- and with `--embeddings` unset the pipeline derives it from
//     `--input`'s own directory, which gets it right by construction.
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
        """
        verify_embeddings.py \\
            --db-dir ${database_dir} \\
            ${pairs} \\
            --min-coverage ${params.min_embedding_coverage}

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
