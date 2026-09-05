#!/usr/bin/env python3

"""_summary_
Per-database metadata construction script for the domainbenchmark workflow.

Based on the complete metadata sets, containing metadata for all domains/ddis, the metadata for the current database is extracted
Prevents redundancy in metadata management.

Gets path to metadata directory, the ddis of the test set of the current database and the output file

NOTE: Internally, metadata is still aggregated at the pfam_id level (that's the
level the source metadata lives at), but the final output is re-keyed to use
domain_a / domain_b (the ids from the test-set ddi instances) instead of
pfam_id_a / pfam_id_b, for downstream compatibility.
"""


import argparse as ap
import pandas as pd
import os
import numpy as np


def read_metadata(metadata_dir):
    domain_metadata_path = os.path.join(metadata_dir, "domain_metadata.csv")
    ddi_metadata_path = os.path.join(metadata_dir, "ddi_metadata.csv")

    if not os.path.exists(domain_metadata_path):
        raise FileNotFoundError(f"Domain metadata file not found at {domain_metadata_path}")
    if not os.path.exists(ddi_metadata_path):
        raise FileNotFoundError(f"DDI metadata file not found at {ddi_metadata_path}")

    domain_metadata_df = pd.read_csv(domain_metadata_path)  # pfam_id, uniprot_id, ...
    ddi_metadata_df = pd.read_csv(ddi_metadata_path) # pfam_id_a, pfam_id_b, uniprot_id_a, uniprot_id_b, ...
    # Rename d1,d2,p1,p2 to pfam_id_a, pfam_id_b, uniprot_id_a, uniprot_id_b for consistency if they are not already named that way

    # Get first four columns of ddi_metadata_df
    first_four_cols = ddi_metadata_df.columns[:4].tolist()
    if first_four_cols != ['pfam_id_a', 'pfam_id_b', 'uniprot_id_a', 'uniprot_id_b']:
        ddi_metadata_df.rename(columns={'d1': 'pfam_id_a', 'd2': 'pfam_id_b', 'p1': 'uniprot_id_a', 'p2': 'uniprot_id_b'}, inplace=True)

    return domain_metadata_df, ddi_metadata_df


def read_ddi_instances(ddi_instances_path):
    if not os.path.exists(ddi_instances_path):
        raise FileNotFoundError(f"DDI instances file not found at {ddi_instances_path}")

    ddi_instances_df = pd.read_csv(ddi_instances_path)
    return ddi_instances_df


def read_mapping(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")

    # domain_id,pfam_id
    mapping_df = pd.read_csv(mapping_path)
    return mapping_df


def create_database_specific_metadata(domain_metadata_df, ddi_metadata_df, ddi_instances_df, mapping_df):

    # Merge ddi_instances with mapping to get pfam ids for both domains in the DDI
    merged_df = ddi_instances_df.merge(mapping_df, left_on='domain_1', right_on='domain_id', how='left') \
                                .merge(mapping_df, left_on='domain_2', right_on='domain_id', how='left', suffixes=('_a', '_b'))

    all_pfam_ids = set(merged_df['pfam_id_a'].dropna()).union(set(merged_df['pfam_id_b'].dropna()))

    print(f"Found {len(all_pfam_ids)} unique pfam ids in the DDI instances for the current database")

    # Subet domain metadata for the current database
    db_domain_metadata_df = domain_metadata_df[domain_metadata_df['pfam_id'].isin(all_pfam_ids)]

    print(f"Subsetted domain metadata for the current database: {len(db_domain_metadata_df)} rows")

    # Subset ddi metadata for the current database, create key based on pfam_id_a and pfam_id_b using sorted order to ensure consistency
    ddi_metadata_df['ddi_key'] = ddi_metadata_df.apply(lambda row: tuple(sorted((row['pfam_id_a'], row['pfam_id_b']))), axis=1)
    print(f"Created ddi_key for ddi_metadata_df: {len(ddi_metadata_df)} rows")
    merged_df['ddi_key'] = merged_df.apply(lambda row: tuple(sorted((row['pfam_id_a'], row['pfam_id_b']))), axis=1)
    print(f"Created ddi_key for merged_df: {len(merged_df)} rows")
    db_ddi_metadata_df = ddi_metadata_df[ddi_metadata_df['ddi_key'].isin(merged_df['ddi_key'])]

    # merged_df keeps the ORIGINAL (non-sorted) orientation of the test-set
    # instances: domain_1/pfam_id_a <-> domain_2/pfam_id_b as they appear in
    # the ddi instances file. We carry this along so the final output can be
    # re-keyed to domain ids while preserving the correct a/b orientation.
    instance_map_df = merged_df[['domain_1', 'domain_2', 'pfam_id_a', 'pfam_id_b', 'ddi_key']].drop_duplicates()

    return db_domain_metadata_df, db_ddi_metadata_df, instance_map_df




def aggregate_metadata_to_domain_level(metadata, key="pfam_id", method="mean"):
    """
    Metadata originates at protein level: each domain (identified by `key`)
    can appear multiple times, once per protein it occurs in. Before we can
    join metadata onto domain-domain interactions, we need exactly one row
    per domain.

    For now, only numerical features are aggregated (via `method`, default
    mean). Categorical features are detected and dropped with a warning --
    they need a separate encoding strategy (one-hot + proportion, or entropy)
    which is not implemented yet.

    Returns:
        agg_metadata: DataFrame with one row per domain (key + numerical features)
        numerical_features: list of numerical feature column names retained
    """
    if key not in metadata.columns:
        raise ValueError(f"Metadata key '{key}' not found in metadata columns: {metadata.columns.tolist()}")

    feature_cols = [c for c in metadata.columns if c != key]
    numerical_features = metadata[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numerical_features]

    if categorical_features:
        print(f"Skipping {len(categorical_features)} categorical feature(s) for now: {categorical_features}")

    n_rows_before = len(metadata)
    n_domains = metadata[key].nunique()
    if n_rows_before > n_domains:
        print(f"Aggregating metadata: {n_rows_before} rows -> {n_domains} unique domains "
              f"(method='{method}')")

    agg_metadata = metadata.groupby(key)[numerical_features].agg(method).reset_index()

    return agg_metadata, numerical_features




def aggregate_metadata_to_ddi_level(ddi_metadata, key_cols=("d1", "d2"), method="mean"):
    """
    DDI-level metadata already has (at most) one row per DDI *instance*
    (i.e. per (domain_a, domain_b, protein_a, protein_b) combination), but
    the same *domain pair* can occur multiple times across different
    protein instances. Before joining onto predictions -- which only carry
    the domain pair, not the protein pair -- we need exactly one row per
    unordered domain pair.
 
    Only numerical features are aggregated (via `method`, default mean).
    Categorical features (and any protein-identifying columns) are dropped.
 
    Returns:
        agg_ddi_metadata: DataFrame with one row per unordered domain pair
            (columns: "_pair_key" (frozenset-free sorted tuple) + numerical features)
        ddi_numerical_features: list of numerical feature column names retained
    """
    d1_col, d2_col = key_cols
    missing = [c for c in key_cols if c not in ddi_metadata.columns]
    if missing:
        raise ValueError(f"DDI metadata key column(s) {missing} not found in columns: {ddi_metadata.columns.tolist()}")
 
    # Drop anything that identifies the protein instance rather than the domain
    # pair itself -- those aren't features and shouldn't be aggregated/kept.
    id_like_cols = set(key_cols) | {"p1", "p2"}
    feature_cols = [c for c in ddi_metadata.columns if c not in id_like_cols]
 
    ddi_numerical_features = ddi_metadata[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in ddi_numerical_features]
    if categorical_features:
        print(f"Skipping {len(categorical_features)} categorical DDI-level feature(s) for now: {categorical_features}")
 
    ddi_metadata = ddi_metadata.copy()
    ddi_metadata["_pair_key"] = ddi_metadata.apply(lambda r: tuple(sorted((r[d1_col], r[d2_col]))), axis=1)
 
    n_rows_before = len(ddi_metadata)
    n_pairs = ddi_metadata["_pair_key"].nunique()
    if n_rows_before > n_pairs:
        print(f"Aggregating DDI-level metadata: {n_rows_before} rows -> {n_pairs} unique domain pairs "
              f"(method='{method}')")
 
    agg_ddi_metadata = ddi_metadata.groupby("_pair_key")[ddi_numerical_features].agg(method).reset_index()
 
    return agg_ddi_metadata, ddi_numerical_features



def create_single_metadata_file(agg_domain_metadata, agg_ddi_metadata, instance_map_df, out_path):
    """
    Create a single metadata file, one row per DDI *instance* (i.e. one row
    per domain_id pair from the test set), one column group for
    interaction-level metadata, and two column groups for domain-level
    metadata (one for each domain in the DDI, with _a and _b suffixes).

    The output is keyed on domain_a / domain_b instead of
    pfam_id_a / pfam_id_b, for downstream compatibility. Internally the
    metadata is still aggregated at the pfam_id / sorted-pfam-pair level
    (agg_domain_metadata, agg_ddi_metadata), and is joined back onto the
    original (non-sorted) test-set instances via `instance_map_df` so that
    each output row corresponds to a specific domain_id pair with the
    correct a/b orientation preserved.
    """
    if agg_ddi_metadata.empty:
        raise ValueError("Warning: agg_ddi_metadata is empty -- no DDI keys matched between the test-set ")

    bad_keys = agg_ddi_metadata['_pair_key'].apply(lambda k: not isinstance(k, tuple) or len(k) != 2)
    if bad_keys.any():
        raise ValueError(
            f"Found {bad_keys.sum()} '_pair_key' entries that are not length-2 tuples "
            f"(likely due to NaN pfam_id_a/pfam_id_b upstream). Example bad rows:\n"
            f"{agg_ddi_metadata.loc[bad_keys, '_pair_key'].head()}"
        )

    # Split the _pair_key into two separate columns (still sorted-pfam orientation)
    agg_ddi_metadata = agg_ddi_metadata.copy()
    agg_ddi_metadata[['pfam_id_a', 'pfam_id_b']] = pd.DataFrame(
        agg_ddi_metadata['_pair_key'].tolist(), index=agg_ddi_metadata.index
    )
    agg_ddi_metadata.drop(columns=['_pair_key'], inplace=True)

    # Merge domain-level metadata for both domains in the DDI (still keyed on
    # sorted pfam_id_a / pfam_id_b at this point). Explicitly suffix the
    # domain-level feature columns before each merge (instead of relying on
    # pandas' automatic suffixing) to avoid a column-name collision between
    # the auto-suffixed 'pfam_id' key column and the already-present
    # 'pfam_id_a' / 'pfam_id_b' columns.
    domain_feat_cols = [c for c in agg_domain_metadata.columns if c != 'pfam_id']
    agg_domain_a = agg_domain_metadata.rename(columns={c: f"{c}_a" for c in domain_feat_cols})
    agg_domain_b = agg_domain_metadata.rename(columns={c: f"{c}_b" for c in domain_feat_cols})

    merged_df = agg_ddi_metadata.merge(agg_domain_a, left_on='pfam_id_a', right_on='pfam_id', how='left')
    merged_df.drop(columns=['pfam_id'], inplace=True)
    merged_df = merged_df.merge(agg_domain_b, left_on='pfam_id_b', right_on='pfam_id', how='left')
    merged_df.drop(columns=['pfam_id'], inplace=True)

    # --- Re-key from (sorted) pfam_id_a/pfam_id_b to domain_a/domain_b ---
    # instance_map_df has the ORIGINAL orientation for each test-set instance:
    # domain_1 <-> pfam_id_a_orig, domain_2 <-> pfam_id_b_orig, plus ddi_key
    # (the sorted pfam pair used to join against merged_df).
    merged_df['ddi_key'] = merged_df.apply(lambda r: tuple(sorted((r['pfam_id_a'], r['pfam_id_b']))), axis=1)

    instance_map = instance_map_df.rename(
        columns={'pfam_id_a': 'pfam_id_a_orig', 'pfam_id_b': 'pfam_id_b_orig'}
    )

    expanded_df = instance_map.merge(merged_df, on='ddi_key', how='left')

    # Whether the sorted pfam_id_a in merged_df matches the instance's
    # original pfam_id_a (domain_1) or its pfam_id_b (domain_2) determines
    # whether the "_a"/"_b" metadata suffixes line up with domain_1/domain_2
    # directly, or need to be swapped.
    same_orientation = expanded_df['pfam_id_a'] == expanded_df['pfam_id_a_orig']

    expanded_df['domain_a'] = expanded_df['domain_1']
    expanded_df['domain_b'] = expanded_df['domain_2']

    # Identify the domain-level metadata columns (always suffixed _a / _b now,
    # coming from agg_domain_metadata, excluding the id/key columns)
    domain_feature_cols_a = [f"{c}_a" for c in domain_feat_cols]
    domain_feature_cols_b = [f"{c}_b" for c in domain_feat_cols]

    # For rows where the orientation is flipped (instance's domain_1 actually
    # corresponds to the sorted pfam_id_b), swap the _a/_b domain-level metadata
    # columns so they stay aligned with domain_a/domain_b.
    flip_idx = expanded_df.index[~same_orientation]
    for col_a, col_b in zip(domain_feature_cols_a, domain_feature_cols_b):
        if col_a == col_b:
            continue
        tmp = expanded_df.loc[flip_idx, col_a].copy()
        expanded_df.loc[flip_idx, col_a] = expanded_df.loc[flip_idx, col_b]
        expanded_df.loc[flip_idx, col_b] = tmp

    # Drop all pfam-id / helper columns, keep only domain ids + metadata
    drop_cols = ['domain_1', 'domain_2', 'pfam_id_a_orig', 'pfam_id_b_orig',
                 'pfam_id_a', 'pfam_id_b', 'ddi_key']
    expanded_df.drop(columns=[c for c in drop_cols if c in expanded_df.columns], inplace=True)

    # Put domain_a / domain_b as the first two columns
    other_cols = [c for c in expanded_df.columns if c not in ('domain_a', 'domain_b')]
    expanded_df = expanded_df[['domain_a', 'domain_b'] + other_cols]

    # Save to CSV
    expanded_df.to_csv(out_path, index=False)
    print(f"Saved combined metadata to {out_path} ({len(expanded_df)} DDIs)")


def main():

    parser = ap.ArgumentParser(description="Create metadata for a specific database")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Path to the directory containing the complete metadata sets (domain_metadata.csv and ddi_metadata.csv)")
    parser.add_argument("--ddi", type=str, required=True, help="Path to the file containing the test set DDIs for the current database")
    parser.add_argument("--mapping", type=str, required=True, help="Path to the file containing the mapping of domain ids to pfam ids for the current database")
    parser.add_argument("--out", type=str, required=True, help="Path to the output file where the database-specific metadata will be saved")
    args = parser.parse_args()

    domain_metadata_df, ddi_metadata_df = read_metadata(args.metadata_dir)
    ddi_instances_df = read_ddi_instances(args.ddi)
    mapping_df = read_mapping(args.mapping)

    # Subet metadata for the current database
    db_domain_metadata_df, db_ddi_metadata_df, instance_map_df = create_database_specific_metadata(domain_metadata_df, ddi_metadata_df, ddi_instances_df, mapping_df)

    print(f"db_ddi_metadata_df: {len(db_ddi_metadata_df)} rows")

    # Aggregate metadata on domain/ddi-level, e.g., for each domain, aggregate all uniprot ids, for each ddi, aggregate all uniprot pairs

    db_domain_metadata_agg_df, domain_numerical_features = aggregate_metadata_to_domain_level(db_domain_metadata_df, key="pfam_id", method="mean")

    if not db_domain_metadata_agg_df.empty:
        db_ddi_metadata_agg_df, ddi_numerical_features = aggregate_metadata_to_ddi_level(db_ddi_metadata_df, key_cols=("pfam_id_a", "pfam_id_b"), method="mean")

    # If db_ddi_metadata_df is empty, create a df based on the ddi_instances_df + the mapping, containing just pfam_id_a and pfam_id_b, and the _pair_key, to ensure that the output file has the same number of rows as the ddi_instances_df
    if db_ddi_metadata_agg_df.empty:
        print("Warning: db_ddi_metadata_agg_df is empty -- no DDI keys matched between the test-set and the complete metadata sets. Creating a placeholder dataframe with only pfam_id_a, pfam_id_b, and _pair_key.")
        db_ddi_metadata_agg_df = ddi_instances_df.merge(mapping_df, left_on='domain_1', right_on='domain_id', how='left')
        db_ddi_metadata_agg_df = db_ddi_metadata_agg_df.merge(mapping_df, left_on='domain_2', right_on='domain_id', how='left', suffixes=('_a', '_b'))
        db_ddi_metadata_agg_df['_pair_key'] = db_ddi_metadata_agg_df.apply(lambda r: tuple(sorted((r['pfam_id_a'], r['pfam_id_b']))), axis=1) 
        db_ddi_metadata_agg_df = db_ddi_metadata_agg_df[['_pair_key', 'pfam_id_a', 'pfam_id_b']].drop_duplicates().reset_index(drop=True)

    # Create single metadata file, one row per ddi instance (domain_a / domain_b),
    # one column group for interaction-level metadata, two column groups for
    # domain-level metadata (one for each domain in the ddi, with _a and _b suffixes)
    create_single_metadata_file(db_domain_metadata_agg_df, db_ddi_metadata_agg_df, instance_map_df, args.out)




if __name__ == "__main__":
    main()