#!/usr/bin/env python3

"""_summary_
Per-database metadata construction script for the domainbenchmark workflow.

Based on the complete metadata sets, containing metadata for all domains/ddis, the metadata for the current database is extracted
Prevents redundancy in metadata management.

Gets path to metadata directory, the ddis of the test set of the current database and the output file

NOTE: domain_1 / domain_2 in the ddi instances file are now already pfam ids
directly (no separate domain_id -> pfam_id mapping step is needed anymore).
Internally, metadata is still aggregated at the pfam_id level (that's the
level the source metadata lives at) using a *sorted* (pfam_id_a, pfam_id_b)
key, since the same unordered domain pair can appear in different orders
across different protein-pair instances. The final output is re-keyed back
to the ORIGINAL (unsorted) pfam_id_a / pfam_id_b orientation from the
test-set ddi instances, so both the id columns and the domain-level feature
columns stay consistent with each other.
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

    domain_metadata_df = pd.read_csv(domain_metadata_path)  # instance_id, pfam_id, uniprot_id, ...
    ddi_metadata_df = pd.read_csv(ddi_metadata_path)  # pfam_id_a, pfam_id_b, uniprot_id_a, uniprot_id_b, ...

    # Rename d1,d2,p1,p2 to pfam_id_a, pfam_id_b, uniprot_id_a, uniprot_id_b for consistency if they are not already named that way
    first_four_cols = ddi_metadata_df.columns[:4].tolist()
    if first_four_cols != ['pfam_id_a', 'pfam_id_b', 'uniprot_id_a', 'uniprot_id_b']:
        ddi_metadata_df.rename(columns={'d1': 'pfam_id_a', 'd2': 'pfam_id_b', 'p1': 'uniprot_id_a', 'p2': 'uniprot_id_b'}, inplace=True)

    return domain_metadata_df, ddi_metadata_df


def read_ddi_instances(ddi_instances_path):
    if not os.path.exists(ddi_instances_path):
        raise FileNotFoundError(f"DDI instances file not found at {ddi_instances_path}")

    ddi_instances_df = pd.read_csv(ddi_instances_path)
    #  INstance df has columns domain_1, domain_2, instance_1, instance_2, interaction
    first_two_cols = ddi_instances_df.columns[:2].tolist()
    if first_two_cols != ['pfam_id_a', 'pfam_id_b']:
        ddi_instances_df.rename(columns={'domain_1': 'pfam_id_a', 'domain_2': 'pfam_id_b'}, inplace=True)
    return ddi_instances_df


def create_database_specific_metadata(domain_metadata_df, ddi_metadata_df, ddi_instances_df):

    all_pfam_ids = set(ddi_instances_df['pfam_id_a'].dropna()).union(set(ddi_instances_df['pfam_id_b'].dropna()))
    all_instance_ids = set(ddi_instances_df['instance_1'].dropna()).union(set(ddi_instances_df['instance_2'].dropna()))

    print(f"Found {len(all_pfam_ids)} unique pfam ids in the DDI instances for the current database")
    print(f"Found {len(all_instance_ids)} unique instance ids in the DDI instances for the current database")

    # Check if there are any instance ids in the DDI instances that are not present in the domain metadata
    missing_instance_ids = all_instance_ids - set(domain_metadata_df['instance_id'].dropna())
    if missing_instance_ids:
        print(f"Warning: {len(missing_instance_ids)} instance ids from the DDI instances are missing in the domain metadata. Example missing ids: {list(missing_instance_ids)[:10]}")

    # Subset domain metadata for the current database
    # INstance ids subset ensures that only domain-protein-combinations which are actually present in the ddi instances are kept, since the domain metadata contains all domains in the database, not just those which are part of a ddi instance
    db_domain_metadata_df = domain_metadata_df[domain_metadata_df['instance_id'].isin(all_instance_ids)]

    print(f"Subsetted domain metadata for the current database using instance ids: {len(db_domain_metadata_df)} rows")

    # Subset ddi metadata for the current database, create key based on pfam_id_a and pfam_id_b using sorted order to ensure consistency
    # NOTE: later ddi metadata will also be keyed on instance_id, but for now we just want to subset based on the pfam ids present in the ddi instances
    ddi_metadata_df = ddi_metadata_df.copy()
    ddi_instances_df = ddi_instances_df.copy()

    ddi_metadata_df['ddi_key'] = ddi_metadata_df.apply(lambda row: tuple(sorted((row['pfam_id_a'], row['pfam_id_b']))), axis=1)
    print(f"Created ddi_key for ddi_metadata_df: {len(ddi_metadata_df)} rows (ddi metadata content)")
    ddi_instances_df['ddi_key'] = ddi_instances_df.apply(lambda row: tuple(sorted((row['pfam_id_a'], row['pfam_id_b']))), axis=1)
    print(f"Created ddi_key for ddi_instances_df: {len(ddi_instances_df)} rows (database content)")
    db_ddi_metadata_df = ddi_metadata_df[ddi_metadata_df['ddi_key'].isin(ddi_instances_df['ddi_key'])]
    # Drop the helper key -- it's not a feature and must not leak into the
    # aggregation step (aggregate_metadata_to_ddi_level would otherwise see it
    # as a stray non-numeric "categorical" column).
    db_ddi_metadata_df = db_ddi_metadata_df.drop(columns=['ddi_key'])
    print(f"Subsetted ddi metadata for the current database: {len(db_ddi_metadata_df)} rows")

    # Subset ddi metadata based on instance ids in the ddi instances
    # db_ddi_metadata_df = ddi_metadata_df[ddi_metadata_df['instance_id'].isin(all_instance_ids)]
    print(f"Subsetted ddi metadata for the current database using instance ids: {len(db_ddi_metadata_df)} rows")

    # instance_map_df keeps the ORIGINAL (non-sorted) orientation of the
    # test-set instances: pfam_id_a <-> pfam_id_b as they appear in the ddi
    # instances file, alongside the sorted ddi_key used to join against the
    # aggregated metadata later on.
    instance_map_df = ddi_instances_df[['pfam_id_a', 'pfam_id_b', 'ddi_key']].drop_duplicates().reset_index(drop=True)

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


def aggregate_metadata_to_ddi_level(ddi_metadata, key_cols=("pfam_id_a", "pfam_id_b"), method="mean"):
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
            (columns: "_pair_key" (sorted tuple) + numerical features)
        ddi_numerical_features: list of numerical feature column names retained
    """
    d1_col, d2_col = key_cols
    missing = [c for c in key_cols if c not in ddi_metadata.columns]
    if missing:
        raise ValueError(f"DDI metadata key column(s) {missing} not found in columns: {ddi_metadata.columns.tolist()}")

    # Drop anything that identifies the protein instance rather than the domain
    # pair itself -- those aren't features and shouldn't be aggregated/kept.
    id_like_cols = set(key_cols) | {"uniprot_id_a", "uniprot_id_b", "p1", "p2"}
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
    per pfam_id pair from the test set), one column group for
    interaction-level metadata, and two column groups for domain-level
    metadata (one for each domain in the DDI, with _a and _b suffixes).

    Internally the metadata is aggregated at the sorted pfam-pair level
    (agg_domain_metadata, agg_ddi_metadata). It's joined back onto the
    original (non-sorted) test-set instances via `instance_map_df` so that
    each output row corresponds to a specific pfam_id pair with the correct
    a/b orientation preserved -- for BOTH the id columns and the domain-level
    feature columns.
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


    # Merge the agg_domain_metadata into the instance_map_df to get the domain-level features for both domains in the DDI, keyed on the original pfam_id_a / pfam_id_b orientation from the test set.
    merged_df = instance_map_df.merge(agg_domain_metadata, left_on='pfam_id_a', right_on='pfam_id', how='left')
    # Rename the domain-level feature columns to have a "_a" suffix for the first domain
    merged_df = merged_df.rename(columns={c: f"{c}_a" for c in agg_domain_metadata.columns if c != 'pfam_id'})
    # Drop column pfam_id after the merge, since it's not a feature and must not leak into the output
    merged_df.drop(columns=['pfam_id'], inplace=True)
    merged_df = merged_df.merge(agg_domain_metadata, left_on='pfam_id_b', right_on='pfam_id', how='left')
    # Rename the domain-level feature columns to have a "_b" suffix for the second domain
    merged_df = merged_df.rename(columns={c: f"{c}_b" for c in agg_domain_metadata.columns if c != 'pfam_id'})
    # Drop column pfam_id after the merge, since it's not a feature and must not leak into the output
    merged_df.drop(columns=['pfam_id'], inplace=True)
    # Check how many a & b domain-level features are present in the merged_df
    domain_feature_cols_a = [c for c in merged_df.columns if c.endswith('_a')]
    domain_feature_cols_b = [c for c in merged_df.columns if c.endswith('_b')]
    print(f"Merged domain-level features: {len(domain_feature_cols_a)} features for domain_a, {len(domain_feature_cols_b)} features for domain_b")
    
    # Check if there are any NaN values in the domain-level features after the merge
    domain_feature_cols_a = [c for c in merged_df.columns if c.endswith('_a')]
    domain_feature_cols_b = [c for c in merged_df.columns if c.endswith('_b')]
    if merged_df[domain_feature_cols_a + domain_feature_cols_b].isnull().any().any():
        nan_rows = merged_df[merged_df[domain_feature_cols_a + domain_feature_cols_b].isnull().any(axis=1)]
        print(f"Warning: {len(nan_rows)} rows contain NaN values in the domain-level features after merging. Example NaN rows:\n{nan_rows.head()}")
    # Drop ddi_key column, since it's not a feature and must not leak into the output
    merged_df.drop(columns=['ddi_key'], inplace=True)

    # For now only write out this dataframe, dd is not relevant in the current workflow, since the ddi metadata is incomplete and therefore disturbs downstream enrichment analysis
    merged_df.to_csv(out_path, index=False)

    # Split the _pair_key into two separate columns (still sorted-pfam orientation)
    # agg_ddi_metadata = agg_ddi_metadata.copy()
    # agg_ddi_metadata[['pfam_id_a', 'pfam_id_b']] = pd.DataFrame(
    #     agg_ddi_metadata['_pair_key'].tolist(), index=agg_ddi_metadata.index
    # )
    # agg_ddi_metadata.drop(columns=['_pair_key'], inplace=True)

    # # Merge domain-level metadata for both domains in the DDI (still keyed on
    # # sorted pfam_id_a / pfam_id_b at this point). Explicitly suffix the
    # # domain-level feature columns before each merge (instead of relying on
    # # pandas' automatic suffixing) to avoid a column-name collision between
    # # the auto-suffixed 'pfam_id' key column and the already-present
    # # 'pfam_id_a' / 'pfam_id_b' columns.
    # domain_feat_cols = [c for c in agg_domain_metadata.columns if c != 'pfam_id']
    # agg_domain_a = agg_domain_metadata.rename(columns={c: f"{c}_a" for c in domain_feat_cols})
    # agg_domain_b = agg_domain_metadata.rename(columns={c: f"{c}_b" for c in domain_feat_cols})

    # merged_df = agg_ddi_metadata.merge(agg_domain_a, left_on='pfam_id_a', right_on='pfam_id', how='outer')
    # merged_df.drop(columns=['pfam_id'], inplace=True)
    # merged_df = merged_df.merge(agg_domain_b, left_on='pfam_id_b', right_on='pfam_id', how='outer')
    # merged_df.drop(columns=['pfam_id'], inplace=True)

    # --- Re-key from (sorted) pfam_id_a/pfam_id_b back to the ORIGINAL
    # per-instance orientation ---
    # instance_map_df has the ORIGINAL orientation for each test-set instance:
    # pfam_id_a_orig / pfam_id_b_orig, plus ddi_key (the sorted pfam pair used
    # to join against merged_df).
    # merged_df['ddi_key'] = merged_df.apply(lambda r: tuple(sorted((r['pfam_id_a'], r['pfam_id_b']))), axis=1)

    # instance_map = instance_map_df.rename(
    #     columns={'pfam_id_a': 'pfam_id_a_orig', 'pfam_id_b': 'pfam_id_b_orig'}
    # )

    # expanded_df = instance_map.merge(merged_df, on='ddi_key', how='left')

    # # Whether the sorted pfam_id_a in merged_df matches the instance's
    # # original pfam_id_a determines whether the "_a"/"_b" metadata suffixes
    # # (and the pfam_id_a/pfam_id_b id columns themselves) line up with the
    # # original orientation directly, or need to be swapped.
    # same_orientation = expanded_df['pfam_id_a'] == expanded_df['pfam_id_a_orig']
    # flip_idx = expanded_df.index[~same_orientation]

    # # For rows where the orientation is flipped, swap the _a/_b domain-level
    # # metadata columns so they stay aligned with the original a/b instance.
    # domain_feature_cols_a = [f"{c}_a" for c in domain_feat_cols]
    # domain_feature_cols_b = [f"{c}_b" for c in domain_feat_cols]
    # for col_a, col_b in zip(domain_feature_cols_a, domain_feature_cols_b):
    #     if col_a == col_b:
    #         continue
    #     tmp = expanded_df.loc[flip_idx, col_a].copy()
    #     expanded_df.loc[flip_idx, col_a] = expanded_df.loc[flip_idx, col_b]
    #     expanded_df.loc[flip_idx, col_b] = tmp

    # # BUGFIX: the id columns themselves must be re-keyed to the original
    # # orientation too -- otherwise pfam_id_a/pfam_id_b stay in *sorted* order
    # # while the feature columns above have just been flipped to match the
    # # *original* order, silently mismatching ids and features for every
    # # flipped row.
    # expanded_df['pfam_id_a'] = expanded_df['pfam_id_a_orig']
    # expanded_df['pfam_id_b'] = expanded_df['pfam_id_b_orig']

    # # Drop all helper columns, keep only pfam ids + metadata
    # drop_cols = ['pfam_id_a_orig', 'pfam_id_b_orig', 'ddi_key']
    # expanded_df.drop(columns=[c for c in drop_cols if c in expanded_df.columns], inplace=True)

    # # Put pfam_id_a / pfam_id_b as the first two columns
    # other_cols = [c for c in expanded_df.columns if c not in ('pfam_id_a', 'pfam_id_b')]
    # expanded_df = expanded_df[['pfam_id_a', 'pfam_id_b'] + other_cols]

    # # For now the ddi_metadata is incomplete and therefore disturbs downstreams enrichment anlysis
    # # Get all feature columns in the ddi metadata and drop them from the final output
    # ddi_feature_cols = [c for c in agg_ddi_metadata.columns if c not in ('pfam_id_a', 'pfam_id_b')]
    # print(f"Dropping {len(ddi_feature_cols)} DDI-level feature columns from the final output: {ddi_feature_cols}")
    # expanded_df.drop(columns=[c for c in ddi_feature_cols if c in expanded_df.columns], inplace=True)

    # # Check if there are still rows/columns containing NaN values in the final output
    # if expanded_df.isnull().any().any():
    #     nan_rows = expanded_df[expanded_df.isnull().any(axis=1)]
    #     nan_cols = expanded_df.columns[expanded_df.isnull().any()].tolist()
    #     print(f"Warning: {len(nan_rows)} rows and {len(nan_cols)} columns contain NaN values in the final output. Example NaN rows:\n{nan_rows.head()}")
    #     print(f"Columns with NaN values: {nan_cols}")

    # # Save to CSV
    # expanded_df.to_csv(out_path, index=False)
    # print(f"Saved combined metadata to {out_path} ({len(expanded_df)} DDIs)")


def main():

    parser = ap.ArgumentParser(description="Create metadata for a specific database")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Path to the directory containing the complete metadata sets (domain_metadata.csv and ddi_metadata.csv)")
    parser.add_argument("--ddi", type=str, required=True, help="Path to the file containing the test set DDIs for the current database")
    parser.add_argument("--out", type=str, required=True, help="Path to the output file where the database-specific metadata will be saved")
    args = parser.parse_args()

    domain_metadata_df, ddi_metadata_df = read_metadata(args.metadata_dir)
    ddi_instances_df = read_ddi_instances(args.ddi)

    # Subset metadata for the current database
    db_domain_metadata_df, db_ddi_metadata_df, instance_map_df = create_database_specific_metadata(domain_metadata_df, ddi_metadata_df, ddi_instances_df)

    print(f"db_ddi_metadata_df: {len(db_ddi_metadata_df)} rows")

    # Aggregate metadata on domain/ddi-level, e.g., for each domain, aggregate all uniprot ids, for each ddi, aggregate all uniprot pairs
    db_domain_metadata_agg_df, domain_numerical_features = aggregate_metadata_to_domain_level(db_domain_metadata_df, key="pfam_id", method="mean")

    db_ddi_metadata_agg_df = pd.DataFrame()
    if not db_domain_metadata_agg_df.empty:
        db_ddi_metadata_agg_df, ddi_numerical_features = aggregate_metadata_to_ddi_level(db_ddi_metadata_df, key_cols=("pfam_id_a", "pfam_id_b"), method="mean")

    # If db_ddi_metadata_agg_df is empty, create a df based on the ddi_instances_df,
    # containing just pfam_id_a, pfam_id_b, and the _pair_key, to ensure that the
    # output file has the same number of rows as the ddi_instances_df
    if db_ddi_metadata_agg_df.empty:
        print("Warning: db_ddi_metadata_agg_df is empty -- no DDI keys matched between the test-set and the complete metadata sets. Creating a placeholder dataframe with only pfam_id_a, pfam_id_b, and _pair_key.")
        db_ddi_metadata_agg_df = ddi_instances_df.copy()
        db_ddi_metadata_agg_df['_pair_key'] = db_ddi_metadata_agg_df.apply(lambda r: tuple(sorted((r['pfam_id_a'], r['pfam_id_b']))), axis=1)
        db_ddi_metadata_agg_df = db_ddi_metadata_agg_df[['_pair_key', 'pfam_id_a', 'pfam_id_b']].drop_duplicates().reset_index(drop=True)

    # Create single metadata file, one row per ddi instance (pfam_id_a / pfam_id_b),
    # one column group for interaction-level metadata, two column groups for
    # domain-level metadata (one for each domain in the ddi, with _a and _b suffixes)
    create_single_metadata_file(db_domain_metadata_agg_df, db_ddi_metadata_agg_df, instance_map_df, args.out)


if __name__ == "__main__":
    main()