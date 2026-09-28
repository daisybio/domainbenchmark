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
    ddi_metadata_df = pd.read_csv(ddi_metadata_path)  # pfam_id_a, pfam_id_b, uniprot_id_a, uniprot_id_b, instance_id_a, instance_id_b, source, ...

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

    # all_pfam_ids = set(ddi_instances_df['pfam_id_a'].dropna()).union(set(ddi_instances_df['pfam_id_b'].dropna()))
    all_instance_ids = set(ddi_instances_df['instance_1'].dropna()).union(set(ddi_instances_df['instance_2'].dropna()))

    # print(f"Found {len(all_pfam_ids)} unique pfam ids in the DDI instances for the current database")
    print(f"Found {len(all_instance_ids)} unique instance ids in the DDI instances for the current database")

    # Check if there are any instance ids in the DDI instances that are not present in the domain metadata
    missing_instance_ids = all_instance_ids - set(domain_metadata_df['instance_id'].dropna())
    if missing_instance_ids:
        print(f"Warning: {len(missing_instance_ids)} instance ids from the DDI instances are missing in the domain metadata. Example missing ids: {list(missing_instance_ids)[:10]}")

    # Subset domain metadata for the current database
    # Instance ids subset ensures that only domain-protein-combinations which are actually present in the ddi instances are kept, since the domain metadata contains all domains in the database, not just those which are part of a ddi instance
    db_domain_metadata_df = domain_metadata_df[domain_metadata_df['instance_id'].isin(all_instance_ids)]

    print(f"Subsetted domain metadata for the current database using instance ids: {len(db_domain_metadata_df)} rows")

    ddi_metadata_instance_ids = set(ddi_metadata_df['instance_id_a'].dropna()).union(set(ddi_metadata_df['instance_id_b'].dropna()))
    missing_ddi_instance_ids = all_instance_ids - ddi_metadata_instance_ids
    if missing_ddi_instance_ids:
        print(f"Warning: {len(missing_ddi_instance_ids)} instance ids from the DDI instances are missing in the DDI metadata. Example missing ids: {list(missing_ddi_instance_ids)[:10]}")

    db_ddi_metadata_df = ddi_metadata_df[ddi_metadata_df['instance_id_a'].isin(all_instance_ids) & ddi_metadata_df['instance_id_b'].isin(all_instance_ids)]
    
    ddi_instances_df = ddi_instances_df.copy()
    ddi_instances_df['ddi_key'] = ddi_instances_df.apply(lambda row: tuple(sorted((row['pfam_id_a'], row['pfam_id_b']))), axis=1)
    print(f"Created ddi_key for ddi_instances_df: {len(ddi_instances_df)} rows (database content)")

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
    id_like_cols = set(key_cols) | {"uniprot_id_a", "uniprot_id_b", "instance_id_a", "instance_id_b"}
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

    # In addition, for column source, which contains values downlloaded/inferred, calculate the percent of downloaded instances per ddi pair and add as column source_downloaded_fraction
    if "source" in ddi_metadata.columns:
        source_downloaded_fraction = ddi_metadata.groupby("_pair_key")["source"].apply(lambda x: (x == "downloaded").mean()).reset_index(name="source_downloaded_fraction")
        agg_ddi_metadata = agg_ddi_metadata.merge(source_downloaded_fraction, on="_pair_key", how="left")
        ddi_numerical_features.append("source_downloaded_fraction")

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

    # --- Merge in the DDI-level (interaction) metadata ---
    # agg_ddi_metadata is keyed on '_pair_key', the SORTED (pfam_id_a, pfam_id_b)
    # tuple -- exactly like merged_df['ddi_key']. Since instance_map_df/merged_df
    # already carries the ORIGINAL (unsorted) pfam_id_a/pfam_id_b orientation,
    # no a/b-swapping is needed for the ddi-level features: a sorted-pair key
    # refers to a single interaction-level row regardless of which side is
    # 'a' or 'b' in the original orientation.
    merged_df = merged_df.merge(agg_ddi_metadata, left_on='ddi_key', right_on='_pair_key', how='left')
    ddi_feature_cols = [c for c in agg_ddi_metadata.columns if c != '_pair_key']
    merged_df.drop(columns=['_pair_key'], inplace=True)  # drop the _pair_key column after the merge, since it's not a feature and must not leak into the output

    # Check for NaNs introduced by the ddi-metadata merge (e.g. domain pairs
    # present in the test set but missing from the complete ddi metadata)
    if merged_df[ddi_feature_cols].isnull().any().any():
        nan_rows = merged_df[merged_df[ddi_feature_cols].isnull().any(axis=1)]
        print(f"Warning: {len(nan_rows)} rows contain NaN values in the DDI-level features after merging. Example NaN rows:\n{nan_rows.head()}")

    # Drop helper key columns, since they're not features and must not leak into the output
    merged_df.drop(columns=['ddi_key'], inplace=True)  # , '_pair_key'

    # Put pfam_id_a / pfam_id_b as the first two columns
    other_cols = [c for c in merged_df.columns if c not in ('pfam_id_a', 'pfam_id_b')]
    merged_df = merged_df[['pfam_id_a', 'pfam_id_b'] + other_cols]

    merged_df.to_csv(out_path, index=False)
    print(f"Saved combined metadata to {out_path} ({len(merged_df)} DDIs)")


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