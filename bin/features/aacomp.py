#!/usr/bin/env python3
import h5py
import pandas as pd
import sqlite3
from features import embeddings

AA = "ACDEFGHIKLMNPQRSTVWY"


def get_aa_comp(seq: str) -> list[float]:
    """Calculate amino acid composition"""
    aa_count = {aa: 0 for aa in AA}
    for aa in seq:
        if aa in aa_count:
            aa_count[aa] += 1
    total = sum(aa_count.values())
    return [aa_count[aa] / total for aa in AA]


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int):
    # Deterministic encoder: `seed` is part of the encoder ABI so that a
    # sampling encoder cannot be added without one. Unused here.
    #
    # If the database has a ddi_split_membership table, only domain instances
    # used by real (is_mock = 0) DDI rows are encoded, so the domain features
    # are built from exactly the same data as the DDI features.
    domain_sequence_df = pd.read_sql(
        f"""
                SELECT {embeddings.DOMAIN_KEY_SQL},
                       {embeddings.INSTANCE_KEY_SQL},
                       UPPER(domain_sequence) AS sequence
                FROM domain_protein_map
                {embeddings.DOMAIN_JOIN_SQL}
                {embeddings.real_instance_filter_sql(conn)};
            """,
        conn,
    )

    domain_sequence_df["aacomp"] = domain_sequence_df["sequence"].apply(get_aa_comp)
    domain_sequence_df["domain_key"] = domain_sequence_df["domain_key"].astype(str)
    domain_sequence_df["instance_key"] = domain_sequence_df["instance_key"].astype(str)

    for (
        domain_key,
        instance_key,
        domain_sequence,
        aacomp,
    ) in domain_sequence_df.itertuples(index=False):
        embeddings.write_instance(out_file, domain_key, instance_key, aacomp)

    print(domain_sequence_df.head())
