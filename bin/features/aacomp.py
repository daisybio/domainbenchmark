#!/usr/bin/env python3
import h5py
import pandas as pd
import sqlite3
from features import embeddings

AA = "ACDEFGHIKLMNPQRSTVWY"


def get_aa_comp(seq: str) -> list[float]:
    """Amino acid composition of a sequence already passed through
    `embeddings.clean_sequence` (standard residues only, non-empty)."""
    aa_count = {aa: 0 for aa in AA}
    for aa in seq:
        aa_count[aa] += 1
    return [aa_count[aa] / len(seq) for aa in AA]


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int):
    # Deterministic encoder: `seed` is part of the encoder ABI so that a
    # sampling encoder cannot be added without one. Unused here.
    domain_sequence_df = pd.read_sql(
        f"""
                SELECT {embeddings.DOMAIN_KEY_SQL},
                       {embeddings.INSTANCE_KEY_SQL},
                       UPPER(domain_sequence) AS sequence
                FROM domain_protein_map
                {embeddings.DOMAIN_JOIN_SQL};
            """,
        conn,
    )

    domain_sequence_df["domain_key"] = domain_sequence_df["domain_key"].astype(str)
    domain_sequence_df["instance_key"] = domain_sequence_df["instance_key"].astype(str)
    domain_sequence_df["aacomp"] = [
        get_aa_comp(embeddings.clean_sequence(sequence, domain_key, instance_key))
        for domain_key, instance_key, sequence in domain_sequence_df.itertuples(index=False)
    ]

    for (
        domain_key,
        instance_key,
        domain_sequence,
        aacomp,
    ) in domain_sequence_df.itertuples(index=False):
        embeddings.write_instance(out_file, domain_key, instance_key, aacomp)

    print(domain_sequence_df.head())
