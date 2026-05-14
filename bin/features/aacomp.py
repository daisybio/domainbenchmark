#!/usr/bin/env python3
import h5py
import pandas as pd
import sqlite3

AA = "ACDEFGHIKLMNPQRSTVWY"


def get_aa_comp(seq: str) -> list[float]:
    """Calculate amino acid composition"""
    aa_count = {aa: 0 for aa in AA}
    for aa in seq:
        if aa in aa_count:
            aa_count[aa] += 1
    total = sum(aa_count.values())
    return [aa_count[aa] / total for aa in AA]


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    domain_sequence_df = pd.read_sql(
        """
                SELECT domain_id, protein_id, UPPER(domain_sequence) AS sequence
                FROM domain_protein_map;
            """,
        conn,
    )

    domain_sequence_df["aacomp"] = domain_sequence_df["sequence"].apply(get_aa_comp)
    domain_sequence_df["domain_id"] = domain_sequence_df["domain_id"].astype(str)
    domain_sequence_df["protein_id"] = domain_sequence_df["protein_id"].astype(str)

    for domain_id, uniprot_id, domain_sequence, aacomp in domain_sequence_df.itertuples(
        index=False
    ):
        # create a group for each domain_id and put uniprot_id as a subgroup
        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[uniprot_id] = aacomp

    print(domain_sequence_df.head())
