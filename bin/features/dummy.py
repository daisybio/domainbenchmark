#!/usr/bin/env python3
"""Dummy feature encoder.

Emits a constant zero vector for every (domain, protein) pair so the
downstream ML/RF classifiers have no learnable signal. Used as a
sanity-floor baseline — AUROC should land around 0.5 on any held-out
split. If a real model fails to beat this, something is wrong with the
data or the training loop.
"""
import h5py
import pandas as pd
import sqlite3

DUMMY_DIM = 8


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    domain_protein_df = pd.read_sql(
        """
        SELECT domain_id, protein_id
        FROM domain_protein_map;
        """,
        conn,
    )

    domain_protein_df["domain_id"] = domain_protein_df["domain_id"].astype(str)
    domain_protein_df["protein_id"] = domain_protein_df["protein_id"].astype(str)

    dummy_vector = [0.0] * DUMMY_DIM

    for domain_id, protein_id in domain_protein_df.itertuples(index=False):
        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = dummy_vector

    print(f"dummy: wrote {len(domain_protein_df)} (domain, protein) entries x {DUMMY_DIM}-dim zero vector")
