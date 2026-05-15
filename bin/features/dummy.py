#!/usr/bin/env python3
"""Dummy feature encoder.

Emits a 512-dim random vector per (domain, protein) pair. 512 chosen
as a mid-point of real embedding dimensionalities (ESM/ProtT5 range
~480-2560), so the dummy carries comparable input shape to real
encoders without leaking any signal.

Rationale: a constant zero vector causes classifiers to collapse to
the majority class. Per-row random values produce unique inputs;
AUROC converges to ~0.5 (the true sanity floor). If a real model
fails to beat this, something is wrong with the data or training loop.
"""
import h5py
import numpy as np
import pandas as pd
import sqlite3

DUMMY_DIM = 512

_RNG = np.random.default_rng()


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

    for domain_id, protein_id in domain_protein_df.itertuples(index=False):
        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = _RNG.standard_normal(DUMMY_DIM, dtype=np.float32)

    print(
        f"dummy: wrote {len(domain_protein_df)} (domain, protein) entries x "
        f"{DUMMY_DIM}-dim random vectors"
    )
