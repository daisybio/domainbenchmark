#!/usr/bin/env python3
"""Dummy feature encoder.

Emits a 512-dim random vector per (domain, instance) pair. 512 chosen
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
from features import embeddings

DUMMY_DIM = 512


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    domain_protein_df = pd.read_sql(
        f"""
        SELECT domain_id, {embeddings.INSTANCE_KEY_SQL}
        FROM domain_protein_map;
        """,
        conn,
    )

    domain_protein_df["domain_id"] = domain_protein_df["domain_id"].astype(str)
    domain_protein_df["instance_key"] = domain_protein_df["instance_key"].astype(str)

    for domain_id, instance_key in domain_protein_df.itertuples(index=False):
        embeddings.write_instance(
            out_file,
            domain_id,
            instance_key,
            # The module-global numpy RNG, seeded by extract_features.py from
            # --seed. A private default_rng() with no seed made every run of
            # this encoder emit a different h5, and every model trained on it
            # irreproducible.
            np.random.standard_normal(DUMMY_DIM).astype(np.float32),
        )

    print(
        f"dummy: wrote {len(domain_protein_df)} (domain, instance) entries x "
        f"{DUMMY_DIM}-dim random vectors"
    )
