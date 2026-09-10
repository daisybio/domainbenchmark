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
from determinism import derive_seed
from features import embeddings

DUMMY_DIM = 512


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int):
    domain_protein_df = pd.read_sql(
        f"""
        SELECT {embeddings.DOMAIN_KEY_SQL}, {embeddings.INSTANCE_KEY_SQL}
        FROM domain_protein_map
        {embeddings.DOMAIN_JOIN_SQL};
        """,
        conn,
    )

    domain_protein_df["domain_key"] = domain_protein_df["domain_key"].astype(str)
    domain_protein_df["instance_key"] = domain_protein_df["instance_key"].astype(str)

    for domain_key, instance_key in domain_protein_df.itertuples(index=False):
        embeddings.write_instance(
            out_file,
            domain_key,
            instance_key,
            # A per-instance RNG derived from --seed, not the module-global one.
            # The global RNG is seeded (extract_features.py calls
            # seed_everything), but drawing from it in a loop makes each
            # vector's value depend on the *position* of its row, and the query
            # above has no ORDER BY -- so a row-order change would silently
            # reassign every dummy vector. derive_seed keys on the (domain,
            # instance) pair instead: each pair gets the same vector regardless
            # of the order the rows arrive in, or of how many rows precede it.
            np.random.default_rng(derive_seed(seed, "dummy", domain_key, instance_key))
            .standard_normal(DUMMY_DIM)
            .astype(np.float32),
        )

    print(
        f"dummy: wrote {len(domain_protein_df)} (domain, instance) entries x "
        f"{DUMMY_DIM}-dim random vectors"
    )
