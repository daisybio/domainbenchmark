#!/usr/bin/env python3
"""Dummy feature encoder.

Emits a small Gaussian-noise vector (seeded deterministically per
(domain, protein) pair via blake2b -> uint32) for every entry so
downstream ML/RF classifiers receive zero-information features that
are nonetheless distinct per row.

Rationale: a constant zero vector causes classifiers to collapse to
the majority class (predicts every test pair positive on imbalanced
DDI data). This makes AUROC undefined-by-ties and inflates F1 from
the prior. Per-row noise produces unique inputs, so predictions are
not constant; AUROC converges to ~0.5 (the true sanity floor) and
balanced accuracy stays ~0.5. If a real model fails to beat this,
something is wrong with the data or training loop.
"""
import h5py
import hashlib
import numpy as np
import pandas as pd
import sqlite3

DUMMY_DIM = 8


def _seeded_noise(domain_id: str, protein_id: str) -> np.ndarray:
    """Deterministic per-pair Gaussian noise. blake2b seed -> RNG."""
    h = hashlib.blake2b(f"{domain_id}\t{protein_id}".encode(), digest_size=8).digest()
    seed = int.from_bytes(h, "big") & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DUMMY_DIM, dtype=np.float32)


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

        pfam_group[protein_id] = _seeded_noise(domain_id, protein_id)

    print(
        f"dummy: wrote {len(domain_protein_df)} (domain, protein) entries x "
        f"{DUMMY_DIM}-dim seeded Gaussian noise"
    )
