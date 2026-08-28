#!/usr/bin/env python3
"""Template for adding a new feature encoding to the benchmark pipeline.

Steps to add a new feature:
1. Copy this file to bin/features/<your_feature>.py
2. Implement extract_features() below
3. Add '<your_feature>' to params.machine_learning_features in nextflow.config
4. If your feature needs GPU or large memory, also add it to params.large_features

The pipeline auto-discovers features by name: extract_features.py calls
importlib.import_module(f"features.{feature_name}").extract_features(conn, out_file).

Database schema (domain_protein_map table):
    domain_id       INT     -- FK to domain.id
    protein_id      INT     -- FK to protein.id
    domain_sequence TEXT    -- amino acid sequence of the domain
    start_pos       INT     -- domain start position in protein sequence
    end_pos         INT     -- domain end position in protein sequence
    instance_id     TEXT    -- domainsplit's domain-instance id (opaque, may be NULL)
    clan            TEXT
    taxon_id        TEXT

There are no embedding BLOBs in the database. domainsplit embeds the cut domain
sequence outside SQLite now and publishes the vectors as HDF5, so an
embedding-backed feature is not written here at all -- it goes in
`params.published_features` and is read straight from `--embeddings`. This
template is for something you compute from the columns above.

HDF5 output structure (required by downstream ML models):
    /<domain_id>/<instance_key> = numpy array of shape (feature_dim,)

The key is the *instance*, not the protein: domain_protein_map is unique on
(domain_id, protein_id, start_pos, end_pos), so a protein carrying two copies
of one family has two rows that would collide on protein_id. Use
`embeddings.INSTANCE_KEY_SQL` in the SELECT and `embeddings.write_instance()`
to write, as every shipped encoder does. Never parse an instance key apart.

See aacomp.py for a minimal example, embeddings.py for pre-computed
embedding extraction with helper utilities.
"""

import h5py
import numpy as np
import pandas as pd
import sqlite3
from features import embeddings


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / validation.sqlite3 /
              test*.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (domain, instance)
                  pair, grouped by domain_id.
    """
    domain_protein_df = pd.read_sql(
        f"""
        SELECT domain_id, {embeddings.INSTANCE_KEY_SQL},
               UPPER(domain_sequence) AS sequence
        FROM domain_protein_map;
        """,
        conn,
    )

    domain_protein_df["domain_id"] = domain_protein_df["domain_id"].astype(str)
    domain_protein_df["instance_key"] = domain_protein_df["instance_key"].astype(str)

    for domain_id, instance_key, sequence in domain_protein_df.itertuples(index=False):
        # --- Replace this block with your feature computation ---
        feature_vector = np.zeros(128, dtype=np.float32)  # placeholder
        # --------------------------------------------------------

        embeddings.write_instance(out_file, domain_id, instance_key, feature_vector)

    print(f"new_feature: wrote {len(domain_protein_df)} entries")
