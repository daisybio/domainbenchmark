#!/usr/bin/env python3
"""Template for adding a new feature encoding to the benchmark pipeline.

Steps to add a new feature:
1. Copy this file to bin/features/<your_feature>.py
2. Implement extract_features() below
3. Add '<your_feature>' to params.machine_learning_features in nextflow.config
4. If your feature needs GPU or large memory, also add it to params.large_features

The pipeline auto-discovers features by name: extract_features.py calls
importlib.import_module(f"features.{feature_name}").extract_features(conn, out_file).

CREATE TABLE IF NOT EXISTS domain_structure (
    id         INTEGER PRIMARY KEY,
    ddi_id     REFERENCES domain_domain_interaction ON DELETE CASCADE,
    domain1    REFERENCES domain ON DELETE CASCADE,
    domain2    REFERENCES domain ON DELETE CASCADE,
    protein1   REFERENCES protein ON DELETE CASCADE,
    protein2   REFERENCES protein ON DELETE CASCADE,
    source     TEXT    NOT NULL,
    pdb_gz     BLOB    NOT NULL,
    z_score    REAL,
    UNIQUE (ddi_id, protein1, protein2, source)
);

HDF5 output structure (required by downstream ML models):
    /<domain_id>/<protein_id> = numpy array of shape (feature_dim,)


"""

import h5py
import numpy as np
import pandas as pd
import sqlite3

from structure_utils import bytes_to_pdb_structure

from Bio.PDB.SASA import ShrakeRupley

def calculate_sasa_structure_level(domain):
    sr = ShrakeRupley()
    sr.compute(domain, level="S")  # Compute SASA at the structure level
    return domain.sasa



def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (domain, protein)
                  pair, grouped by domain_id.
    """
    domain_protein_df = pd.read_sql(
        """
        SELECT domain_id, protein_id, pdb_af_gz, pdb_rf_gz
        FROM domain_protein_map;
        """,
        conn,
    )


    domain_protein_df["domain_id"] = domain_protein_df["domain_id"].astype(str)
    domain_protein_df["protein_id"] = domain_protein_df["protein_id"].astype(str)


    for domain_id, protein_id, pdb_af_gz, pdb_rf_gz in domain_protein_df.itertuples(index=False):
        # Initialize feature vector with shape 1,
        feature_vector = np.zeros(1, dtype=np.float32)
        
        pdb_files = (pdb_af_gz, pdb_rf_gz)
        vector_list = []

        for pdb_gz in pdb_files:
            if pdb_gz is None:
                print(f"Warning: Missing PDB file for domain {domain_id}, protein {protein_id}. Skipping.")
                continue
        
            structure = bytes_to_pdb_structure(pdb_gz)

            sasa_structure = calculate_sasa_structure_level(structure)
            vector_list.append(sasa_structure)

        # Average the feature vectors from AF and RF if both are available
        if vector_list:
            feature_vector = np.mean(vector_list, axis=0)

        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = feature_vector # pyright: ignore[reportIndexIssue]

    print(f"sasa_structure: wrote {len(domain_protein_df)} entries")
