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
    domain_id       TEXT    -- Pfam domain ID (e.g. PF00001)
    protein_id      TEXT    -- UniProt protein ID (e.g. P12345)
    domain_sequence TEXT    -- amino acid sequence of the domain
    start_pos       INT    -- domain start position in protein sequence
    end_pos         INT    -- domain end position in protein sequence
    pdb_af_gz        BLOB    -- gzipped PDB file of the domain structure (from AlphaFold3)
    pdb_rf_gz        BLOB    -- gzipped PDB file of the domain structure (from RoseTTAFold2)
    (+ embedding columns like esm3_per_domain, esmc_per_residue, etc.)

    CREATE TABLE IF NOT EXISTS domain_structure (
        id         INTEGER PRIMARY KEY AUTOINCREMENT, 
        ddi_id INTEGER NOT NULL REFERENCES domain_domain_interaction(id),
        protein1 INTEGER NOT NULL REFERENCES protein(id),
        protein2 INTEGER NOT NULL REFERENCES protein(id),
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

from structure_utils import bytes_to_pdb_structure, calculate_rsa_residue_level, encode_weighted_graph, embed_fingerprints_single

isoelectric_points = {
    'ALA': 6.0, 'ARG': 10.76, 'ASN': 5.41, 'ASP': 2.77,
    'CYS': 5.07, 'GLN': 5.65, 'GLU': 3.22, 'GLY': 5.97,
    'HIS': 7.59, 'ILE': 6.02, 'LEU': 5.98, 'LYS': 9.74,
    'MET': 5.74, 'PHE': 5.48, 'PRO': 6.30, 'SER': 5.68,
    'THR': 5.60, 'TRP': 5.89, 'TYR': 5.66, 'VAL': 5.98
}


def calculate_min_heavy_atom_distance(res1, res2):
    min_dist = float('inf')
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            dist = atom1 - atom2
            if dist < min_dist:
                min_dist = dist
    return min_dist

def define_shell(domain, rsa_threshold=0.2):
    rsa_residue = calculate_rsa_residue_level(domain)
    shell_residues = {rid for rid, rsa in rsa_residue.items() if rsa is not None and rsa > rsa_threshold}
    return shell_residues



def build_shell_cci_graph(shell_residues):
    graph = [[0.0 for _ in shell_residues] for _ in shell_residues]
    for i, res1 in enumerate(shell_residues):
        for j, res2 in enumerate(shell_residues):
            if i <= j:
                pi1 = isoelectric_points.get(res1.get_resname(), 0)
                pi2 = isoelectric_points.get(res2.get_resname(), 0)
                cci = 11 - abs(((pi1 - 7) * (pi2 - 7)) * 19 / 33.8)
                graph[i][j] = cci
                graph[j][i] = cci  # Symmetric graph
    return graph


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
        # Initialize feature vector with shape 20,
        feature_vector = np.zeros(20, dtype=np.float32)
        
        pdb_files = (pdb_af_gz, pdb_rf_gz)
        vector_list = []

        for pdb_gz in pdb_files:
            if pdb_gz is None:
                print(f"Warning: Missing PDB file for domain {domain_id}, protein {protein_id}. Skipping.")
                continue
        
            structure = bytes_to_pdb_structure(pdb_gz)

            shell_matrix = build_shell_cci_graph(define_shell(structure))

            # Call matrix encoding 
            fingerprints, adjacency_norm, fp_dict = encode_weighted_graph(shell_matrix)

            # Call fingerprint embedding
            feature_vector = embed_fingerprints_single(fingerprints, adjacency_norm, n_fingerprint=len(fp_dict))
            vector_list.append(feature_vector)

        # Average the feature vectors from AF and RF if both are available
        if vector_list:
            feature_vector = np.mean(vector_list, axis=0)

        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = feature_vector # pyright: ignore[reportIndexIssue]

    print(f"charge_map: wrote {len(domain_protein_df)} entries")
