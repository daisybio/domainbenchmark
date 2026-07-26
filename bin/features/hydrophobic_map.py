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

from structure_utils import bytes_to_pdb_structure, calculate_rsa_residue_level, embed_fingerprints_single, encode_weighted_graph

hydrophobic_moments = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 4.5, 'LYS': -3.9, 
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 
    'THR': -0.7, 'TRP': 1.6, 'TYR': -1.3, 'VAL': 4.2
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



def build_shell_hci_graph(shell_residues):
    graph = [[0.0 for _ in shell_residues] for _ in shell_residues]
    for i, res1 in enumerate(shell_residues):
        for j, res2 in enumerate(shell_residues):
            if i <= j:
                hm1 = hydrophobic_moments.get(res1.get_resname(), 0)
                hm2 = hydrophobic_moments.get(res2.get_resname(), 0)
                hci = 20 - abs((hm1 - hm2) * 19 / 10.6)
                graph[i][j] = hci
                graph[j][i] = hci  # Symmetric graph
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

            shell_matrix = build_shell_hci_graph(define_shell(structure))

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

    print(f"hydrophobic_map: wrote {len(domain_protein_df)} entries")
