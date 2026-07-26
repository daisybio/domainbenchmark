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

from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import sqlite3

from structure_utils import bytes_to_pdb_structure, calculate_rsa_residue_level, embed_fingerprints_single, normalize_weighted_adjacency


FINGERPRINT_DICT = defaultdict(lambda: len(FINGERPRINT_DICT))



def create_fingerprints(adjacency_matrix, node_features=None, radius=1):
    """
    Create r-radius fingerprints for each node in a weighted graph.
    Similar to Weisfeiler-Lehman algorithm but for weighted graphs.
    
    Args:
        adjacency_matrix: weighted adjacency matrix (n x n)
        node_features: optional node feature vector (n,) or (n x f)
        radius: neighborhood radius (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
    
    Returns:
        fingerprints: array of fingerprint IDs for each node
    """
    adjacency = np.array(adjacency_matrix, dtype=float)
    n = adjacency.shape[0]
    
    
    # Start with node features or node indices
    if node_features is None:
        current_features = np.arange(n)
    else:
        current_features = np.array(node_features)
    
    fingerprints = []

    
    # For each node, create a fingerprint based on neighborhood
    for i in range(n):
        # Get neighbors (non-zero adjacency)
        neighbors_idx = np.where(adjacency[i] > 0.0001)[0]
        neighbor_weights = adjacency[i][neighbors_idx]
        
        # Create signature: (node_feature, sorted_neighbor_features_with_weights)
        if len(neighbors_idx) > 0:
            # Sort neighbors by weight (descending) for consistent ordering
            sorted_idx = np.argsort(-neighbor_weights)
            neighbors_sorted = neighbors_idx[sorted_idx]
            weights_sorted = neighbor_weights[sorted_idx]
            
            # Create tuple signature
            neighbor_sig = tuple((int(n_idx), round(float(w), 4)) for n_idx, w in zip(neighbors_sorted, weights_sorted))
            fingerprint = (int(current_features[i]), neighbor_sig)
        else:
            fingerprint = (int(current_features[i]),)
        
        fingerprints.append(FINGERPRINT_DICT[fingerprint])
    
    return np.array(fingerprints)


def encode_weighted_graph(adjacency_matrix, node_features=None, radius=1, normalize=True):
    """
    Encode a weighted graph using struct2graph approach:
    1. Create fingerprints (node identity + neighborhood)
    2. Normalize adjacency matrix
    
    Args:
        adjacency_matrix: weighted adjacency matrix (n x n)
        node_features: optional node labels/features
        radius: neighborhood radius for fingerprints
        normalize: whether to apply normalization
    
    Returns:
        fingerprints: array of fingerprint IDs
        adjacency_normalized: normalized adjacency matrix
        fingerprint_dict: mapping of fingerprints to IDs
    """
    fingerprints, fp_dict = create_fingerprints(adjacency_matrix, node_features, radius)
    
    if normalize:
        adjacency_norm = normalize_weighted_adjacency(adjacency_matrix)
    else:
        adjacency_norm = np.array(adjacency_matrix, dtype=float)
    
    return fingerprints, adjacency_norm, fp_dict



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



def build_shell_adjacency_graph(shell_residues, distance_threshold=14.0, d0 = 4.0):
    graph = [[0.0 for _ in shell_residues] for _ in shell_residues]
    for i, res1 in enumerate(shell_residues):
        for j, res2 in enumerate(shell_residues):
            # for i=j: s_ii = 1 according to formula
            if i == j:
                graph[i][j] = 1.0
            if i < j:
                min_dist = calculate_min_heavy_atom_distance(res1, res2)
                if min_dist <= distance_threshold:
                    graph[i][j] = (2 * d0) / (d0 + max(d0, min_dist))
                    graph[j][i] = graph[i][j]  # Symmetric graph
    return graph


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (domain, protein)
                  pair, grouped by domain_id.
    """
    domain_structure = pd.read_sql(
        """
        SELECT domain_id_a, domain_id_b, protein_id_a, protein_id_b, pdb_gz, source
        FROM domain_structure;
        """,
        conn,
    )


    domain_structure["domain_id_a"] = domain_structure["domain_id_a"].astype(str)
    domain_structure["domain_id_b"] = domain_structure["domain_id_b"].astype(str)
    domain_structure["protein_id_a"] = domain_structure["protein_id_a"].astype(str)
    domain_structure["protein_id_b"] = domain_structure["protein_id_b"].astype(str)


    encoding_prep = {}

    for domain_id_a, domain_id_b, protein_id_a, protein_id_b, pdb_gz, source in domain_structure.itertuples(index=False):
        # Initialize feature vector with shape 20,
        
        structure = bytes_to_pdb_structure(pdb_gz)

        structA = structure[0]["A"]
        structB = structure[0]["B"]


        shell_matrix_A = build_shell_adjacency_graph(define_shell(structA))
        shell_matrix_B = build_shell_adjacency_graph(define_shell(structB))

        # Call matrix encoding 
        fingerprints_a  , adjacency_norm_a, fp_dict_a = encode_weighted_graph(shell_matrix_A)
        fingerprints_b  , adjacency_norm_b, fp_dict_b = encode_weighted_graph(shell_matrix_B)

        encoding_prep[(domain_id_a, protein_id_a, source)] = (fingerprints_a, adjacency_norm_a, fp_dict_a)
        encoding_prep[(domain_id_b, protein_id_b, source)] = (fingerprints_b, adjacency_norm_b, fp_dict_b)

    for (domain_id, protein_id, source), (fingerprints, adjacency_norm, fp_dict) in encoding_prep.items():
        # Call fingerprint embedding
        feature_vector = np.zeros(20, dtype=np.float32)
        feature_vector = embed_fingerprints_single(fingerprints, adjacency_norm, n_fingerprint=len(fp_dict))

        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = feature_vector # pyright: ignore[reportIndexIssue]

    print(f"adjacency_matrix: wrote {len(encoding_prep)} entries")
