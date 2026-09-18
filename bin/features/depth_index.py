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

from structure_utils import bytes_to_pdb_structure, pad_vector


from Bio.PDB.SASA import ShrakeRupley

def calculate_sasa_atom_level(domain):
    sr = ShrakeRupley()
    sr.compute(domain, level="A")  # Compute SASA at the atom level
    sasa_values = {}
    for residue in domain.get_residues():
        for atom in residue.get_atoms():
            id = f"{residue.get_id()}:{atom.get_id()}"
            sasa_values[id] = atom.sasa
    return sasa_values


def calc_dist_atom_matrix(domain):
    atoms = list(domain.get_atoms())
    num_atoms = len(atoms)
    dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = atoms[i] - atoms[j]
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


def calculate_depth(domain):
    sasa_values = calculate_sasa_atom_level(domain)
    dist_matrix = calc_dist_atom_matrix(domain)
    
    depth_values = np.zeros(len(domain.get_atoms()), dtype=np.float32)
    for i, atom in enumerate(domain.get_atoms()):
        id = f"{atom.get_parent().get_id()}:{atom.get_id()}"
        if sasa_values[id] > 0:
            depth_values[i] = 0.0
        else:
            # Get distances to all solvent accessible atoms
            solvent_accessible_indices = [j for j, a in enumerate(domain.get_atoms()) if sasa_values[f"{a.get_parent().get_id()}:{a.get_id()}"] > 0]
            if solvent_accessible_indices:
                min_dist = min(dist_matrix[i, j] for j in solvent_accessible_indices)
                depth_values[i] = min_dist    
    return depth_values



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
        SELECT domain_id, protein_id, pdb_af_gz, pdb_rf_gz, start_pos, end_pos
        FROM domain_protein_map;
        """,
        conn,
    )


    domain_protein_df["domain_id"] = domain_protein_df["domain_id"].astype(str)
    domain_protein_df["protein_id"] = domain_protein_df["protein_id"].astype(str)
    domain_protein_df["start_pos"] = domain_protein_df["start_pos"].astype(int)
    domain_protein_df["end_pos"] = domain_protein_df["end_pos"].astype(int)
    # Calculate max length of a possible domain for padding, i.e.
    domain_protein_df["length"] = domain_protein_df["end_pos"] - domain_protein_df["start_pos"] + 1
    max_length = domain_protein_df["length"].max()


    for domain_id, protein_id, pdb_af_gz, pdb_rf_gz in domain_protein_df.itertuples(index=False):
        # Initialize feature vector with shape 20,
        feature_vector = np.zeros(max_length, dtype=np.float32)
        
        pdb_files = (pdb_af_gz, pdb_rf_gz)
        vector_list = []

        for pdb_gz in pdb_files:
            if pdb_gz is None:
                print(f"Warning: Missing PDB file for domain {domain_id}, protein {protein_id}. Skipping.")
                continue
        
            structure = bytes_to_pdb_structure(pdb_gz)

            depth_index = calculate_depth(structure)
            depth_index = pad_vector(depth_index, max_length)
            vector_list.append(depth_index)

        # Average the feature vectors from AF and RF if both are available
        if vector_list:
            feature_vector = np.mean(vector_list, axis=0)

        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = feature_vector # pyright: ignore[reportIndexIssue]

    print(f"depth_index: wrote {len(domain_protein_df)} entries")
