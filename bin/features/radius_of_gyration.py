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

from structure_utils import bytes_to_pdb_structure


def get_center_of_mass(residue_coords):
    """Calculate the center of mass for a list of residues."""
    coords = np.array(residue_coords)
    center_of_mass = np.mean(coords, axis=0)
    return center_of_mass


def calculate_distances_from_center(atom_coord, center_of_mass):
    return np.linalg.norm(atom_coord - center_of_mass)



def calculate_radius_of_gyration(domain):
    """Calculate the radius of gyration for a given domain.
       As the center of mass from the whole protein is not meaningfuul here,
       the center is calculated from the domain itself as the average of the coordinates of the residues in the domain."""

    residues = domain.get_residues()
    coords = []
    for res in residues:
        for atom in res.get_atoms():
            coords.append(atom.get_vector())
    
    center = get_center_of_mass(coords)
    # Calculate the distance of each atom from the center of mass
    distances_from_center = np.array([])
    for residue in domain:
        for atom in residue.get_atoms():
            distance_from_com = calculate_distances_from_center(atom.get_vector(), center)
            distances_from_center = np.append(distances_from_center, distance_from_com)

    
    R_g = np.sqrt(np.sum(distances_from_center**2) / len(distances_from_center))
    return R_g




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

            R_g = calculate_radius_of_gyration(structure)
            vector_list.append(R_g)

        # Average the feature vectors from AF and RF if both are available
        if vector_list:
            feature_vector = np.mean(vector_list, axis=0)

        if domain_id not in out_file:
            pfam_group = out_file.create_group(domain_id)
        else:
            pfam_group = out_file[domain_id]

        pfam_group[protein_id] = feature_vector # pyright: ignore[reportIndexIssue]

    print(f"radius_of_gyration: wrote {len(domain_protein_df)} entries")
