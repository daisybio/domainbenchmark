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
    (+ embedding columns like esm3_per_domain, esmc_per_residue, etc.)

    CREATE TABLE IF NOT EXISTS domain_structure (
        id         INTEGER PRIMARY KEY AUTOINCREMENT, 
        ddi_id INTEGER NOT NULL REFERENCES domain_domain_interaction(id),
        domain_id_a TEXT NOT NULL REFERENCES domain(id),
        domain_id_b TEXT NOT NULL REFERENCES domain(id),
        protein1 INTEGER NOT NULL REFERENCES protein(id),
        protein2 INTEGER NOT NULL REFERENCES protein(id),
        source     TEXT    NOT NULL,
        pdb_gz     BLOB    NOT NULL,
        UNIQUE (ddi_id, protein1, protein2, source)
    );

HDF5 output structure (required by downstream ML models):
    /<domain_id_a _ domain_id_b>/<protein_id_a protein_id_b> = numpy array of shape (feature_dim,)

Features are saved double, once for af, once for rf


"""

import h5py
import numpy as np
import pandas as pd
import sqlite3

from structure_utils import bytes_to_pdb_structure


def build_interchain_distance_matrix(structA, structB):
    # Build a distance matrix for inter-chain residue pairs
    residuesA = list(structA.get_residues())
    residuesB = list(structB.get_residues())
    distance_matrix = [[float('inf') for _ in residuesB] for _ in residuesA]
    
    for i, resA in enumerate(residuesA):
        for j, resB in enumerate(residuesB):
            coord_resA = resA['CA'].get_coord() if 'CA' in resA else None
            coord_resB = resB['CA'].get_coord() if 'CA' in resB else None
            if coord_resA is not None and coord_resB is not None:
                dist = np.linalg.norm(coord_resA - coord_resB)
                distance_matrix[i][j] = float(dist)
    
    return distance_matrix


def count_contacts(distance_matrix, threshold):
    count = 0
    for row in distance_matrix:
        for dist in row:
            if dist <= threshold:
                count += 1
    return count


def normalize_contacts(count, lenA, lenB):
    return count / (lenA + lenB) if (lenA + lenB) > 0 else 0



def contacts(structA, structB, threshold=5.0):
    distance_matrix = build_interchain_distance_matrix(structA, structB)
    contact_count = count_contacts(distance_matrix, threshold)
    normalized_contact_count = normalize_contacts(contact_count, len(list(structA.get_residues())), len(list(structB.get_residues())))
    return normalized_contact_count


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                  pair, grouped by ddi_id.
    """
    domain_structure = pd.read_sql(
        """
        SELECT domain_id_a, domain_id_b, protein_id_a, protein_id_b, source, pdb_gz
        FROM domain_structure;
        """,
        conn,
    )


    domain_structure["domain_id_a"] = domain_structure["domain_id_a"].astype(str)
    domain_structure["domain_id_b"] = domain_structure["domain_id_b"].astype(str)
    domain_structure["protein_id_a"] = domain_structure["protein_id_a"].astype(str)
    domain_structure["protein_id_b"] = domain_structure["protein_id_b"].astype(str)


    for domain_id_a, domain_id_b, protein_id_a, protein_id_b, source, pdb_gz in domain_structure.itertuples(index=False):

        feature_vector = np.zeros(1, dtype=np.float32)  # 20 amino acids

        structure = bytes_to_pdb_structure(pdb_gz)

        structA = structure[0]["A"]
        structB = structure[0]["B"]

        
        contact_count = contacts(structA, structB, threshold=5.0)
        feature_vector = np.array([contact_count], dtype=np.float32)

        def write_to_h5(domain_key, protein_key):
            # create a group for each pfam_id and put uniprot_id as a subgroup
            if domain_key not in out_file:
                pfam_group = out_file.create_group(domain_key)
            else:
                pfam_group = out_file[domain_key]

            if protein_key in pfam_group:
                print(f"Warning: Duplicate entry for {protein_key} in {domain_key}.")
            else:
                pfam_group[protein_key] = feature_vector  # pyright: ignore[reportIndexIssue]

        # write both directions
        write_to_h5(f"{domain_id_a}_{domain_id_b}", f"{protein_id_a}_{protein_id_b}")
        write_to_h5(f"{domain_id_b}_{domain_id_a}", f"{protein_id_b}_{protein_id_a}")

        
    print(f"patch_mean_contacts: wrote {len(domain_structure)} entries")
