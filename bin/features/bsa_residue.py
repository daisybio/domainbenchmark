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

from structure_utils import bytes_to_pdb_structure, calculate_sasa_residue_level, pad_vector



def calculate_bsa_residue_level(struct):

    structA = struct[0]["A"]
    structB = struct[0]["B"]
    
    sasa_A_residue = calculate_sasa_residue_level(structA)
    sasa_B_residue = calculate_sasa_residue_level(structB)

   

    sasa_complex_residue = calculate_sasa_residue_level(struct)

    bsa_residue = {}
    for residue in structA.get_residues():
        rid = residue.get_id()
        bsa_residue[rid] = (sasa_A_residue.get(rid, 0) + sasa_B_residue.get(rid, 0)) - sasa_complex_residue.get(rid, 0)

    for residue in structB.get_residues():
        rid = residue.get_id()
        # if already present (from structA), sum/overwrite appropriately
        bsa_residue[rid] = (sasa_A_residue.get(rid, 0) + sasa_B_residue.get(rid, 0)) - sasa_complex_residue.get(rid, 0)

    return bsa_residue

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

    max_length = 0
    features = {}

    for domain_id_a, domain_id_b, protein_id_a, protein_id_b, source, pdb_gz in domain_structure.itertuples(index=False):

        feature_vector = np.zeros(20, dtype=np.float32)  # 20 amino acids

        structure = bytes_to_pdb_structure(pdb_gz)

        bsa_residue = calculate_bsa_residue_level(structure)
        feature_vector = np.array(list(bsa_residue.values()), dtype=np.float32)
        max_length = max(max_length, len(feature_vector))

        features[(domain_id_a, domain_id_b, protein_id_a, protein_id_b)] = feature_vector

        

    for (domain_id_a, domain_id_b, protein_id_a, protein_id_b), feature_vector in features.items():

        feature_vector = pad_vector(feature_vector, max_length)
        
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

    

    print(f"interface_area: wrote {len(domain_structure)} entries")
