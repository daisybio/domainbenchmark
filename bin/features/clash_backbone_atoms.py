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


"""

import h5py
import numpy as np
import pandas as pd
import sqlite3
from Bio.PDB.NeighborSearch import NeighborSearch

from .structure_utils import bytes_to_pdb_structure

vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52}

backbone_atoms = {'N', 'CA', 'C', 'O'}


def calculate_clash_backbone_atoms(structure, overlap_tolerance = 0.4):
    
    chain_a_atoms = [a for a in structure[0]['A'].get_atoms() if a.element.strip() in backbone_atoms]
    chain_b_atoms = [a for a in structure[0]['B'].get_atoms() if a.element.strip() in backbone_atoms]

    ns = NeighborSearch(chain_b_atoms)

    clashes = 0
        
    for atom_a in chain_a_atoms:
        r_a = vdw_radii.get(atom_a.element.strip(), 1.7)
        close_atoms = ns.search(atom_a.coord, r_a + max(vdw_radii.values()) - overlap_tolerance)
        for atom_b in close_atoms:
            r_b = vdw_radii.get(atom_b.element.strip(), 1.7)
            dist = atom_a - atom_b
            if dist < (r_a + r_b - overlap_tolerance):
                clashes += 1

    return clashes



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
        # Initialize feature vector with shape 1,
        feature_vector = np.zeros(1, dtype=np.float32)
        

        if pdb_gz is None:
            print(f"Warning: Missing PDB file for ddi {domain_id_a}_{domain_id_b} in ppi {protein_id_a}_{protein_id_b}. Skipping.")
            continue
        
        structure = bytes_to_pdb_structure(pdb_gz)

        clashes = calculate_clash_backbone_atoms(structure, 0.5)

        feature_vector = np.array([clashes], dtype=np.float32)

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
        
    print(f"clash_backbone_atoms: wrote {len(domain_structure)} entries")
