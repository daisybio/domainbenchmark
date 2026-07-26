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

from .sasa_structure import calculate_sasa_structure_level
from structure_utils import bytes_to_pdb_structure, calculate_sasa_residue_level


def calculate_interface_area(struct, structA, structB):
    sasa_A = calculate_sasa_structure_level(structA)
    sasa_B = calculate_sasa_structure_level(structB)
    sasa_AB =  calculate_sasa_structure_level(struct)

    interface_area = 0.5 * (sasa_A + sasa_B - sasa_AB)
    return interface_area
    

    

def identifiy_interface_residues(struct):
    # StructA has chain A
    # StructB has chain B
    structA = struct[0]["A"]
    structB = struct[0]["B"]

    
    sasa_A_unbound = calculate_sasa_residue_level(structA)
    sasa_B_unbound = calculate_sasa_residue_level(structB)


    sasa_AB = calculate_sasa_residue_level(struct)

    interface_residues_A = {}
    interface_residues_B = {}

    for residue in structA.get_residues():
        id = residue.get_id()
        # Skip HOH residues
        if residue.get_resname() == "HOH":
            continue
        sasa_decrease = sasa_A_unbound[id] - sasa_AB.get(id, 0)
        if sasa_decrease > 1.0:
            interface_residues_A[residue] = sasa_decrease

    for residue in structB.get_residues():
        id = residue.get_id()
        # Skip HOH residues
        if residue.get_resname() == "HOH":
            continue
        sasa_decrease = sasa_B_unbound[id] - sasa_AB.get(id, 0)
        if sasa_decrease > 1.0:
            interface_residues_B[residue] = sasa_decrease

    return interface_residues_A, interface_residues_B


def calculate_aa_composition(interface_residues):
    aa_count = {}
    total_residues = len(interface_residues)

    for residue in interface_residues:
        resname = residue.get_resname()
        aa_count[resname] = aa_count.get(resname, 0) + 1

    aa_composition = {res: count / total_residues for res, count in aa_count.items()}
    return aa_composition


def get_area_based_aa_composition(struct, interface_residues_a, interface_residues_b):
    # Initialize with all 20 amino acids set to 0.0
    area_based_composition = {amino_acid: 0.0 for amino_acid in "ACDEFGHIKLMNPQRSTVWY"}
    structA = struct[0]["A"]
    structB = struct[0]["B"]
    
    interface_area = calculate_interface_area(struct, structA, structB)
    factor = 1 / (2 * interface_area) if interface_area != 0 else 0

    # For each residue type, get a list with SASA decrease values for residues of that type
    sasa_decreases = {}
    for resA, decreaseA in interface_residues_a.items():
        resname = resA.get_resname()
        sasa_decreases.setdefault(resname, []).append(decreaseA)
    
    for resB, decreaseB in interface_residues_b.items():
        resname = resB.get_resname()
        sasa_decreases.setdefault(resname, []).append(decreaseB)
    
    for resname, decreases in sasa_decreases.items():
        area_based_composition[resname] = factor * sum(decreases)


    return area_based_composition




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

        feature_vector = np.zeros(20, dtype=np.float32)  # 20 amino acids

        structure = bytes_to_pdb_structure(pdb_gz)

        aacomp_interface = get_area_based_aa_composition(structure, *identifiy_interface_residues(structure))
        feature_vector = np.array([aacomp_interface[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"], dtype=np.float32)        

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

        

    print(f"aacomp_interface: wrote {len(domain_structure)} entries")
