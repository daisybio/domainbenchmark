#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure, calculate_sasa_values
from features import embeddings


def identify_interface_residues_and_area(struct):
    # StructA has chain A
    # StructB has chain B
    structA = struct[0]["A"]
    structB = struct[0]["B"]

    # One SASA pass per domain (A, B, AB) instead of two (residue-level +
    # structure-level) -- calculate_sasa_values gives both from a single
    # ShrakeRupley computation.
    sasa_A_res, sasa_A_total = calculate_sasa_values(structA)
    sasa_B_res, sasa_B_total = calculate_sasa_values(structB)
    sasa_AB_res, sasa_AB_total = calculate_sasa_values(struct)

    interface_area = 0.5 * (sasa_A_total + sasa_B_total - sasa_AB_total)

    interface_residues_A = {}
    interface_residues_B = {}

    for residue in structA.get_residues():
        # Skip HOH residues
        if residue.get_resname() == "HOH":
            continue
        id = residue.get_id()
        chain_id = residue.get_parent().get_id()
        sasa_decrease = sasa_A_res[(chain_id, id)] - sasa_AB_res.get((chain_id, id), 0)
        if sasa_decrease > 1.0:
            interface_residues_A[residue] = sasa_decrease

    for residue in structB.get_residues():
        # Skip HOH residues
        if residue.get_resname() == "HOH":
            continue
        id = residue.get_id()
        chain_id = residue.get_parent().get_id()
        sasa_decrease = sasa_B_res[(chain_id, id)] - sasa_AB_res.get((chain_id, id), 0)
        if sasa_decrease > 1.0:
            interface_residues_B[residue] = sasa_decrease

    return interface_residues_A, interface_residues_B, interface_area


def calculate_aa_composition(interface_residues):
    aa_count = {}
    total_residues = len(interface_residues)

    for residue in interface_residues:
        resname = residue.get_resname()
        aa_count[resname] = aa_count.get(resname, 0) + 1

    aa_composition = {res: count / total_residues for res, count in aa_count.items()}
    return aa_composition


def get_area_based_aa_composition(interface_area, interface_residues_a, interface_residues_b):
    # Initialize with all 20 amino acids set to 0.0
    area_based_composition = {amino_acid: 0.0 for amino_acid in "ACDEFGHIKLMNPQRSTVWY"}

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


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                  pair, grouped by ddi_id.
        struct_file: Readable HDF5 file holding the DDI structures, keyed
                     as embeddings.interaction_group_name /
                     interaction_dataset_name describe.
    """
    ddi_df = embeddings.load_structure_data(conn)

    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return

    n_written = 0
    for (
        ddi_id,
        instance_id_a,
        instance_id_b,
        pfam_id_a,
        pfam_id_b,
    ) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(
            struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b
        )
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        interface_residues_a, interface_residues_b, interface_area = identify_interface_residues_and_area(structure)
        aacomp_interface = get_area_based_aa_composition(
            interface_area, interface_residues_a, interface_residues_b
        )
        feature_vector = np.array(
            [aacomp_interface[aa] for aa in "ACDEFGHIKLMNPQRSTVWY"], dtype=np.float32
        )

        embeddings.write_interaction_instance(
            out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector
        )
        n_written += 1

    print(f"aacomp_interface: wrote {n_written} entries")