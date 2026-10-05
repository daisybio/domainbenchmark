#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


def calculate_radius_of_gyration(structure) -> float:
    """Radius of gyration over all atoms (unweighted), centered on the
    geometric center of those atoms.

    As the center of mass of the whole protein is not meaningful here, the
    center is the average of the atom coordinates themselves.
    """
    coords = np.array([atom.coord for atom in structure[0].get_atoms()], dtype=np.float64)
    if coords.shape[0] == 0:
        return float("nan")

    center = coords.mean(axis=0)
    diff = coords - center
    return float(np.sqrt(np.einsum("ij,ij->i", diff, diff).mean()))


def calculate_compactness(structure):
    R_g = calculate_radius_of_gyration(structure)
    num_residues = len(list(structure[0].get_residues()))
    compactness = R_g / num_residues
    return compactness



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
    for (ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        compactness = calculate_compactness(structure)
        feature_vector = np.array([compactness], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1
         
    print(f"compactness: wrote {n_written} entries")
