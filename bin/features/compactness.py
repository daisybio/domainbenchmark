#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


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


def calculate_compactness(domain):
    R_g = calculate_radius_of_gyration(domain)
    num_residues = len(list(domain.get_residues()))
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
