#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
from features import embeddings
from .structure_utils import bytes_to_pdb_structure, pool_residue_feature


def _angle_internal_coord(residue, which):
    ic = getattr(residue, 'internal_coord', None)
    if ic is None:
        return None
    if which == 'phi':
        candidates = ('phi', 'C:N:CA:C')
    elif which == 'psi':
        candidates = ('psi', 'N:CA:C:N')
    else:
        return None
    for key in candidates:
        try:
            val = ic.get_angle(key)
            if val is not None:
                return float(val)
        except Exception:
            continue
    return None


def calculate_phi(domain):
    phi_angles = np.zeros(len(list(domain.get_residues())), dtype=np.float32)
    for i, residue in enumerate(domain.get_residues()):
        # try internal_coord first
        angle = _angle_internal_coord(residue, 'phi')
        if angle is not None:
            phi_angles[i] = angle
    return phi_angles


def calculate_psi(domain):
    psi_angles = np.zeros(len(list(domain.get_residues())), dtype=np.float32)
    for i, residue in enumerate(domain.get_residues()):
        # try internal_coord first
        angle = _angle_internal_coord(residue, 'psi')
        if angle is not None:
            psi_angles[i] = angle
    return psi_angles



def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File, angle_type: str = 'phi'):
    """Extract features from the database and write them to the HDF5 file.
    
        Args:
            conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
                  optimization.sqlite3. Read-only — do not write.
            out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                      pair, grouped by ddi_id.
            struct_file: Readable HDF5 file holding the DDI structures, keyed
                         as embeddings.interaction_group_name /
                         interaction_dataset_name describe.
            angle_type: Type of angle to calculate ('phi' or 'psi').
        """
    ddi_df = embeddings.load_structure_data(conn)
    
    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return


    n_written = 0
    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):
        
        feature_vector = np.zeros(0, dtype=np.float32)
        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}") # pyright: ignore[reportAttributeAccessIssue]

        if angle_type == 'phi':
            phi_angles = calculate_phi(structure)
            feature_vector = pool_residue_feature(np.array(phi_angles, dtype=np.float32))
        elif angle_type == 'psi':
            psi_angles = calculate_psi(structure)
            feature_vector = pool_residue_feature(np.array(psi_angles, dtype=np.float32))
        else:
            raise ValueError(f"Invalid angle_type: {angle_type}. Must be 'phi' or 'psi'.")
        
        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, 
                                            instance_id_a, instance_id_b, 
                                            feature_vector)
        n_written += 1

    print(f"angles {angle_type}: wrote {n_written} entries")    