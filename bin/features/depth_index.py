#!/usr/bin/env python3
import h5py
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist
from features import embeddings

from .structure_utils import bytes_to_pdb_structure, pool_residue_feature

from Bio.PDB.SASA import ShrakeRupley


def calculate_depth(domain):
    atoms = list(domain.get_atoms())
    num_atoms = len(atoms)

    if num_atoms == 0:
        return np.zeros(0, dtype=np.float32)

    sr = ShrakeRupley()
    sr.compute(domain, level="A")  # Compute SASA at the atom level

    # Pull sasa/coords from the SAME atom list/order used everywhere else,
    # instead of re-calling domain.get_atoms() and re-keying by string id.
    sasa = np.array([atom.sasa for atom in atoms], dtype=np.float64)
    coords = np.array([atom.coord for atom in atoms], dtype=np.float64)

    accessible_mask = sasa > 0
    depth_values = np.zeros(num_atoms, dtype=np.float32)

    if not accessible_mask.any():
        # No solvent-accessible atoms at all -- matches original behavior
        # (min_dist over an empty set left depth_values[i] at its 0.0 default).
        return depth_values

    dist_matrix = cdist(coords, coords)
    min_dist_to_accessible = dist_matrix[:, accessible_mask].min(axis=1)

    depth_values = np.where(accessible_mask, 0.0, min_dist_to_accessible).astype(np.float32)
    return depth_values


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
    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        depths = calculate_depth(structure)
        feature_vector = pool_residue_feature(np.array(depths, dtype=np.float32))

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b,
                                            instance_id_a, instance_id_b,
                                            feature_vector)
        n_written += 1

    print(f"depth_index: wrote {n_written} entries")