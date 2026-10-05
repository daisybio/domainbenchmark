#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
from scipy.spatial.distance import cdist

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


def _ca_coords(chain):
    """CA coordinates of all residues in a chain that have a CA atom, shape (n, 3)."""
    coords = [res['CA'].get_coord() for res in chain.get_residues() if 'CA' in res]
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def patch_mean_contacts(structure, threshold):
    """Number of inter-chain CA-CA contacts (<= threshold), normalized by the
    total number of residues in both chains (including residues without a CA)."""
    structA = structure[0]["A"]
    structB = structure[0]["B"]

    len_a = len(structA)  # number of residues, same as len(list(get_residues()))
    len_b = len(structB)
    total = len_a + len_b
    if total == 0:
        return 0.0

    coords_a = _ca_coords(structA)
    coords_b = _ca_coords(structB)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0

    contact_count = np.count_nonzero(cdist(coords_a, coords_b) <= threshold)
    return float(contact_count) / total


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File, threshold_distance: float = 5.0):
    """Extract features from the database and write them to the HDF5 file.

        Args:
            conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
                  optimization.sqlite3. Read-only — do not write.
            out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                      pair, grouped by ddi_id.
            struct_file: Readable HDF5 file holding the DDI structures, keyed
                         as embeddings.interaction_group_name /
                         interaction_dataset_name describe.
            threshold_distance: Distance threshold for counting contacts.
        """
    ddi_df = embeddings.load_structure_data(conn)

    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return

    n_written = 0
    n_skipped = 0
    for (ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]
        if len(structure[0]) < 2: # pyright: ignore[reportOptionalSubscript]
            print(f"Warning: Structure for ddi_id {ddi_id} has less than 2 chains. Skipping. (instance_id_a: {instance_id_a}, instance_id_b: {instance_id_b}, pfam_id_a: {pfam_id_a}, pfam_id_b: {pfam_id_b})")
            n_skipped += 1
            continue

        contact_count = patch_mean_contacts(structure, threshold_distance)
        feature_vector = np.array([contact_count], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"patch_mean_contacts: wrote {n_written} entries, skipped {n_skipped} entries")