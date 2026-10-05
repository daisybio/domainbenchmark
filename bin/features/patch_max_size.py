#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


def _ca_coords(chain):
    """CA coordinates of all residues in a chain that have a CA atom, shape (n, 3)."""
    coords = [res['CA'].get_coord() for res in chain.get_residues() if 'CA' in res]
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def largest_patch_both(structure, threshold):
    """Size of the largest connected patch (residues linked by CA-CA <= threshold,
    within or between chains) that contains residues from both chain A and chain B.
    Residues without a CA can never be part of a patch with both chains."""
    coords_a = _ca_coords(structure[0]["A"])
    coords_b = _ca_coords(structure[0]["B"])
    n_a, n_b = len(coords_a), len(coords_b)
    if n_a == 0 or n_b == 0:
        return 0

    coords = np.vstack((coords_a, coords_b))
    n = n_a + n_b

    pairs = cKDTree(coords).query_pairs(threshold, output_type="ndarray")  # d <= threshold
    if len(pairs) == 0:
        return 0

    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(n, n),
    )
    _, labels = connected_components(graph, directed=False)

    is_b = np.arange(n) >= n_a
    sizes = np.bincount(labels)
    count_b = np.bincount(labels, weights=is_b.astype(np.float64), minlength=len(sizes))
    count_a = sizes - count_b
    both = (count_a > 0) & (count_b > 0)

    return int(sizes[both].max()) if both.any() else 0


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

        largest_patch = largest_patch_both(structure, threshold_distance)
        feature_vector = np.array([largest_patch], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"patch_max_size_{threshold_distance}: wrote {n_written} entries, skipped {n_skipped} entries")