#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


def get_ca_coords(chain) -> np.ndarray:
    """Return an (n, 3) float32 array of CA coordinates (residues without CA are skipped)."""
    return np.array(
        [res["CA"].get_coord() for res in chain.get_residues() if "CA" in res],
        dtype=np.float32,
    ).reshape(-1, 3)


def build_interchain_distance_matrix(structure) -> np.ndarray:
    """Vectorized CA-CA distance matrix (n_A x n_B) between chains A and B."""
    coords_a = get_ca_coords(structure[0]["A"])
    coords_b = get_ca_coords(structure[0]["B"])
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.empty((0, 0), dtype=np.float32)

    # Squared distances via broadcasting; sqrt only on the values we need later.
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def get_mean_distances(distance_matrix: np.ndarray, k: int) -> float:
    """Mean of the k smallest inter-chain distances (or of all, if fewer than k)."""
    flat = distance_matrix.ravel()
    if flat.size == 0:
        return float("nan")
    if flat.size > k:
        flat = np.partition(flat, k - 1)[:k]
    return float(flat.mean())


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int,
                     struct_file: h5py.File, res_num: int = 5):
    """Extract features from the database and write them to the HDF5 file.

        Args:
            conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
                  optimization.sqlite3. Read-only — do not write.
            out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                      pair, grouped by ddi_id.
            struct_file: Readable HDF5 file holding the DDI structures, keyed
                         as embeddings.interaction_group_name /
                         interaction_dataset_name describe.
            res_num: Number of closest residues to consider for distance calculation.
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
            print(f"Warning: Structure for ddi_id {ddi_id} has less than 2 chains. Skipping. "
                  f"(instance_id_a: {instance_id_a}, instance_id_b: {instance_id_b}, "
                  f"pfam_id_a: {pfam_id_a}, pfam_id_b: {pfam_id_b})")
            n_skipped += 1
            continue

        distance_matrix = build_interchain_distance_matrix(structure)
        mean_distance = get_mean_distances(distance_matrix, k=res_num)
        feature_vector = np.array([mean_distance], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"k_closest_distance_{res_num}: wrote {n_written} entries, skipped {n_skipped} entries")