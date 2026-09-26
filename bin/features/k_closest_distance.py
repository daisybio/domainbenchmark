#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings



def build_interchain_distance_matrix(structure) :
    structA = structure[0]["A"]
    structB = structure[0]["B"]
    
    # Build a distance matrix for inter-chain residue pairs
    residuesA = list(structA.get_residues())
    residuesB = list(structB.get_residues())
    distance_matrix = [[float('inf') for _ in residuesB] for _ in residuesA]
    
    for i, resA in enumerate(residuesA):
        for j, resB in enumerate(residuesB):
            coord_resA = resA['CA'].get_coord() if 'CA' in resA else None
            coord_resB = resB['CA'].get_coord() if 'CA' in resB else None
            if coord_resA is not None and coord_resB is not None:
                dist = np.linalg.norm(coord_resA - coord_resB)
                distance_matrix[i][j] = float(dist)
    
    return distance_matrix


def get_mean_distances(distance_matrix, k):
    # Get mean distances for the k closest inter-chain residue pairs
    flat_distances = [dist for row in distance_matrix for dist in row if dist != float('inf')]
    flat_distances.sort()
    return np.mean(flat_distances[:k]) if len(flat_distances) >= k else np.mean(flat_distances)



def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File, res_num: int = 5):
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
    for (ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        distance_matrix = build_interchain_distance_matrix(structure)
        mean_distances = get_mean_distances(distance_matrix, k=res_num)  # Example: get mean of res_num closest distances
        feature_vector = np.array([mean_distances], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"k_closest_distance_{res_num}: wrote {n_written} entries")