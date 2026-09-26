#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
        self.source = list(range(size))  # Track the source domain for each residue (0 for structA, 1 for structB)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1


def build_interchain_patches(structure, threshold):
    structA = structure[0]["A"]
    structB = structure[0]["B"]
    all_residues = list(structA.get_residues()) + list(structB.get_residues())
    uf = UnionFind(len(all_residues))
    # Mark source for each residue by its index in `all_residues` (0..n-1).
    structA_count = len(list(structA.get_residues()))
    for idx in range(len(all_residues)):
        uf.source[idx] = 0 if idx < structA_count else 1

    # Build Union Find strucutre for all possible contacts within & between domains
    for i, res in enumerate(all_residues):
        for j in range(i + 1, len(all_residues)):
            coord_i = res['CA'].get_coord() if 'CA' in res else None
            coord_j = all_residues[j]['CA'].get_coord() if 'CA' in all_residues[j] else None
            if coord_i is not None and coord_j is not None:
                dist = np.linalg.norm(coord_i - coord_j)
                if dist <= threshold:
                    uf.union(i, j)

    return uf, all_residues


def calculate_largest_patch(uf, all_residues):
    patch_sizes = {}
    for i in range(len(all_residues)):
        root = uf.find(i)
        if root not in patch_sizes:
            patch_sizes[root] = {'size': 0, 'source': set()}
        patch_sizes[root]['size'] += 1
        patch_sizes[root]['source'].add(uf.source[i])

    largest_patch_both = max((size['size'] for size in patch_sizes.values() if size['source'] == {0, 1}), default=0)

    return largest_patch_both



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
    for (ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        uf, all_residues = build_interchain_patches(structure, threshold_distance)
        largest_patch = calculate_largest_patch(uf, all_residues)
        feature_vector = np.array([largest_patch], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"patch_max_size_{threshold_distance}: wrote {n_written} entries")