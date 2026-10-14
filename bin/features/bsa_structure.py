#!/usr/bin/env python3
import os
import h5py
import numpy as np
import sqlite3

from features import embeddings
from .sasa_cache import SasaCache


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, normalize: bool = False):
    """Extract features from the database and write them to the HDF5 file.
    
    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
                optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                    pair, grouped by ddi_id.
        normalize: If True, normalize the BSA by the total SASA of the two chains.
    """

    path = os.environ.get("SASA_CACHE_H5")
    if not path:
        raise RuntimeError("Set SASA_CACHE_H5 to the file written by features.sasa_cache")
    cache = SasaCache(path)
    
    ddi_df = embeddings.load_structure_data(conn)

    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return

    n_written = n_skipped = 0
    for (_ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        b = cache.get(pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if b is None:
            # No structure, or < 2 chains (skipped when the cache was built)
            n_skipped += 1
            continue

        total_AB = b["total_AB"]
        total_A = b["total_A"]
        total_B = b["total_B"]
        

        bsa = (total_A + total_B) - total_AB
        if normalize:
            # Normalize the feature vector by the total SASA of the two chains
            bsa_normalized = bsa / (total_A + total_B) if (total_A + total_B) > 0 else 0.0
            feature_vector = np.array([bsa_normalized], dtype=np.float32)
        else:
            feature_vector = np.array([bsa], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"bsa: wrote {n_written} entries, skipped {n_skipped} entries")
        