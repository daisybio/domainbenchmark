#!/usr/bin/env python3
import os
import sqlite3

import h5py
import numpy as np
from scipy.spatial.distance import cdist

from features import embeddings
from .sasa_cache import SasaCache
from .structure_utils import encode_matrix

CONTACT_DISTANCE_THRESHOLD = 5.0  # Angstrom, CA-CA


def chain_ca_coords(b, chain_id):
    """CA coordinates of one chain, residues without a CA dropped, in chain order."""
    ca = b["ca"][b["chain"] == chain_id].astype(np.float64)
    return ca[~np.isnan(ca).any(axis=1)]


def contact_matrix(coords_1, coords_2=None, threshold=CONTACT_DISTANCE_THRESHOLD):
    """Binary CA-CA contact matrix (1.0 where distance <= threshold).

    With one argument: intra-chain map, diagonal set to 0 (as in the old
    joint graph, which only looked at i < j).
    With two arguments: rectangular inter-chain map.
    """
    if coords_2 is None:
        m = (cdist(coords_1, coords_1) <= threshold).astype(np.float64)
        np.fill_diagonal(m, 0.0)
        return m
    return (cdist(coords_1, coords_2) <= threshold).astype(np.float64)


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                  pair, grouped by ddi_id.
        seed: Unused; kept for the common extract_features interface.
        struct_file: Unused (structures are read via the SASA cache); kept
                     for the common extract_features interface.
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
    for _ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):

        b = cache.get(pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if b is None:
            # No structure, or < 2 chains (skipped when the cache was built)
            n_skipped += 1
            continue

        coords_a = chain_ca_coords(b, "A")
        coords_b = chain_ca_coords(b, "B")
        if len(coords_a) == 0 or len(coords_b) == 0:
            print(f"Warning: Missing CA coordinates for {pfam_id_a}|{pfam_id_b}|{instance_id_a}|{instance_id_b}. Skipping.")
            n_skipped += 1
            continue

        feature_vector = np.concatenate([
            encode_matrix(contact_matrix(coords_a),symmetric = True, banded=True),
            encode_matrix(contact_matrix(coords_b), symmetric=True, banded=True),
            encode_matrix(contact_matrix(coords_a, coords_b), symmetric=False, banded=False),
        ]).astype(np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b,
                                              instance_id_a, instance_id_b,
                                              feature_vector)
        n_written += 1

    print(f"contact_map: wrote {n_written} entries, skipped {n_skipped} entries")