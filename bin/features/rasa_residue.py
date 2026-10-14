#!/usr/bin/env python3
import os
import sqlite3

import h5py
import numpy as np

from features import embeddings
from .sasa_cache import SasaCache, MAX_SASA
from .structure_utils import pool_residue_feature


def calculate_rasa_residue_level(resnames, sasa_residue):
    """Relative SASA per residue (SASA / MaxSASA, Tien et al. 2013).

    Residues without a MaxSASA entry (e.g. HOH, non-standard) get 0.
    """
    rsa_residue = np.zeros(len(resnames), dtype=np.float32)
    for i, resname in enumerate(resnames):
        max_sasa = MAX_SASA.get(resname, None)
        if max_sasa is not None and max_sasa > 0:
            rsa_residue[i] = sasa_residue[i] / max_sasa
    return rsa_residue


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, complex: bool = False):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                  pair, grouped by ddi_id.
        complex: If True, RSA is computed from the residue SASA inside the
                 A+B complex and pooled over all residues. If False, RSA is
                 computed from each chain's isolated SASA, pooled per chain,
                 and the two pooled vectors are concatenated [A, B].
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

        if complex:
            rsa = calculate_rasa_residue_level(b["resname"], b["sasa_cplx"])
            feature_vector = pool_residue_feature(rsa)
        else:
            rsa = calculate_rasa_residue_level(b["resname"], b["sasa_iso"])
            is_a = b["chain"] == "A"
            is_b = b["chain"] == "B"
            feature_vector = np.concatenate([
                pool_residue_feature(rsa[is_a]),
                pool_residue_feature(rsa[is_b]),
            ])

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b,
                                              instance_id_a, instance_id_b,
                                              feature_vector)
        n_written += 1

    print(f"rasa_residue: wrote {n_written} entries, skipped {n_skipped} entries")