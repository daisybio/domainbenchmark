#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from features import embeddings
from .sasa_cache import SasaCache
import os

AA3 = ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
       "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR")


def aacomp_from_bundle(b, min_decrease=1.0):
    decrease = b["sasa_iso"] - b["sasa_cplx"]
    interface_area = 0.5 * (b["total_A"] + b["total_B"] - b["total_AB"])
    keep = (b["resname"] != "HOH") & (decrease > min_decrease)
    factor = 1.0 / (2.0 * interface_area) if interface_area != 0 else 0.0
    return np.array(
        [factor * decrease[keep & (b["resname"] == aa)].sum() for aa in AA3],
        dtype=np.float32,
    )


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int):
    """Extract features from the database and write them to the HDF5 file.
        
        Args:
            conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
                    optimization.sqlite3. Read-only — do not write.
            out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                        pair, grouped by ddi_id.
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
            n_skipped += 1
            continue
        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, aacomp_from_bundle(b))
        n_written += 1

    print(f"aacomp_interface: wrote {n_written} entries, skipped {n_skipped} entries")