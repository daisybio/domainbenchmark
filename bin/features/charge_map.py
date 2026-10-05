#!/usr/bin/env python3
import os
import sqlite3

import h5py
import numpy as np

from features import embeddings
from .sasa_cache import SasaCache, MAX_SASA
from .structure_utils import encode_matrix

isoelectric_points = {
    'ALA': 6.0, 'ARG': 10.76, 'ASN': 5.41, 'ASP': 2.77,
    'CYS': 5.07, 'GLN': 5.65, 'GLU': 3.22, 'GLY': 5.97,
    'HIS': 7.59, 'ILE': 6.02, 'LEU': 5.98, 'LYS': 9.74,
    'MET': 5.74, 'PHE': 5.48, 'PRO': 6.30, 'SER': 5.68,
    'THR': 5.60, 'TRP': 5.89, 'TYR': 5.66, 'VAL': 5.98
}


def shell_isoelectric_points(resnames, sasa_iso, rsa_threshold=0.2):
    """Isoelectric points of the surface-shell residues of ONE chain.

    Shell = residues with RSA (isolated-chain SASA / MaxSASA) > threshold, in
    residue order. Residues without a MaxSASA entry are never in the shell.
    """
    shell_pi = []
    for resname, sasa in zip(resnames, sasa_iso):
        max_sasa = MAX_SASA.get(resname)
        if max_sasa is not None and sasa / max_sasa > rsa_threshold:
            shell_pi.append(isoelectric_points.get(resname, 0.0))
    return np.array(shell_pi, dtype=np.float64)


def build_shell_cci_matrix(pi):
    """Symmetric (n_shell x n_shell) CCI matrix: 11 - |(pi_i-7)(pi_j-7) * 19 / 33.8|."""
    d = pi - 7.0
    return 11.0 - np.abs(d[:, None] * d[None, :] * 19.0 / 33.8)


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

        vecs = []
        for chain_id in ("A", "B"):
            m = b["chain"] == chain_id
            pi = shell_isoelectric_points(b["resname"][m], b["sasa_iso"][m])
            vecs.append(encode_matrix(build_shell_cci_matrix(pi), symmetric=True, banded=True))

        feature_vector = np.concatenate(vecs).astype(np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b,
                                              instance_id_a, instance_id_b,
                                              feature_vector)
        n_written += 1

    print(f"charge_map: wrote {n_written} entries, skipped {n_skipped} entries")