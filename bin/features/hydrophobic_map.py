#!/usr/bin/env python3
import os
import sqlite3

import h5py
import numpy as np

from features import embeddings
from .sasa_cache import SasaCache, MAX_SASA
from .structure_utils import encode_matrix


# Kyte-Doolittle hydropathy
hydrophobic_moments = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 4.5, 'LYS': -3.9,
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
    'THR': -0.7, 'TRP': 1.6, 'TYR': -1.3, 'VAL': 4.2
}


def shell_hydrophobicity(resnames, sasa_iso, rsa_threshold=0.2):
    """Hydropathy values of the surface-shell residues of ONE chain.

    Shell = residues with RSA (isolated-chain SASA / MaxSASA) > threshold, in
    residue order. Residues without a MaxSASA entry (HOH, non-standard) are
    never part of the shell.
    """
    shell_hm = []
    for resname, sasa in zip(resnames, sasa_iso):
        max_sasa = MAX_SASA.get(resname)
        if max_sasa is not None and sasa / max_sasa > rsa_threshold:
            shell_hm.append(hydrophobic_moments.get(resname, 0.0))
    return np.array(shell_hm, dtype=np.float64)


def build_shell_hci_matrix(hm):
    """Symmetric (n_shell x n_shell) HCI matrix: 20 - |hm_i - hm_j| * 19 / 10.6."""
    return 20.0 - np.abs(hm[:, None] - hm[None, :]) * 19.0 / 10.6


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
            hm = shell_hydrophobicity(b["resname"][m], b["sasa_iso"][m])
            vecs.append(encode_matrix(build_shell_hci_matrix(hm), symmetric=True, banded=True, fill=0.0))

        feature_vector = np.concatenate(vecs).astype(np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b,
                                              instance_id_a, instance_id_b,
                                              feature_vector)
        n_written += 1

    print(f"hydrophobic_map: wrote {n_written} entries, skipped {n_skipped} entries")