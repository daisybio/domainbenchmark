#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
from Bio.PDB.Polypeptide import is_aa
from features import embeddings
from .structure_utils import bytes_to_pdb_structure, pool_residue_feature

# Max C(i-1)-N(i) distance (Angstrom) to treat two residues as peptide-bonded.
# The real bond is ~1.33 A; anything larger is a chain break / gap.
MAX_PEPTIDE_BOND = 2.0


def _backbone_arrays(structure):
    """Collect N, CA, C coordinates of all standard amino acids.

    Residues with a missing backbone atom get NaN coordinates, so every
    angle that depends on them becomes NaN automatically.
    """
    nan3 = np.full(3, np.nan, dtype=np.float64)
    n_xyz, ca_xyz, c_xyz, chain_key = [], [], [], []
    for res in structure.get_residues():
        if not is_aa(res, standard=True):
            continue
        n_xyz.append(res["N"].coord if "N" in res else nan3)
        ca_xyz.append(res["CA"].coord if "CA" in res else nan3)
        c_xyz.append(res["C"].coord if "C" in res else nan3)
        chain_key.append(id(res.get_parent()))
    return (np.asarray(n_xyz, dtype=np.float64).reshape(-1, 3),
            np.asarray(ca_xyz, dtype=np.float64).reshape(-1, 3),
            np.asarray(c_xyz, dtype=np.float64).reshape(-1, 3),
            np.asarray(chain_key, dtype=np.int64))


def _dihedral(p0, p1, p2, p3):
    """Vectorized dihedral angle in degrees (IUPAC sign convention)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def _phi_psi(structure):
    """Return (phi, psi) arrays in degrees, one entry per standard residue.

    Missing angles (chain termini, chain breaks, missing backbone atoms)
    are NaN.
    """
    N, CA, C, chain = _backbone_arrays(structure)
    n = len(CA)
    phi = np.full(n, np.nan, dtype=np.float32)
    psi = np.full(n, np.nan, dtype=np.float32)
    if n < 2:
        return phi, psi

    with np.errstate(invalid="ignore", divide="ignore"):
        # residue i is linked to i+1 if same chain and C(i)-N(i+1) is a real bond
        linked = (chain[1:] == chain[:-1]) & \
                 (np.linalg.norm(N[1:] - C[:-1], axis=1) < MAX_PEPTIDE_BOND)

        phi_vals = _dihedral(C[:-1], N[1:], CA[1:], C[1:])   # for residues 1..n-1
        psi_vals = _dihedral(N[:-1], CA[:-1], C[:-1], N[1:])  # for residues 0..n-2

    phi[1:] = np.where(linked, phi_vals, np.nan)
    psi[:-1] = np.where(linked, psi_vals, np.nan)
    return phi, psi


def calculate_phi(structure):
    return _phi_psi(structure)[0]


def calculate_psi(structure):
    return _phi_psi(structure)[1]


def _circular_features(angles_deg):
    """Pool sin and cos of the angles separately and concatenate.

    Angles are circular (179 and -179 degrees are neighbours), so pooling raw
    degrees is not meaningful. NaN residues (termini, chain breaks) are dropped
    instead of being filled with a fake value.
    Returns None if there are no valid angles.
    """
    valid = angles_deg[~np.isnan(angles_deg)]
    if valid.size == 0:
        return None
    rad = np.radians(valid).astype(np.float32)
    sin_pooled = np.asarray(pool_residue_feature(np.sin(rad)), dtype=np.float32).ravel()
    cos_pooled = np.asarray(pool_residue_feature(np.cos(rad)), dtype=np.float32).ravel()
    return np.concatenate([sin_pooled, cos_pooled])


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File, angle_type: str = "phi"):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection (read-only).
        out_file: Writable HDF5 file, one dataset per (ddi, ppi) pair.
        struct_file: Readable HDF5 file holding the DDI structures.
        angle_type: 'phi' or 'psi'.

    The feature vector is [pooled sin(angle), pooled cos(angle)].
    """
    if angle_type not in ("phi", "psi"):
        raise ValueError(f"Invalid angle_type: {angle_type}. Must be 'phi' or 'psi'.")

    ddi_df = embeddings.load_structure_data(conn)
    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return

    n_written = 0
    n_degenerate = 0
    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):
        pdb_gz = embeddings.read_interaction_instance(
            struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            continue
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        angles = calculate_phi(structure) if angle_type == "phi" else calculate_psi(structure)

        feature_vector = _circular_features(angles)
        if feature_vector is None:
            n_degenerate += 1
            continue  # don't write an all-zero feature

        embeddings.write_interaction_instance(
            out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"angles {angle_type}: wrote {n_written} entries, skipped {n_degenerate} without valid angles")
    if n_written == 0:
        raise RuntimeError(f"No valid {angle_type} angles computed for any entry")