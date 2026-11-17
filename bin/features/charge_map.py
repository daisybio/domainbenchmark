#!/usr/bin/env python3
"""Charge-complementarity map across the interface.

Rows = first-chain interface residues, columns = second-chain interface
residues (su.interface_residues: SASA drop > 1 A^2), PDB order.
CCI_ij = 11 - |(pI_i-7)(pI_j-7) * 19/33.8|, dimensionless, in [0.94, 11].
Encoded rectangular by su.encode_matrix: 35 values, chain-ordered (the
swapped orientation is a permutation). No interface residue on either chain
gives zeros(35), which no real map can produce (every CCI > 0).
"""
import numpy as np

from features import structure_utils as su

ISOELECTRIC_POINT = {
    "ALA": 6.0, "ARG": 10.76, "ASN": 5.41, "ASP": 2.77, "CYS": 5.07, "GLN": 5.65,
    "GLU": 3.22, "GLY": 5.97, "HIS": 7.59, "ILE": 6.02, "LEU": 5.98, "LYS": 9.74,
    "MET": 5.74, "PHE": 5.48, "PRO": 6.30, "SER": 5.68, "THR": 5.60, "TRP": 5.89,
    "TYR": 5.66, "VAL": 5.98,
}
WIDTH = 35  # su.encode_matrix, rectangular


def interface_values(cx, table):
    """`table` values of the (first, second) interface residues."""
    return tuple(np.array([table[name] for name in su.resnames(chain)], dtype=np.float64)[idx]
                 for chain, idx in zip((cx.first, cx.second), su.interface_residues(cx)))


def encode(cx):
    a, b = interface_values(cx, ISOELECTRIC_POINT)
    if a.size == 0 or b.size == 0:
        return np.zeros(WIDTH)
    a, b = a - 7.0, b - 7.0
    return su.encode_matrix(11.0 - np.abs(a[:, None] * b[None, :] * 19.0 / 33.8), symmetric=False)


extract_features = su.make_extractor(encode)
