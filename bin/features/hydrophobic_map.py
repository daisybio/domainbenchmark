#!/usr/bin/env python3
"""Hydrophobic-complementarity map across the interface.

Rows/columns as in charge_map (first- x second-chain interface residues,
SASA drop > 1 A^2, PDB order). HCI_ij = 20 - |h_i - h_j| * 19/10.6
(Kyte-Doolittle h), dimensionless, in [3.87, 20]. Encoded rectangular by
su.encode_matrix: 35 values, chain-ordered (the swapped orientation is a
permutation). No interface residue on either chain gives zeros(35).
"""
import numpy as np

from features import structure_utils as su
from features.charge_map import WIDTH, interface_values

KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
    "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 4.5, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": 1.6,
    "TYR": -1.3, "VAL": 4.2,
}


def encode(cx):
    a, b = interface_values(cx, KYTE_DOOLITTLE)
    if a.size == 0 or b.size == 0:
        return np.zeros(WIDTH)
    return su.encode_matrix(20.0 - np.abs(a[:, None] - b[None, :]) * 19.0 / 10.6, symmetric=False)


extract_features = su.make_extractor(encode)
