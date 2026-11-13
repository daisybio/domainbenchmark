#!/usr/bin/env python3
"""Hydrophobic-complementarity map of each chain's surface shell.

Shell as in charge_map. HCI_ij = 20 - |h_i - h_j| * 19/10.6 (Kyte-Doolittle),
encoded by su.encode_matrix per chain. Chain-ordered: [first (32), second (32)].
"""
import numpy as np

from features import structure_utils as su
from features.charge_map import shell_values

KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
    "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 4.5, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": 1.6,
    "TYR": -1.3, "VAL": 4.2,
}


def encode(cx):
    out = []
    for chain, sasa in zip((cx.first, cx.second), cx.sasa()):
        h = shell_values(cx, chain, sasa, KYTE_DOOLITTLE)
        out.append(su.encode_matrix(20.0 - np.abs(h[:, None] - h[None, :]) * 19.0 / 10.6, symmetric=True, banded=True))
    return np.concatenate(out)


extract_features = su.make_extractor(encode)
