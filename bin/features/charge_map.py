#!/usr/bin/env python3
"""Charge-complementarity map of each chain's surface shell.

Shell = residues with RSA (chain alone) > 0.2. CCI_ij = 11 - |(pI_i-7)(pI_j-7) * 19/33.8|,
encoded by su.encode_matrix per chain. Chain-ordered: [first (32), second (32)].
"""
import numpy as np

from features import structure_utils as su

ISOELECTRIC_POINT = {
    "ALA": 6.0, "ARG": 10.76, "ASN": 5.41, "ASP": 2.77, "CYS": 5.07, "GLN": 5.65,
    "GLU": 3.22, "GLY": 5.97, "HIS": 7.59, "ILE": 6.02, "LEU": 5.98, "LYS": 9.74,
    "MET": 5.74, "PHE": 5.48, "PRO": 6.30, "SER": 5.68, "THR": 5.60, "TRP": 5.89,
    "TYR": 5.66, "VAL": 5.98,
}
RSA_SHELL = 0.2


def shell_values(cx, chain, sasa, table):
    names = np.array(su.resnames(chain))[su.rsa(chain, sasa["iso"]) > RSA_SHELL]
    if names.size == 0:
        cx.fail(f"chain {chain.id} has no surface residue (RSA > {RSA_SHELL})")
    return np.array([table[name] for name in names])


def encode(cx):
    out = []
    for chain, sasa in zip((cx.first, cx.second), cx.sasa()):
        d = shell_values(cx, chain, sasa, ISOELECTRIC_POINT) - 7.0
        out.append(su.encode_matrix(11.0 - np.abs(d[:, None] * d[None, :] * 19.0 / 33.8), symmetric=True, banded=True))
    return np.concatenate(out)


extract_features = su.make_extractor(encode)
