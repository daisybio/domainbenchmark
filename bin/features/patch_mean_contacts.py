#!/usr/bin/env python3
"""Mean number of partner contacts per interface residue.

Interface residues (su.interface_residues: SASA drop > 1 A^2) at CB (CA for
GLY); inter-chain interface pairs with CB-CB <= threshold_distance (A),
divided by the number of interface residues on both chains. Symmetric,
1 value; no interface gives 0.
"""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su


def encode(cx, threshold_distance=5.0):
    idx_first, idx_second = su.interface_residues(cx)
    n = len(idx_first) + len(idx_second)
    if len(idx_first) == 0 or len(idx_second) == 0:
        return [0.0]  # no partner to contact
    d = cdist(su.cb_coords(cx.first)[idx_first], su.cb_coords(cx.second)[idx_second])
    return [np.count_nonzero(d <= threshold_distance) / n]


extract_features = su.make_extractor(encode)
