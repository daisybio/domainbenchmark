#!/usr/bin/env python3
"""Inter-chain LYS/ARG - ASP/GLU residue pairs with CA-CA <= 5.5 A. Symmetric, 1 value."""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su

THRESHOLD = 5.5
BASIC, ACIDIC = {"LYS", "ARG"}, {"ASP", "GLU"}


def encode(cx):
    n1, n2 = np.array(su.resnames(cx.first)), np.array(su.resnames(cx.second))
    opposite = (np.isin(n1, list(BASIC))[:, None] & np.isin(n2, list(ACIDIC))[None, :]) | (
        np.isin(n1, list(ACIDIC))[:, None] & np.isin(n2, list(BASIC))[None, :]
    )
    close = cdist(su.ca_coords(cx.first), su.ca_coords(cx.second)) <= THRESHOLD
    return [np.count_nonzero(opposite & close)]


extract_features = su.make_extractor(encode)
