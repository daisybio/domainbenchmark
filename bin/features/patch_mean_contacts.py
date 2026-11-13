#!/usr/bin/env python3
"""Inter-chain CA-CA contacts (<= threshold_distance) per residue of the complex. Symmetric, 1 value."""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su


def encode(cx, threshold_distance=5.0):
    contacts = np.count_nonzero(cdist(su.ca_coords(cx.first), su.ca_coords(cx.second)) <= threshold_distance)
    return [contacts / len(cx.residues())]


extract_features = su.make_extractor(encode)
