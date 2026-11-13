#!/usr/bin/env python3
"""Mean of the res_num smallest inter-chain CA-CA distances (all, if fewer). Symmetric, 1 value."""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su


def encode(cx, res_num=5):
    d = cdist(su.ca_coords(cx.first), su.ca_coords(cx.second)).ravel()
    if d.size > res_num:
        d = np.partition(d, res_num - 1)[:res_num]
    return [d.mean()]


extract_features = su.make_extractor(encode)
