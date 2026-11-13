#!/usr/bin/env python3
"""Cysteine pairs (within or between chains) with SG-SG < 2.25 A. Symmetric, 1 value."""
import numpy as np
from scipy.spatial.distance import pdist

from features import structure_utils as su

MAX_SG_DISTANCE = 2.25


def encode(cx):
    cys = [res for res in cx.residues() if res.get_resname() == "CYS"]
    for res in cys:
        if "SG" not in res:
            cx.fail(f"CYS {res.get_full_id()} has no SG atom")
    if len(cys) < 2:
        return [0]
    sg = np.array([res["SG"].coord for res in cys], dtype=np.float64)
    return [int((pdist(sg) < MAX_SG_DISTANCE).sum())]


extract_features = su.make_extractor(encode)
