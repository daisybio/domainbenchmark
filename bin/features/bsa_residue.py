#!/usr/bin/env python3
"""Per-residue buried surface area (alone minus in complex), pooled over both chains. Symmetric, 6 values."""
import numpy as np

from features import structure_utils as su


def encode(cx):
    s1, s2 = cx.sasa()
    return su.pool(np.concatenate([s1["iso"] - s1["cplx"], s2["iso"] - s2["cplx"]]))


extract_features = su.make_extractor(encode)
