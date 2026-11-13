#!/usr/bin/env python3
"""Atom depth in the complex (distance to the nearest solvent-accessible atom), pooled. Symmetric, 6 values."""
import numpy as np

from features import structure_utils as su


def encode(cx):
    s1, s2 = cx.sasa()
    return su.pool(np.concatenate([s1["depth"], s2["depth"]]))


extract_features = su.make_extractor(encode)
