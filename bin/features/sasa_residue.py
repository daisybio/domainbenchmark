#!/usr/bin/env python3
"""Per-residue SASA, pooled.

complex: SASA inside the complex, pooled over both chains (symmetric, 6 values).
otherwise: each chain alone, pooled per chain (chain-ordered, [first, second] = 12).
"""
import numpy as np

from features import structure_utils as su


def encode(cx, complex=False):
    s1, s2 = cx.sasa()
    if complex:
        return su.pool(np.concatenate([s1["cplx"], s2["cplx"]]))
    return np.concatenate([su.pool(s1["iso"]), su.pool(s2["iso"])])


extract_features = su.make_extractor(encode)
