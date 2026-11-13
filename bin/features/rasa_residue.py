#!/usr/bin/env python3
"""Per-residue relative SASA (SASA / MaxSASA, Tien et al. 2013), pooled.

complex: in the complex, pooled over both chains (symmetric, 6 values).
otherwise: each chain alone, pooled per chain (chain-ordered, [first, second] = 12).
"""
import numpy as np

from features import structure_utils as su


def encode(cx, complex=False):
    s1, s2 = cx.sasa()
    if complex:
        return su.pool(np.concatenate([su.rsa(cx.first, s1["cplx"]), su.rsa(cx.second, s2["cplx"])]))
    return np.concatenate([su.pool(su.rsa(cx.first, s1["iso"])), su.pool(su.rsa(cx.second, s2["iso"]))])


extract_features = su.make_extractor(encode)
