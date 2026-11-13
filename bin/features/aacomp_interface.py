#!/usr/bin/env python3
"""Interface amino-acid composition: share of the buried area per residue type.

Residues losing more than 1 A^2 on complex formation, buried area summed per
type over both chains and divided by the total buried area (2x interface area).
Symmetric, 20 values.
"""
import numpy as np

from features import structure_utils as su

AA3 = tuple(sorted(su.MAX_SASA))
MIN_DECREASE = 1.0


def encode(cx):
    s1, s2 = cx.sasa()
    decrease = np.concatenate([s1["iso"] - s1["cplx"], s2["iso"] - s2["cplx"]])
    names = np.array(su.resnames(cx.first) + su.resnames(cx.second))
    total_a, total_b, total_ab = cx.totals()
    buried = total_a + total_b - total_ab
    keep = decrease > MIN_DECREASE
    if not keep.any():
        # No interface (chains not in contact, common for AF3 negatives): genuinely
        # no buried area of any type, the same 0 that bsa_structure reports.
        return np.zeros(len(AA3))
    return [decrease[keep & (names == aa)].sum() / buried for aa in AA3]


extract_features = su.make_extractor(encode)
