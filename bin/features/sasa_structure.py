#!/usr/bin/env python3
"""Total SASA of the complex; normalize: per residue. Symmetric, 1 value."""
from features import structure_utils as su


def encode(cx, normalize=False):
    total = cx.totals()[2]
    return [total / len(cx.residues()) if normalize else total]


extract_features = su.make_extractor(encode)
