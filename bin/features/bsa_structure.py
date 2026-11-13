#!/usr/bin/env python3
"""Buried surface area of the complex (SASA_A + SASA_B - SASA_AB).

normalize: divided by SASA_A + SASA_B. Symmetric, 1 value.
"""
from features import structure_utils as su


def encode(cx, normalize=False):
    total_a, total_b, total_ab = cx.totals()
    bsa = total_a + total_b - total_ab
    return [bsa / (total_a + total_b) if normalize else bsa]


extract_features = su.make_extractor(encode)
