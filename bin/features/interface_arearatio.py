#!/usr/bin/env python3
"""Interface area relative to the smaller chain: 0.5 * BSA / min(SASA_A, SASA_B). Symmetric, 1 value."""
from features import structure_utils as su


def encode(cx):
    total_a, total_b, total_ab = cx.totals()
    return [0.5 * (total_a + total_b - total_ab) / min(total_a, total_b)]


extract_features = su.make_extractor(encode)
