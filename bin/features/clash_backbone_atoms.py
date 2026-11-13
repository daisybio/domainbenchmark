#!/usr/bin/env python3
"""Inter-chain backbone (N, CA, C, O) atom pairs closer than r_i + r_j - 0.5 A. Symmetric, 1 value."""
from features import structure_utils as su
from features.clash_heavy_atoms import count_clashes

VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52}


def encode(cx):
    def backbone(chain):
        return [a for a in chain.get_atoms() if a.get_id() in su.BACKBONE]

    return [count_clashes(cx, backbone(cx.first), backbone(cx.second), VDW_RADII)]


extract_features = su.make_extractor(encode)
