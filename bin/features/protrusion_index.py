#!/usr/bin/env python3
"""Per-atom protrusion index (Vext / Vint in a 10 A sphere), pooled over the complex. Symmetric, 6 values."""
import numpy as np
from scipy.spatial import cKDTree

from features import structure_utils as su

RADIUS = 10.0
ATOM_VOLUME = 20.1  # mean atomic volume in proteins, A^3


def encode(cx):
    coords = np.array([a.coord for a in cx.atoms()], dtype=np.float64)
    tree = cKDTree(coords)
    neighbours = np.array([len(hits) - 1 for hits in tree.query_ball_point(coords, RADIUS)])
    if (neighbours == 0).any():
        cx.fail(f"an atom has no neighbour within {RADIUS} A: protrusion index undefined")
    v_int = neighbours * ATOM_VOLUME
    v_ext = 4.0 / 3.0 * np.pi * RADIUS ** 3 - v_int
    return su.pool(v_ext / v_int)


extract_features = su.make_extractor(encode)
