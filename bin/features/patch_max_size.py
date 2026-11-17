#!/usr/bin/env python3
"""Size (in residues) of the largest connected interface patch.

Nodes = interface residues (su.interface_residues: SASA drop > 1 A^2), placed
at CB (CA for GLY); edges = CB-CB <= threshold_distance (A, the variant's
linking distance). Chain-ordered, 3 values: [largest patch within the first
chain, within the second chain, overall with inter-chain edges]. No interface
gives [0, 0, 0]. The swapped orientation swaps elements 0 and 1.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from features import structure_utils as su


def largest_component(coords, threshold_distance):
    n = len(coords)
    if n == 0:
        return 0
    pairs = cKDTree(coords).query_pairs(threshold_distance, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return int(np.bincount(labels).max())


def encode(cx, threshold_distance=5.0):
    idx_first, idx_second = su.interface_residues(cx)
    first = su.cb_coords(cx.first)[idx_first]
    second = su.cb_coords(cx.second)[idx_second]
    return [largest_component(first, threshold_distance),
            largest_component(second, threshold_distance),
            largest_component(np.vstack([first, second]), threshold_distance)]


extract_features = su.make_extractor(encode)
