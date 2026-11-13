#!/usr/bin/env python3
"""Size of the largest CA-contact patch (<= threshold_distance) spanning both chains.

Residues linked by CA-CA distance within or between chains; 0 if no connected
component reaches both chains. Symmetric, 1 value.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from features import structure_utils as su


def encode(cx, threshold_distance=5.0):
    n_a = len(cx.first)
    coords = np.vstack([su.ca_coords(cx.first), su.ca_coords(cx.second)])
    n = len(coords)
    pairs = cKDTree(coords).query_pairs(threshold_distance, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    in_a = np.bincount(labels[:n_a], minlength=len(sizes))
    both = (in_a > 0) & (in_a < sizes)
    return [sizes[both].max() if both.any() else 0]


extract_features = su.make_extractor(encode)
