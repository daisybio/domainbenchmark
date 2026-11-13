#!/usr/bin/env python3
"""CA-CA contact maps (<= 5 A): intra-chain for each chain plus the inter-chain map.

Chain-ordered: [intra first (32), intra second (32), inter first x second (38)] = 102.
"""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su

CONTACT_DISTANCE = 5.0


def intra(chain):
    ca = su.ca_coords(chain)
    m = (cdist(ca, ca) <= CONTACT_DISTANCE).astype(np.float64)
    np.fill_diagonal(m, 0.0)
    return su.encode_matrix(m, symmetric=True, banded=True)


def encode(cx):
    inter = (cdist(su.ca_coords(cx.first), su.ca_coords(cx.second)) <= CONTACT_DISTANCE).astype(np.float64)
    return np.concatenate([intra(cx.first), intra(cx.second), su.encode_matrix(inter, symmetric=False)])


extract_features = su.make_extractor(encode)
