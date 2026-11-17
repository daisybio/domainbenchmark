#!/usr/bin/env python3
"""Binary residue contact maps: CB-CB <= 8 A (CA for GLY), the CASP contact convention.

Intra-chain map per chain (diagonal zeroed), encoded square with |resseq|
bands (su.encode_matrix, 29 each); inter-chain map, rows = first chain,
columns = second chain, encoded rectangular (35). Distances in A, entries 0/1.
Chain-ordered: [intra first (29), intra second (29), inter (35)] = 93; the
swapped orientation swaps the intra halves and permutes the inter block
(encode_matrix of the transpose).
"""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su

CONTACT_DISTANCE = 8.0


def intra(chain):
    cb = su.cb_coords(chain)
    m = (cdist(cb, cb) <= CONTACT_DISTANCE).astype(np.float64)
    np.fill_diagonal(m, 0.0)
    return su.encode_matrix(m, symmetric=True, positions=su.residue_numbers(chain))


def encode(cx):
    inter = (cdist(su.cb_coords(cx.first), su.cb_coords(cx.second)) <= CONTACT_DISTANCE).astype(np.float64)
    return np.concatenate([intra(cx.first), intra(cx.second), su.encode_matrix(inter, symmetric=False)])


extract_features = su.make_extractor(encode)
