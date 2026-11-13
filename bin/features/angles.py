#!/usr/bin/env python3
"""Backbone dihedral (phi or psi) distribution over both chains.

sin and cos pooled separately (angles are circular). Angles that do not exist
(chain termini, chain breaks with C-N > 2 A) are excluded. Symmetric, 12 values.
"""
import numpy as np

from features import structure_utils as su

MAX_PEPTIDE_BOND = 2.0  # A; the real C-N bond is ~1.33


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1))


def chain_angles(chain, angle_type):
    N, CA, C = (np.array([res[a].coord for res in chain], dtype=np.float64) for a in ("N", "CA", "C"))
    linked = np.linalg.norm(N[1:] - C[:-1], axis=1) < MAX_PEPTIDE_BOND
    if angle_type == "phi":
        angles = _dihedral(C[:-1], N[1:], CA[1:], C[1:])
    else:
        angles = _dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
    return angles[linked]


def encode(cx, angle_type="phi"):
    if angle_type not in ("phi", "psi"):
        raise ValueError(f"angle_type must be 'phi' or 'psi', got {angle_type!r}")
    rad = np.concatenate([chain_angles(cx.first, angle_type), chain_angles(cx.second, angle_type)])
    if rad.size == 0:
        cx.fail(f"no defined {angle_type} angle")
    return np.concatenate([su.pool(np.sin(rad)), su.pool(np.cos(rad))])


extract_features = su.make_extractor(encode)
