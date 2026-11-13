#!/usr/bin/env python3
"""Radius of gyration of the complex over all atoms (unweighted, geometric centre). Symmetric, 1 value."""
import numpy as np

from features import structure_utils as su


def radius_of_gyration(cx):
    coords = np.array([a.coord for a in cx.atoms()], dtype=np.float64)
    diff = coords - coords.mean(axis=0)
    return float(np.sqrt(np.einsum("ij,ij->i", diff, diff).mean()))


def encode(cx):
    return [radius_of_gyration(cx)]


extract_features = su.make_extractor(encode)
