#!/usr/bin/env python3
"""Radius of gyration divided by the number of residues of the complex. Symmetric, 1 value."""
from features import structure_utils as su
from features.radius_of_gyration import radius_of_gyration


def encode(cx):
    return [radius_of_gyration(cx) / len(cx.residues())]


extract_features = su.make_extractor(encode)
