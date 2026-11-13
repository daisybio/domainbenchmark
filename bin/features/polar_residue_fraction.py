#!/usr/bin/env python3
"""Fraction of SER/THR/ASN/GLN/TYR residues in the complex. Symmetric, 1 value."""
from features import structure_utils as su

POLAR = {"SER", "THR", "ASN", "GLN", "TYR"}


def encode(cx):
    residues = cx.residues()
    return [sum(res.get_resname() in POLAR for res in residues) / len(residues)]


extract_features = su.make_extractor(encode)
