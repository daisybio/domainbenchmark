#!/usr/bin/env python3
"""Inter-chain salt bridges: residue pairs with a cationic N within 4.0 A of an anionic O.

Cationic N: LYS NZ; ARG NE, NH1, NH2. Anionic O: ASP OD1, OD2; GLU OE1, OE2.
HIS is excluded (its protonation at pH 7 is ambiguous). Counts unordered
inter-chain residue pairs (several close atom pairs count once); intra-chain
pairs are not counted. A LYS/ARG/ASP/GLU lacking one of these atoms raises.
Symmetric, 1 value.
"""
import numpy as np
from scipy.spatial.distance import cdist

from features import structure_utils as su

MAX_DISTANCE = 4.0
CATIONIC = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2")}
ANIONIC = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}


def charged_atoms(cx, chain, table):
    """(residue index per atom, coords) of `table`'s atoms in `chain`."""
    owner, coords = [], []
    for i, res in enumerate(chain):
        for name in table.get(res.get_resname(), ()):
            if name not in res:
                cx.fail(f"{res.get_resname()} {res.get_full_id()} has no {name} atom")
            owner.append(i)
            coords.append(res[name].coord)
    return np.array(owner, dtype=int), np.array(coords, dtype=np.float64).reshape(-1, 3)


def bridges(cx, cation_chain, anion_chain):
    own_n, n = charged_atoms(cx, cation_chain, CATIONIC)
    own_o, o = charged_atoms(cx, anion_chain, ANIONIC)
    if len(n) == 0 or len(o) == 0:
        return set()
    i, j = np.nonzero(cdist(n, o) <= MAX_DISTANCE)
    return set(zip(own_n[i].tolist(), own_o[j].tolist()))


def encode(cx):
    # A residue is either cationic or anionic, so the two directions never share a pair.
    return [len(bridges(cx, cx.first, cx.second)) + len(bridges(cx, cx.second, cx.first))]


extract_features = su.make_extractor(encode)
