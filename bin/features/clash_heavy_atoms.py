#!/usr/bin/env python3
"""Inter-chain heavy-atom pairs closer than r_i + r_j - 0.5 A (van der Waals). Symmetric, 1 value."""
from Bio.PDB.NeighborSearch import NeighborSearch

from features import structure_utils as su

VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
OVERLAP_TOLERANCE = 0.5


def count_clashes(cx, atoms_1, atoms_2, radii):
    def radius(atom):
        element = atom.element.strip()
        if element not in radii:
            cx.fail(f"atom {atom.get_full_id()} has element {element!r} without a van der Waals radius")
        return radii[element]

    ns = NeighborSearch(atoms_2)
    reach = max(radii.values()) - OVERLAP_TOLERANCE
    return sum(
        1
        for a in atoms_1
        for b in ns.search(a.coord, radius(a) + reach)
        if a - b < radius(a) + radius(b) - OVERLAP_TOLERANCE
    )


def encode(cx):
    def heavy(chain):
        return [a for a in chain.get_atoms() if a.element.strip() != "H"]

    return [count_clashes(cx, heavy(cx.first), heavy(cx.second), VDW_RADII)]


extract_features = su.make_extractor(encode)
