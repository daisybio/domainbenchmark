"""Known-answer tests for the interface encoders on hand-built complexes.

Run from the repo root:
    PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_structure_encoders_synthetic.py
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bin"))

from features import charge_map, contact_map, hydrophobic_map, patch_max_size, patch_mean_contacts  # noqa: E402
from features import salt_bridge_count  # noqa: E402
from features import structure_utils as su  # noqa: E402


def _line(serial, name, resname, chain, resseq, xyz):
    name = name if len(name) == 4 else f" {name:<3s}"
    x, y, z = xyz
    return (f"ATOM  {serial:5d} {name} {resname} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00          {name.strip()[0]:>2s}")


def _res(chain, resseq, resname, cb, **extra):
    """A residue placed by its CB (by CA for GLY), backbone around the CA, plus `extra` atoms."""
    cb = np.asarray(cb, dtype=float)
    ca = cb if resname == "GLY" else cb + (0.0, 0.0, 1.5)
    atoms = {"N": ca - (1.2, 0, 0), "CA": ca, "C": ca + (1.2, 0, 0), "O": ca + (1.2, 0, 1.2)}
    if resname != "GLY":
        atoms["CB"] = cb
    atoms.update({k: np.asarray(v, dtype=float) for k, v in extra.items()})
    return chain, resseq, resname, atoms


def _complex(residues):
    lines, serial = [], 1
    for chain, resseq, resname, atoms in residues:
        for name, xyz in atoms.items():
            lines.append(_line(serial, name, resname, chain, resseq, xyz))
            serial += 1
    blob = np.frombuffer(gzip.compress(("\n".join(lines) + "\nEND\n").encode(), mtime=0), dtype=np.uint8)
    a, b = su.parse_complex(blob, "synthetic")
    return su.Complex("synthetic", a, b, lambda: su.compute_sasa(a, b))


def _chain(chain, xs, y, resname="ALA"):
    return [_res(chain, i + 1, resname, (x, y, 0.0)) for i, x in enumerate(xs)]


def _all_interface(monkeypatch):
    monkeypatch.setattr(su, "interface_residues",
                        lambda cx: (np.arange(len(cx.first)), np.arange(len(cx.second))))


def _transpose_perm():
    """perm with encode_matrix(M.T)[i] == encode_matrix(M)[perm[i]], from a random 7x11 matrix."""
    M = np.random.default_rng(0).random((7, 11))
    v, vt = su.encode_matrix(M, symmetric=False), su.encode_matrix(M.T, symmetric=False)
    perm = np.array([np.argmin(np.abs(v - x)) for x in vt])
    assert len(set(perm)) == len(v) and np.allclose(v[perm], vt)
    return perm


# ------------------------------------------------------------- encode_matrix


def test_encode_matrix_widths():
    rng = np.random.default_rng(1)
    S = rng.random((5, 5))
    S = S + S.T
    assert su.encode_matrix(S, symmetric=True).shape == (22,)
    assert su.encode_matrix(S, symmetric=True, positions=np.arange(5)).shape == (29,)
    assert su.encode_matrix(rng.random((3, 8)), symmetric=False).shape == (35,)
    with pytest.raises(ValueError, match="positions"):
        su.encode_matrix(rng.random((3, 8)), symmetric=False, positions=np.arange(3))


def test_encode_matrix_transpose_is_a_permutation():
    perm = _transpose_perm()
    M = np.random.default_rng(2).random((4, 9))
    assert np.allclose(su.encode_matrix(M, symmetric=False)[perm], su.encode_matrix(M.T, symmetric=False))


def test_bands_use_residue_numbers():
    M = np.zeros((4, 4))
    M[0, 3] = M[3, 0] = 1.0  # residue 1 <-> residue 10: |dresseq| = 9, list index distance 3
    bands = su.encode_matrix(M, symmetric=True, positions=[1, 2, 3, 10])[22:]
    assert bands[3] == 1.0  # [9, 17)
    assert bands[1] == 0.0  # [3, 5)


def test_no_length_term():
    """Two empty maps of different size encode identically (the old log1p(r) did not)."""
    assert np.array_equal(su.encode_matrix(np.zeros((3, 3)), symmetric=True),
                          su.encode_matrix(np.zeros((9, 9)), symmetric=True))


# ------------------------------------------------------------- shared helpers


def test_cb_coords_gly_and_missing_cb():
    cx = _complex(_chain("A", [0.0], 0.0) + [_res("B", 1, "GLY", (0.0, 9.0, 0.0))])
    assert np.allclose(su.cb_coords(cx.second), [[0.0, 9.0, 0.0]])
    cx.first[(" ", 1, " ")].detach_child("CB")
    with pytest.raises(ValueError, match="ALA .* has no CB atom"):
        su.cb_coords(cx.first)


def test_interface_residues_real_sasa():
    near = _complex(_chain("A", [0.0, 3.8, 30.0], 0.0) + _chain("B", [0.0, 3.8], 4.0))
    first, second = su.interface_residues(near)
    assert list(first) == [0, 1] and list(second) == [0, 1]
    swapped = su.interface_residues(near.swapped())
    assert list(swapped[0]) == [0, 1] and list(swapped[1]) == [0, 1]
    far = _complex(_chain("A", [0.0, 3.8], 0.0) + _chain("B", [0.0, 3.8], 50.0))
    assert all(len(i) == 0 for i in su.interface_residues(far))


# ------------------------------------------------------------- contact_map


@pytest.mark.parametrize("y, inter_sum", [(7.0, 16), (8.5, 0)])
def test_contact_map_inter(y, inter_sum):
    xs = [3.8 * i for i in range(6)]
    vec = contact_map.encode(_complex(_chain("A", xs, 0.0) + _chain("B", xs, y)))
    assert vec.shape == (93,)
    assert vec[58 + 1] * 6 == pytest.approx(inter_sum)  # inter rs.mean x 6 rows


def test_contact_map_gly_counts_at_ca():
    vec = contact_map.encode(_complex(_chain("A", [0.0], 0.0) + [_res("B", 1, "GLY", (0.0, 7.0, 0.0))]))
    assert vec[58 + 1] == 1.0


def test_contact_map_swap():
    xs = [3.8 * i for i in range(6)]
    cx = _complex(_chain("A", xs, 0.0) + _chain("B", xs[:4], 7.0))
    ab, ba = contact_map.encode(cx), contact_map.encode(cx.swapped())
    assert np.array_equal(ab[:29], ba[29:58]) and np.array_equal(ab[29:58], ba[:29])
    assert np.allclose(ab[58:][_transpose_perm()], ba[58:])


# ------------------------------------------------------------- patches


def _patch_complex():
    return _complex(_chain("A", [0.0, 3.8, 7.6, 30.0], 0.0) + _chain("B", [0.0, 3.8], 6.0))


@pytest.mark.parametrize("t, expected", [(5.0, [3, 2, 3]), (7.5, [3, 2, 5])])
def test_patch_max_size(monkeypatch, t, expected):
    _all_interface(monkeypatch)
    cx = _patch_complex()
    assert patch_max_size.encode(cx, threshold_distance=t) == expected
    swapped = patch_max_size.encode(cx.swapped(), threshold_distance=t)
    assert swapped == [expected[1], expected[0], expected[2]]


@pytest.mark.parametrize("t, expected", [(5.0, 0.0), (7.5, 5 / 6)])
def test_patch_mean_contacts(monkeypatch, t, expected):
    _all_interface(monkeypatch)
    cx = _patch_complex()
    assert patch_mean_contacts.encode(cx, threshold_distance=t) == [pytest.approx(expected)]
    assert patch_mean_contacts.encode(cx.swapped(), threshold_distance=t) == [pytest.approx(expected)]


def test_no_interface_is_zero():
    far = _complex(_chain("A", [0.0, 3.8], 0.0, "LYS") + _chain("B", [0.0, 3.8], 50.0, "ASP"))
    assert patch_max_size.encode(far, threshold_distance=7.5) == [0, 0, 0]
    assert patch_mean_contacts.encode(far, threshold_distance=7.5) == [0.0]
    assert np.array_equal(charge_map.encode(far), np.zeros(35))
    assert np.array_equal(hydrophobic_map.encode(far), np.zeros(35))


# ------------------------------------------------------------- salt bridges


def _lys(chain, resseq, cb, nz):
    return _res(chain, resseq, "LYS", cb, NZ=nz)


def _asp(chain, resseq, cb, od1, od2):
    return _res(chain, resseq, "ASP", cb, OD1=od1, OD2=od2)


@pytest.mark.parametrize("od1_y, expected", [(5.5, 1), (6.5, 0)])  # NZ-OD1 3.5 A / 4.5 A
def test_salt_bridge_distance(od1_y, expected):
    cx = _complex([_lys("A", 1, (0, 0, 0), (0, 2, 0)),
                   _asp("B", 1, (0, 8, 0), (0, od1_y, 0), (0, od1_y, 5))])
    assert salt_bridge_count.encode(cx) == [expected]
    assert salt_bridge_count.encode(cx.swapped()) == [expected]


def test_salt_bridge_arg_counts_once_and_ignores_intra_and_his():
    cx = _complex([
        # ARG: NE, NH1, NH2 all within 4 A of one GLU OE1 -> one residue pair
        _res("A", 1, "ARG", (0, 0, 0), NE=(0, 2, 0), NH1=(0.5, 2, 0), NH2=(-0.5, 2, 0)),
        # LYS with an intra-chain ASP 3 A from its NZ -> not counted
        _lys("A", 2, (20, 0, 0), (20, 2, 0)),
        _asp("A", 3, (20, 5, 3), (20, 5, 0), (20, 5, 3)),
        # HIS next to an ASP -> not a salt bridge
        _res("A", 4, "HIS", (40, 0, 0)),
        _res("B", 1, "GLU", (0, 8, 0), OE1=(0, 5, 0), OE2=(0, 5, 6)),
        _asp("B", 2, (40, 3, 0), (40, 1.5, 0), (40, 1.5, 3)),
    ])
    assert salt_bridge_count.encode(cx) == [1]
    assert salt_bridge_count.encode(cx.swapped()) == [1]


def test_salt_bridge_missing_atom_raises():
    cx = _complex([_res("A", 1, "LYS", (0, 0, 0)), _asp("B", 1, (0, 8, 0), (0, 5, 0), (0, 5, 3))])
    with pytest.raises(ValueError, match="has no NZ atom"):
        salt_bridge_count.encode(cx)


# ------------------------------------------------------------- charge / hydrophobic maps


def test_charge_map_known_answer(monkeypatch):
    _all_interface(monkeypatch)
    cx = _complex([_res("A", 1, "LYS", (0, 0, 0)), _res("B", 1, "ASP", (0, 9, 0))])
    vec = charge_map.encode(cx)
    assert vec.shape == (35,)
    assert vec[1] == pytest.approx(11 - abs((9.74 - 7) * (2.77 - 7) * 19 / 33.8))  # rs.mean
    assert vec[1] == pytest.approx(4.485, abs=1e-3)


def test_hydrophobic_map_known_answer(monkeypatch):
    _all_interface(monkeypatch)
    cx = _complex([_res("A", 1, "ILE", (0, 0, 0)), _res("B", 1, "ARG", (0, 9, 0))])
    assert hydrophobic_map.encode(cx)[1] == pytest.approx(20 - 9 * 19 / 10.6)


@pytest.mark.parametrize("module", [charge_map, hydrophobic_map])
def test_maps_swap_permutes(monkeypatch, module):
    _all_interface(monkeypatch)
    cx = _complex([_res("A", 1, "LYS", (0, 0, 0)), _res("A", 2, "GLU", (3.8, 0, 0)),
                   _res("A", 3, "ILE", (7.6, 0, 0)), _res("B", 1, "ASP", (0, 9, 0)),
                   _res("B", 2, "PHE", (3.8, 9, 0))])
    ab, ba = module.encode(cx), module.encode(cx.swapped())
    assert not np.allclose(ab, ba)
    assert np.allclose(ab[_transpose_perm()], ba)


@pytest.mark.parametrize("module", [charge_map, hydrophobic_map])
def test_maps_one_sided_interface_is_zero(monkeypatch, module):
    monkeypatch.setattr(su, "interface_residues", lambda cx: (np.arange(len(cx.first)), np.array([], dtype=int)))
    cx = _complex([_res("A", 1, "LYS", (0, 0, 0)), _res("B", 1, "ASP", (0, 9, 0))])
    assert np.array_equal(module.encode(cx), np.zeros(35))


def test_gly_uses_ca_even_with_cb():
    # GLY carrying a stray CB far away: CA geometry must decide the contact.
    gly = _res("B", 1, "GLY", (0.0, 7.0, 0.0), CB=(0.0, 50.0, 0.0))
    cx = _complex([_res("A", 1, "ALA", (0.0, 0.0, 0.0)), gly])
    assert np.allclose(su.cb_coords(cx.second)[0], (0.0, 7.0, 0.0))


def test_non_finite_coordinate_fails_at_parse():
    lys = _res("A", 1, "LYS", (0.0, 0.0, 0.0), NZ=(float("nan"), 2.0, 0.0))
    with pytest.raises(ValueError, match="non-finite"):
        _complex([lys, _res("B", 1, "ASP", (0.0, 3.0, 0.0))])


def test_insertion_codes_rejected_for_bands():
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    chain = Chain("A")
    chain.add(Residue((" ", 10, " "), "ALA", " "))
    chain.add(Residue((" ", 10, "A"), "ALA", " "))
    with pytest.raises(ValueError, match="insertion codes"):
        su.residue_numbers(chain)
