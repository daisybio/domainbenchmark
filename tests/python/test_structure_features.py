"""Structure-derived encoders (bin/features/structure_utils.py and friends).

Run from the repo root:
    PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_structure_features.py
"""
import gzip
import importlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bin"))

from features import sasa_cache  # noqa: E402

DB = REPO / "tests/data/databases/random/test_balanced.sqlite3"
DB_DIR = DB.parent
STRUCTURES = REPO / "tests/data/structures/structures.h5"
VARIANTS = json.loads((REPO / "assets/feature_variants.json").read_text())

# feature -> (width, chain layout). "sym": vec_ba == vec_ab. A list of
# (start, stop) halves: those blocks swap between orientations.
H32 = [(0, 32), (32, 64)]
H6 = [(0, 6), (6, 12)]
EXPECTED = {
    "aacomp_interface": (20, "sym"),
    "angles_phi": (12, "sym"),
    "angles_psi": (12, "sym"),
    "bsa_residue": (6, "sym"),
    "bsa_structure": (1, "sym"),
    "bsa_structure_norm": (1, "sym"),
    "charge_map": (64, H32),
    "clash_backbone_atoms": (1, "sym"),
    "clash_heavy_atoms": (1, "sym"),
    "compactness": (1, "sym"),
    "contact_map": (102, H32),
    "depth_index": (6, "sym"),
    "disulfide_bond_count": (1, "sym"),
    "hydrophobic_map": (64, H32),
    "interface_arearatio": (1, "sym"),
    "k_closest_distance_5k": (1, "sym"),
    "k_closest_distance_10k": (1, "sym"),
    "patch_max_size_5A": (1, "sym"),
    "patch_max_size_7.5A": (1, "sym"),
    "patch_max_size_10A": (1, "sym"),
    "patch_mean_contacts_5A": (1, "sym"),
    "patch_mean_contacts_7.5A": (1, "sym"),
    "patch_mean_contacts_10A": (1, "sym"),
    "polar_residue_fraction": (1, "sym"),
    "protrusion_index": (6, "sym"),
    "radius_of_gyration": (1, "sym"),
    "rasa_residue": (12, H6),
    "rasa_residue_complex": (6, "sym"),
    "salt_bridge_count": (1, "sym"),
    "sasa_residue": (12, H6),
    "sasa_residue_complex": (6, "sym"),
    "sasa_structure": (1, "sym"),
    "sasa_structure_norm": (1, "sym"),
}
# One reader per cached field (iso, cplx, depth); Shrake-Rupley on the fly
# costs ~7 s per feature here, so not every SASA encoder is rerun.
SASA_FEATURES = ["rasa_residue", "sasa_residue_complex", "depth_index"]


def spec(feature):
    entry = VARIANTS.get(feature, {"module": feature, "params": {}})
    return entry["module"], entry["params"]


def run(feature, db=DB, structures=STRUCTURES, cache=None):
    """Extract `feature` and return {(group, dataset): vector}."""
    module, params = spec(feature)
    extract = importlib.import_module(f"features.{module}").extract_features
    conn = sqlite3.connect(db)
    try:
        with h5py.File(f"{feature}.h5", "w", driver="core", backing_store=False) as f:
            extract(conn, f, 42, structures=str(structures), sasa_cache=cache and str(cache), **params)
            assert f.attrs["layout"] == "interaction"
            return {(g, d): f[g][d][()] for g in f for d in f[g]}
    finally:
        conn.close()


def membership():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        """SELECT m.ddi_id, m.instance_id_a, m.instance_id_b, m.has_structure, d1.pfam_id, d2.pfam_id
           FROM ddi_split_membership m
           JOIN domain_protein_map p1 ON p1.instance_id = m.instance_id_a JOIN domain d1 ON d1.id = p1.domain_id
           JOIN domain_protein_map p2 ON p2.instance_id = m.instance_id_b JOIN domain d2 ON d2.id = p2.domain_id"""
    ).fetchall()
    conn.close()
    return rows


@pytest.fixture(scope="session")
def cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sasa")
    dbs = [str(DB_DIR / "test_balanced.sqlite3"), str(DB_DIR / "test_realistic.sqlite3")]
    shards = [tmp / f"shard{i}.h5" for i in range(2)]
    for i, shard in enumerate(shards):
        sasa_cache.build_cache(dbs, str(STRUCTURES), str(shard), procs=2, shard_id=i, n_shards=2)
    keys = []
    for shard in shards:
        with h5py.File(shard, "r") as f:
            keys.append({(g, d) for g in f for d in f[g]})
    assert keys[0] and keys[1] and not keys[0] & keys[1], "shards must be non-empty and disjoint"
    merged = tmp / "sasa_cache.h5"
    sasa_cache.merge_shards([str(s) for s in shards], str(merged))
    with h5py.File(merged, "r") as f:
        assert {(g, d) for g in f for d in f[g]} == keys[0] | keys[1]
    return merged


def _rewrite_structures(src_path, dst_path, transform):
    """Copy structures.h5, passing each PDB's lines through `transform(lines)`."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for group in src:
            for dataset in src[group]:
                lines = gzip.decompress(src[group][dataset][()].tobytes()).decode().splitlines()
                g, d, lines = transform(group, dataset, lines)
                blob = gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)
                dst.require_group(g)[d] = np.frombuffer(blob, dtype=np.uint8)


@pytest.fixture(scope="session")
def outputs(cache):
    return {feature: run(feature, cache=cache) for feature in EXPECTED}


def test_variants_file_covers_expected():
    assert set(VARIANTS) <= set(EXPECTED)
    assert list(VARIANTS) == sorted(VARIANTS)


@pytest.mark.parametrize("feature", sorted(EXPECTED))
def test_encoder_layout(outputs, feature):
    width, layout = EXPECTED[feature]
    out = outputs[feature]
    covered = [(a, b, pa, pb) for _d, a, b, s, pa, pb in membership() if s]
    assert len(out) == 2 * len(covered)
    for inst_a, inst_b, pfam_a, pfam_b in covered:
        ab = out[(f"{pfam_a}_{pfam_b}", f"{inst_a}_{inst_b}")]
        ba = out[(f"{pfam_b}_{pfam_a}", f"{inst_b}_{inst_a}")]
        assert ab.shape == ba.shape == (width,) and ab.dtype == np.float32
        assert np.isfinite(ab).all() and np.isfinite(ba).all()
        if layout == "sym":
            assert np.array_equal(ab, ba)
        else:
            (s1, e1), (s2, e2) = layout
            assert np.array_equal(ab[s1:e1], ba[s2:e2]) and np.array_equal(ab[s2:e2], ba[s1:e1])
    if layout != "sym":  # the halves really differ somewhere, or the check proves nothing
        assert any(not np.array_equal(v[s1:e1], v[s2:e2]) for v in out.values())


def test_coverage_holes_and_homodimer(outputs):
    out = outputs["contact_map"]
    rows = membership()
    by_ddi = {}
    for ddi, a, b, s, pa, pb in rows:
        by_ddi.setdefault(ddi, []).append(((f"{pa}_{pb}", f"{a}_{b}") in out, s))
    absent = [d for d, v in by_ddi.items() if not any(present for present, _ in v)]
    partial = [d for d, v in by_ddi.items() if any(p for p, _ in v) and not all(p for p, _ in v)]
    assert absent == [8] and partial == [1]
    assert all(present == bool(s) for v in by_ddi.values() for present, s in v)
    homodimer = [(pa, a, b) for _d, a, b, s, pa, pb in rows if pa == pb and s]
    assert homodimer
    for pfam, a, b in homodimer:
        assert (f"{pfam}_{pfam}", f"{a}_{b}") in out and (f"{pfam}_{pfam}", f"{b}_{a}") in out


def test_deterministic(outputs, cache):
    for feature in ("contact_map", "angles_psi", "rasa_residue"):
        again = run(feature, cache=cache)
        assert again.keys() == outputs[feature].keys()
        assert all(again[k].tobytes() == outputs[feature][k].tobytes() for k in again)


@pytest.mark.parametrize("feature", SASA_FEATURES)
def test_cache_matches_on_the_fly(outputs, feature):
    fresh = run(feature)
    assert fresh.keys() == outputs[feature].keys()
    assert all(fresh[k].tobytes() == outputs[feature][k].tobytes() for k in fresh)


def _swap(group, dataset, lines):
    pa, pb = group.split("_")
    ia, ib = dataset.split("__")
    swap = {"A": "B", "B": "A"}
    lines = [ln[:21] + swap[ln[21]] + ln[22:] if ln.startswith("ATOM") else ln for ln in lines]
    return f"{pb}_{pa}", f"{ib}__{ia}", lines


def test_swapped_storage_orientation(outputs, tmp_path):
    """A structure stored as (b, a) with chain A = instance b reads back the same."""
    swapped = tmp_path / "structures.h5"
    _rewrite_structures(STRUCTURES, swapped, _swap)
    for feature in ("contact_map", "rasa_residue"):
        again = run(feature, structures=swapped)
        assert again.keys() == outputs[feature].keys()
        assert all(np.array_equal(again[k], outputs[feature][k]) for k in again)


def test_disulfide_fixture_has_bonds(outputs):
    assert any(v[0] > 0 for v in outputs["disulfide_bond_count"].values())


def _drop_sg(group, dataset, lines):
    return group, dataset, [ln for ln in lines if not (ln.startswith("ATOM") and ln[12:16].strip() == "SG")]


def test_missing_atom_raises(tmp_path):
    no_sg = tmp_path / "structures.h5"
    _rewrite_structures(STRUCTURES, no_sg, _drop_sg)
    with pytest.raises(ValueError, match="has no SG atom"):
        run("disulfide_bond_count", structures=no_sg)


@pytest.mark.parametrize(
    "sql, message",
    [
        ("ALTER TABLE ddi_split_membership DROP COLUMN has_structure", "has_structure"),
        ("UPDATE ddi_split_membership SET has_structure = 1 WHERE ddi_id = 8", "neither"),
        ("UPDATE ddi_split_membership SET has_structure = 0", "no membership pair with a structure"),
    ],
)
def test_inconsistent_input_raises(tmp_path, sql, message):
    db = tmp_path / "split.sqlite3"
    shutil.copy(DB, db)
    conn = sqlite3.connect(db)
    conn.execute(sql)
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match=message):
        run("bsa_structure", db=db)


def test_same_instance_self_pair_averages_chain_orders(monkeypatch):
    """An instance paired with itself writes one key: the mean of both chain orders.

    Two AF3 copies of one chain differ, so a chain-ordered feature's two orders
    differ too; writing both used to abort the task.
    """
    import features.structure_utils as su

    conn = sqlite3.connect(DB)
    real = next(su.iter_structures(conn, str(STRUCTURES)))
    _keys, cx = real
    encoder = importlib.import_module("features.sasa_residue")
    module_encode = encoder.encode
    ab = np.asarray(module_encode(cx), dtype=np.float64)
    ba = np.asarray(module_encode(cx.swapped()), dtype=np.float64)
    assert not np.array_equal(ab, ba), "fixture complex must be asymmetric for this test"

    monkeypatch.setattr(
        su, "iter_structures",
        lambda *a, **k: iter([(("PF1", "PF1", "I1", "I1"), cx)]),
    )
    with h5py.File("self_pair.h5", "w", driver="core", backing_store=False) as out:
        su.make_extractor(module_encode)(conn, out, 0, structures=str(STRUCTURES))
        assert list(out) == ["PF1_PF1"] and list(out["PF1_PF1"]) == ["I1_I1"]
        np.testing.assert_allclose(out["PF1_PF1"]["I1_I1"][()], ((ab + ba) / 2).astype(np.float32))
    conn.close()
