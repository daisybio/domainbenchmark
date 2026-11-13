"""bin/preflight.py: what would fail after training must fail here first.

Run from the repo root:  PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_preflight.py
"""
import shutil
import sqlite3
import sys
from pathlib import Path

import h5py
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bin"))

import create_metadata  # noqa: E402
import preflight  # noqa: E402
import verify_embeddings  # noqa: E402
from test_enrichment import _instances  # noqa: E402
from test_structure_features import _drop_sg, _rewrite_structures  # noqa: E402

DATA = REPO / "tests" / "data"
DB_DIR = DATA / "databases" / "random"
ESM3 = DATA / "embeddings" / "esm3_domain_embeddings.h5"
STRUCTURES = DATA / "structures" / "structures.h5"
METADATA = DATA / "metadata"
# The test profile's features and combos, as PER_DB_BENCHMARK builds them.
FEATURES = ["aacomp", "dummy", "esm3_embeddings", "sasa_residue", "k_closest_distance_5k"]
ALL = [f for f in FEATURES if f != "dummy"]


@pytest.fixture(autouse=True)
def no_toolchain(monkeypatch):
    # The toolchain check targets the general image; the test env has no multiqc.
    monkeypatch.setattr(preflight, "check_toolchain", lambda: None)


def test_toolchain_needs_multiqc(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="multiqc is not on PATH"):
        preflight.check_toolchain()


def run(tmp_path, db_dir=DB_DIR, esm3=ESM3, structures=STRUCTURES, metadata=METADATA, extra=()):
    report = tmp_path / "preflight.txt"
    argv = [
        "--db-dir", str(db_dir), "--report", str(report),
        *[a for f in FEATURES for a in ("--combo", f)], "--combo", ",".join(ALL),
        "--published", f"esm3_embeddings={esm3}",
        "--extracted", "aacomp=aacomp", "--extracted", "dummy=dummy",
        "--structure", "sasa_residue=sasa_residue",
        "--structure", "k_closest_distance_5k=k_closest_distance",
        "--structures", str(structures), "--metadata", str(metadata), *extra,
    ]
    return preflight.main(argv), report.read_text()


def db_copy(tmp_path, split=None, sql=None):
    db_dir = tmp_path / "db"
    shutil.copytree(DB_DIR, db_dir)
    if sql:
        with sqlite3.connect(db_dir / f"{split}.sqlite3") as conn:
            conn.execute(sql)
    return db_dir


def test_fixture_passes(tmp_path):
    rc, report = run(tmp_path)
    assert rc == 0, report
    assert "OK: 4 split(s), 6 combo(s)" in report
    assert "test_balanced: metadata row for all 151 DDIs" in report
    assert "WARNING" not in report


def test_instances_frame_matches_ddi_extraction():
    """The metadata check builds the same table METADATA builds from the CSV."""
    labels = preflight.ddi_labels(DB_DIR / "test_balanced.sqlite3")
    pairs = verify_embeddings.read_pairs(DB_DIR / "test_balanced.sqlite3")
    dom, ddi = pd.read_csv(METADATA / "domain_metadata.csv"), pd.read_csv(METADATA / "ddi_metadata.csv")
    ours = create_metadata.build_table(dom, ddi, preflight.instances_frame(pairs, labels))
    theirs = create_metadata.build_table(dom, ddi, _instances())
    pd.testing.assert_frame_equal(ours[0], theirs[0])
    assert ours[1] == theirs[1]


def _extra_chain(group, dataset, lines):
    atoms = [i for i, ln in enumerate(lines) if ln.startswith("ATOM")]
    lines[atoms[-1]] = lines[atoms[-1]][:21] + "C" + lines[atoms[-1]][22:]
    return group, dataset, lines


def _nonstandard(group, dataset, lines):
    first = next(ln for ln in lines if ln.startswith("ATOM"))
    return group, dataset, [ln[:17] + "MSE" + ln[20:] if ln.startswith("ATOM") and ln[21:27] == first[21:27] else ln
                            for ln in lines]


def _drop_backbone_o(group, dataset, lines):
    return group, dataset, [ln for ln in lines if not (ln.startswith("ATOM") and ln[12:16].strip() == "O")]


@pytest.mark.parametrize("transform, message", [
    (_extra_chain, "expected exactly chains A and B"),
    (_nonstandard, "MSE is not a standard amino acid"),
    (_drop_backbone_o, "lacks backbone atom"),
])
def test_malformed_complex(tmp_path, transform, message):
    broken = tmp_path / "structures.h5"
    _rewrite_structures(STRUCTURES, broken, transform)
    rc, report = run(tmp_path, structures=broken)
    assert rc == 1 and message in report


def test_cys_without_sg_only_matters_with_disulfide(tmp_path):
    no_sg = tmp_path / "structures.h5"
    _rewrite_structures(STRUCTURES, no_sg, _drop_sg)
    assert run(tmp_path, structures=no_sg)[0] == 0
    rc, report = run(tmp_path, structures=no_sg, extra=("--structure", "disulfide_bond_count=disulfide_bond_count"))
    assert rc == 1 and "has no SG atom" in report


def test_non_numeric_ppi_score(tmp_path):
    db_dir = db_copy(tmp_path, "train", "UPDATE protein_protein_interaction SET score = 'high' "
                                        "WHERE rowid = (SELECT MIN(rowid) FROM protein_protein_interaction)")
    rc, report = run(tmp_path, db_dir=db_dir)
    assert rc == 1 and "FAILED train ppi" in report


def test_metadata_without_id_column(tmp_path):
    # A test DDI without metadata values still gets a (NaN) row -- build_table
    # reindexes over the split's DDIs -- so what really stops METADATA is the files.
    metadata = tmp_path / "metadata"
    shutil.copytree(METADATA, metadata)
    dom = pd.read_csv(metadata / "domain_metadata.csv")
    dom.drop(columns="instance_id").to_csv(metadata / "domain_metadata.csv", index=False)
    rc, report = run(tmp_path, metadata=metadata)
    assert rc == 1 and "test_balanced metadata" in report and "instance_id" in report


def test_combo_resolving_nothing_on_a_test_split(tmp_path):
    test_keys = {(p, i) for _d, pa, ia, pb, ib in verify_embeddings.read_pairs(DB_DIR / "test_balanced.sqlite3")
                 for p, i in ((pa, ia), (pb, ib))}
    thin = tmp_path / "esm3.h5"
    with h5py.File(ESM3, "r") as src, h5py.File(thin, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for g in src:
            for d in src[g]:
                if (g, d) not in test_keys:
                    dst.require_group(g)[d] = src[g][d][()]
    rc, report = run(tmp_path, esm3=thin)
    assert rc == 1
    assert "test_balanced [esm3_embeddings]: no membership instance pair resolves" in report


SINGLE_CLASS = ("UPDATE domain_domain_interaction SET negative = 1 "
                "WHERE id IN (SELECT ddi_id FROM ddi_split_membership)")


def test_single_class_train_fails(tmp_path):
    rc, report = run(tmp_path, db_dir=db_copy(tmp_path, "train", SINGLE_CLASS))
    assert rc == 1 and "FAILED train [aacomp]: one class only (0 positive" in report


def test_single_class_test_only_warns(tmp_path):
    rc, report = run(tmp_path, db_dir=db_copy(tmp_path, "test_realistic", SINGLE_CLASS))
    assert rc == 0, report
    assert "WARNING test_realistic [aacomp]: one class only" in report
