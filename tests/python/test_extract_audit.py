"""Audit S-2, S-6, S-10, S-14: input-boundary guards in DDI_EXTRACTION, PREFLIGHT,
the sequence encoders and METADATA.

Run from the repo root:  PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_extract_audit.py
"""
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("statsmodels")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bin"))

import create_metadata  # noqa: E402
import preflight  # noqa: E402
from features import aacomp  # noqa: E402
from test_enrichment import _instances  # noqa: E402
from test_preflight import DB_DIR, METADATA, db_copy, no_toolchain, run  # noqa: E402,F401

SPLITS = ["test_balanced", "test_realistic", "train", "validation"]
# Flip the label of one membership DDI's reverse orientation, inserted as a new row.
CONFLICT = ("INSERT INTO domain_domain_interaction (domain_id_a, domain_id_b, negative) "
            "SELECT domain_id_b, domain_id_a, 1 - negative FROM domain_domain_interaction "
            "WHERE domain_id_a != domain_id_b ORDER BY id LIMIT 1")


def ddi_extraction(tmp_path, db_dir):
    """Run DDI_EXTRACTION's embedded Python, as Nextflow would render it."""
    nf = (REPO / "modules/local/ddi_extraction/main.nf").read_text()
    body = re.search(r"python3 <<'PYEOF'\n(.*?)\n\s*PYEOF", nf, re.S).group(1)
    body = (textwrap.dedent(body).replace("${splits}", " ".join(SPLITS))
            .replace("${database_dir}", str(db_dir)).replace("\\\\", "\\"))
    return subprocess.run([sys.executable, "-c", body], cwd=tmp_path, capture_output=True, text=True)


def test_ddi_extraction_fixture_passes(tmp_path):
    r = ddi_extraction(tmp_path, DB_DIR)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "ddi_out/DDI/train_instances.csv").is_file()


# --- S-2: one binary label per canonical pair --------------------------------

def test_conflicting_orientations_fail_in_ddi_extraction(tmp_path):
    r = ddi_extraction(tmp_path, db_copy(tmp_path, "train", CONFLICT))
    assert r.returncode != 0 and "carry both labels across their two orientations" in r.stderr


def test_conflicting_orientations_fail_in_preflight(tmp_path):
    rc, report = run(tmp_path, db_dir=db_copy(tmp_path, "train", CONFLICT))
    assert rc == 1 and "FAILED train labels" in report and "both labels" in report


def test_agreeing_orientations_pass(tmp_path):
    agree = CONFLICT.replace("1 - negative", "negative")
    assert preflight.ddi_labels(db_copy(tmp_path, "train", agree) / "train.sqlite3")


@pytest.mark.parametrize("value", ["2", "NULL", "'True'"])
def test_non_binary_negative_is_a_clean_preflight_error(tmp_path, value):
    sql = f"UPDATE domain_domain_interaction SET negative = {value} WHERE id = (SELECT MIN(id) FROM domain_domain_interaction)"
    rc, report = run(tmp_path, db_dir=db_copy(tmp_path, "train", sql))
    assert rc == 1 and "FAILED train labels" in report and "not integer 0/1" in report


# --- S-6: membership split == file stem --------------------------------------

def test_copied_split_fails_in_ddi_extraction(tmp_path):
    db_dir = tmp_path / "db"
    shutil.copytree(DB_DIR, db_dir)
    shutil.copy(db_dir / "train.sqlite3", db_dir / "validation.sqlite3")
    r = ddi_extraction(tmp_path, db_dir)
    assert r.returncode != 0
    assert "validation.sqlite3: ddi_split_membership says split 'train', but the file is 'validation'" in r.stderr


# --- S-10: sequence queries read membership instances only -------------------

def test_bad_unused_sequence_is_ignored(tmp_path):
    db_dir = db_copy(tmp_path)
    db = db_dir / "train.sqlite3"
    with sqlite3.connect(db) as conn:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(domain_protein_map)")]
        row = dict(zip(cols, conn.execute("SELECT * FROM domain_protein_map LIMIT 1").fetchone()))
        row.update(instance_id="unused_instance", domain_sequence="", start_pos=-1, end_pos=-1)
        row.pop("id", None)
        conn.execute(f"INSERT INTO domain_protein_map ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                     list(row.values()))
    assert preflight.check_sequences(db, "aacomp") == []
    with sqlite3.connect(db) as conn:
        keys = {k for _, k, _ in conn.execute(aacomp.SEQUENCE_QUERY)}
    assert "unused_instance" not in keys and keys


def test_sequence_query_is_key_ordered(tmp_path):
    """Row order (hence HDF5 write order) must not follow the query plan."""
    with sqlite3.connect(db_copy(tmp_path) / "train.sqlite3") as conn:
        keys = [r[:2] for r in conn.execute(aacomp.SEQUENCE_QUERY)]
    assert keys and keys == sorted(keys)
    assert re.search(r"ORDER BY domain_key, instance_key", aacomp.SEQUENCE_QUERY)


# --- S-14: metadata keys must hit -------------------------------------------

def _tables():
    return pd.read_csv(METADATA / "domain_metadata.csv"), pd.read_csv(METADATA / "ddi_metadata.csv")


def test_foreign_instance_ids_fail_in_preflight(tmp_path):
    metadata = tmp_path / "metadata"
    shutil.copytree(METADATA, metadata)
    for name in ("domain_metadata.csv", "ddi_metadata.csv"):
        df = pd.read_csv(metadata / name)
        for c in [c for c in df.columns if c.startswith("instance_id")]:
            df[c] = "X" + df[c].astype(str)
        df.to_csv(metadata / name, index=False)
    rc, report = run(tmp_path, metadata=metadata)
    assert rc == 1 and "FAILED test_balanced metadata" in report and "no value for any DDI" in report


def test_one_empty_column_fails_metadata():
    dom, ddi = _tables()
    col = next(c for c in dom.columns if c not in create_metadata.DOMAIN_ID_COLS)
    dom[col] = float("nan")
    with pytest.raises(SystemExit, match=f"{col}_max"):
        create_metadata.build_table(dom, ddi, _instances())


def test_ddi_metadata_pfam_must_match_instance_pair():
    dom, ddi = _tables()
    inst = _instances()
    members = {tuple(sorted(p)) for p in zip(inst.instance_1, inst.instance_2)}
    row = next(i for i, p in enumerate(zip(ddi.instance_id_a, ddi.instance_id_b)) if tuple(sorted(p)) in members)
    ddi.loc[row, "pfam_id_a"] = "PF99999"
    with pytest.raises(SystemExit, match="not their instance pair's domains"):
        create_metadata.build_table(dom, ddi, _instances())


def test_ddi_metadata_pfam_orientation_is_free():
    dom, ddi = _tables()
    swapped = ddi.rename(columns={"pfam_id_a": "pfam_id_b", "pfam_id_b": "pfam_id_a"})
    create_metadata.build_table(dom, swapped, _instances())
