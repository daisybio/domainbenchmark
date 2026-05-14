#!/usr/bin/env python3
"""Create small test SQLite databases from real CoBiNet data.

Run this on the server where the real data lives:

    python create_test_data.py --src /path/to/real/split_dir --out /tmp/cobinet_test

Then copy /tmp/cobinet_test/ to your local machine.

Tables copied (in dependency order):
    domain, protein, domain_domain_interaction, domain_protein_map,
    protein_protein_interaction, and any protein-GO table found.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
import pickle
import numpy as np

ZERO_BLOB_COLS = {
    "protein": {"prott5_per_residue", "esm3_per_residue", "esmc_per_residue"},
    "domain_protein_map": {"esm3_per_domain", "esmc_per_domain"},
}
TRUNCATE_RESIDUES = 15  # keep only first N residues of per-residue embeddings

N_DDI_TRAIN = 200   # 25 pos + 25 neg — enough proteins for PPI connectivity
N_DDI_TEST = 100
N_DDI_OPT = 100

N_PPI_LIMIT = 150   # max PPI rows per split
N_GO_LIMIT = 150    # max GO rows per split
N_PROTEIN_CAP = 150 # max proteins used for PPI/GO queries



def get_tables(conn):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}


def get_cols(conn, table):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def placeholders(n):
    return ",".join("?" * n)


def zero_and_truncate_blob(blob: bytes) -> bytes:
    arr = pickle.loads(blob)
    if arr.ndim == 2:
        arr = arr[:TRUNCATE_RESIDUES]  # (seq_len, dim) → (10, dim)
    return pickle.dumps(np.zeros(arr.shape, dtype=arr.dtype))

def zero_embedding_cols(rows: list, cols: list, zero_col_names: set) -> list:
    zero_indices = [i for i, c in enumerate(cols) if c in zero_col_names]
    if not zero_indices:
        return rows
    result = []
    for row in rows:
        row = list(row)
        for i in zero_indices:
            if row[i] is not None:
                row[i] = zero_and_truncate_blob(row[i])
        result.append(tuple(row))
    return result

def copy_schema(src, dst):
    rows = src.execute(
        "SELECT sql FROM sqlite_master WHERE type IN ('table') AND sql IS NOT NULL"
    ).fetchall()
    for (sql,) in rows:
        try:
            dst.execute(sql)
        except Exception as e:
            print(f"  schema warning: {e}")


def subset_db(src_path: str, dst_path: str, n_ddi: int):
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)

    copy_schema(src, dst)
    tables = get_tables(src)

    # ── 1. Sample DDI pairs ──────────────────────────────────────────────────
    n_half = n_ddi // 2
    ddi_cols = get_cols(src, "domain_domain_interaction")

    # prefer evaluation-relevant rows; fall back if column absent or too few
    def sample_ddi(where_extra="", n=n_half):
        base = "SELECT * FROM domain_domain_interaction"
        clauses = [c for c in [where_extra] if c]
        q = base + (" WHERE " + " AND ".join(clauses) if clauses else "") + f" LIMIT {n}"
        return src.execute(q).fetchall()

    has_eval = "is_evaluation_relevant" in ddi_cols

    pos = sample_ddi(("NOT negative AND is_evaluation_relevant" if has_eval else "NOT negative"), n_half)
    neg = sample_ddi(("negative AND is_evaluation_relevant" if has_eval else "negative"), n_half)

    # top-up if one class is scarce
    if len(pos) < n_half:
        pos = sample_ddi("NOT negative", n_half)
    if len(neg) < n_half:
        neg = sample_ddi("negative", n_half)

    ddis = pos + neg
    if not ddis:
        print(f"  WARNING: no DDI rows found in {src_path}, skipping")
        src.close(); dst.close()
        return

    dst.executemany(
        f"INSERT OR IGNORE INTO domain_domain_interaction VALUES ({placeholders(len(ddi_cols))})",
        ddis,
    )

    a_idx = ddi_cols.index("domain_id_a")
    b_idx = ddi_cols.index("domain_id_b")
    domain_ids = list({row[a_idx] for row in ddis} | {row[b_idx] for row in ddis})

    print(f"  DDI: {len(ddis)} pairs  ({len(pos)} pos / {len(neg)} neg)")

    # ── 2. domain table ──────────────────────────────────────────────────────
    if "domain" in tables:
        dcols = get_cols(src, "domain")
        drows = src.execute(
            f"SELECT * FROM domain WHERE id IN ({placeholders(len(domain_ids))})",
            domain_ids,
        ).fetchall()
        dst.executemany(f"INSERT OR IGNORE INTO domain VALUES ({placeholders(len(dcols))})", drows)
        print(f"  domain: {len(drows)} rows")

    # ── 3. domain_protein_map ────────────────────────────────────────────────
    protein_ids = []
    if "domain_protein_map" in tables:
        dpm_cols = get_cols(src, "domain_protein_map")
        dpm_rows = src.execute(
            f"SELECT * FROM domain_protein_map WHERE domain_id IN ({placeholders(len(domain_ids))}) LIMIT {N_PROTEIN_CAP}",
            domain_ids,
        ).fetchall()
        dpm_rows = zero_embedding_cols(dpm_rows, dpm_cols, ZERO_BLOB_COLS["domain_protein_map"])
        dst.executemany(
            f"INSERT OR IGNORE INTO domain_protein_map VALUES ({placeholders(len(dpm_cols))})",
            dpm_rows,
        )
        pid_idx = dpm_cols.index("protein_id")
        protein_ids = list({row[pid_idx] for row in dpm_rows})
        print(f"  domain_protein_map: {len(dpm_rows)} rows  ({len(protein_ids)} unique proteins)")

    # ── 4. protein table ─────────────────────────────────────────────────────
    if "protein" in tables and protein_ids:
        pcols = get_cols(src, "protein")
        prows = src.execute(
            f"SELECT * FROM protein WHERE id IN ({placeholders(len(protein_ids))})",
            protein_ids,
        ).fetchall()
        prows = zero_embedding_cols(prows, pcols, ZERO_BLOB_COLS["protein"])
        dst.executemany(f"INSERT OR IGNORE INTO protein VALUES ({placeholders(len(pcols))})", prows)
        print(f"  protein: {len(prows)} rows")

    # ── 5. protein_protein_interaction ───────────────────────────────────────
    if "protein_protein_interaction" in tables and protein_ids:
        ppicols = get_cols(src, "protein_protein_interaction")
        ppi_rows = src.execute(
            f"""SELECT * FROM protein_protein_interaction
                WHERE protein_id_a IN ({placeholders(len(protein_ids))})
                  AND protein_id_b IN ({placeholders(len(protein_ids))})
                LIMIT {N_PPI_LIMIT}""",
            protein_ids + protein_ids,
        ).fetchall()
        dst.executemany(
            f"INSERT OR IGNORE INTO protein_protein_interaction VALUES ({placeholders(len(ppicols))})",
            ppi_rows,
        )
        print(f"  protein_protein_interaction: {len(ppi_rows)} rows")

    # ── 6. protein-GO table (name varies) ────────────────────────────────────
    pgo_candidates = [t for t in tables if "go" in t.lower() and "protein" in t.lower()]
    for pgo_table in pgo_candidates:
        pgcols = get_cols(src, pgo_table)
        # find the protein FK column
        prot_col = next(
            (c for c in pgcols if c in ("protein_id", "protein_id_a")),
            next((c for c in pgcols if "protein" in c.lower()), None),
        )
        if prot_col is None or not protein_ids:
            continue
        pg_rows = src.execute(
            f"SELECT * FROM {pgo_table} WHERE {prot_col} IN ({placeholders(len(protein_ids))}) LIMIT {N_GO_LIMIT}",
            protein_ids,
        ).fetchall()
        dst.executemany(
            f"INSERT OR IGNORE INTO {pgo_table} VALUES ({placeholders(len(pgcols))})",
            pg_rows,
        )
        print(f"  {pgo_table}: {len(pg_rows)} rows")

    dst.commit()
    dst.execute("VACUUM")
    src.close()
    dst.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source directory with train/test/optimization.sqlite3")
    parser.add_argument("--out", required=True, help="Output directory for small test files")
    parser.add_argument("--n-ddi", type=int, default=None, help="Override DDI count for all splits")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": args.n_ddi or N_DDI_TRAIN,
        "test": args.n_ddi or N_DDI_TEST,
        "optimization": args.n_ddi or N_DDI_OPT,
    }

    any_found = False
    for split, n in splits.items():
        src_file = src / f"{split}.sqlite3"
        if not src_file.exists():
            print(f"skip {split}: {src_file} not found")
            continue
        any_found = True
        dst_file = out / f"{split}.sqlite3"
        print(f"\n[{split}]  {src_file}  →  {dst_file}  (n_ddi={n})")
        subset_db(str(src_file), str(dst_file), n)

    if not any_found:
        print("ERROR: no *.sqlite3 files found in --src directory", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Copy {out}/ to your local machine.")


if __name__ == "__main__":
    main()
