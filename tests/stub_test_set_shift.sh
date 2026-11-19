#!/usr/bin/env bash
# W-10 wiring: the fixture's `random` copied with test_balanced/test_realistic
# renamed test_uniform_1_1/test_uniform_1_10 (the fixture itself stays as it
# is), a stub run with --seeds 42, then the same resumed with --seeds 42,43.
# Only the 1:10 EVAL_ONE tasks may stage reference/predictions_uniform_1_1.parquet,
# EVAL_ONE must double, and every seed-42 EVAL_ONE must be reused. No containers needed.
# usage: tests/stub_test_set_shift.sh [workdir]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-$(mktemp -d)}
mkdir -p "$dir/data/databases/random" && cd "$dir"
src="$repo/tests/data/databases/random"
cp "$src/train.sqlite3" "$src/validation.sqlite3" data/databases/random/
cp "$src/test_balanced.sqlite3" data/databases/random/test_uniform_1_1.sqlite3
cp "$src/test_realistic.sqlite3" data/databases/random/test_uniform_1_10.sqlite3
# DDI_EXTRACTION (no stub) checks the membership split against the file name.
python3 - <<'EOF'
import sqlite3
for split in ("test_uniform_1_1", "test_uniform_1_10"):
    with sqlite3.connect(f"data/databases/random/{split}.sqlite3") as con:
        con.execute("UPDATE ddi_split_membership SET split = ?", (split,))
EOF
args=(-profile test -stub-run -ansi-log false --input data/databases
      --structures "$repo/tests/data/structures/structures.h5")
nextflow run "$repo" "${args[@]}" --outdir one -with-trace one.tsv > run1.log
nextflow run "$repo" "${args[@]}" --outdir two -with-trace two.tsv --seeds 42,43 -resume > run2.log
python3 - <<'EOF'
import csv, glob, os, pathlib, sys

def eval_one(path):
    return [r for r in csv.DictReader(open(path), delimiter="\t")
            if r["name"].split(" (")[0].endswith(":EVAL_ONE")]

def workdir(r):
    hits = glob.glob(f"work/{r['hash']}*")
    return pathlib.Path(hits[0]) if len(hits) == 1 else None

one, two = eval_one("one.tsv"), eval_one("two.tsv")
fail = []
if not one:
    fail.append("no EVAL_ONE task in the --seeds 42 run")
if len(two) != 2 * len(one):
    fail.append(f"EVAL_ONE {len(one)} -> {len(two)}, want x2")
n_target = 0
for r in one + two:
    wd = workdir(r)
    if wd is None:
        fail.append(f"no work dir for {r['name']} {r['hash']}")
        continue
    staged = sorted(p.name for p in (wd / "reference").glob("*")) if (wd / "reference").is_dir() else []
    target = "_uniform_1_10_" in r["name"]
    n_target += target
    if staged != (["predictions_uniform_1_1.parquet"] if target else []):
        fail.append(f"{r['name']}: reference/ = {staged}")
    elif target:
        # The staged link must resolve to the 1:1 parquet written by the very
        # task whose 1:10 parquet this EVAL_ONE scores (same model, combo, seed).
        ref = pathlib.Path(os.path.realpath(wd / "reference" / staged[0]))
        tgt = [pathlib.Path(os.path.realpath(p)) for p in wd.glob("predictions_*.parquet")]
        if len(tgt) != 1 or tgt[0].name != "predictions_uniform_1_10.parquet" or ref.parent != tgt[0].parent:
            fail.append(f"{r['name']}: reference -> {ref}, target -> {tgt}")
if not n_target:
    fail.append("no 1:10 EVAL_ONE task")
status = {r["hash"]: r["status"] for r in two}
fail += [f"not reused: {r['name']} {r['hash']} -> {status.get(r['hash'], 'absent')}"
         for r in one if status.get(r["hash"]) != "CACHED"]
if fail:
    sys.exit("FAIL:\n" + "\n".join(fail))
print(f"OK: EVAL_ONE {len(one)} -> {len(two)}, {n_target} target task rows stage the 1:1 "
      f"reference, {len(one)} seed-42 tasks reused by hash")
EOF
