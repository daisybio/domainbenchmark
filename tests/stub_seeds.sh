#!/usr/bin/env bash
# W-6 seeds: a stub run with --seeds 42, then the same resumed with --seeds 42,43.
# Every seed-dependent task (NN/RF/SVM, EVAL_ONE, `dummy` extraction) must run
# twice in total, every other task once, the outputs must move under seed<N>/,
# and resuming must re-run no seed-42 model task. No containers needed.
# usage: tests/stub_seeds.sh [workdir]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-$(mktemp -d)}
mkdir -p "$dir" && cd "$dir"
nextflow run "$repo" -profile test -stub-run --outdir one -ansi-log false -with-trace one.tsv > run1.log
nextflow run "$repo" -profile test -stub-run --outdir two -ansi-log false -with-trace two.tsv \
    --seeds 42,43 -resume > run2.log
python3 - <<'EOF'
import csv, collections, pathlib, sys

MODELS = {"NEURAL_NETWORK": "nn_output", "RANDOM_FOREST": "rf_output", "SVM": "svm_output"}
DOUBLED = set(MODELS) | {"EVAL_ONE", "FEATURE_EXTRACTION_ONE[dummy]"}

def process(r):
    p = r["name"].split(" (")[0].split(":")[-1]
    return f"{p}[dummy]" if p == "FEATURE_EXTRACTION_ONE" and "_dummy_" in r["name"] else p

def listing(*parts):
    d = pathlib.Path(*parts)
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else None

one_rows = list(csv.DictReader(open("one.tsv"), delimiter="\t"))
two_rows = list(csv.DictReader(open("two.tsv"), delimiter="\t"))
one = collections.Counter(map(process, one_rows))
two = collections.Counter(map(process, two_rows))
fail = [f"{p}: no task in the --seeds 42 run" for p in sorted(DOUBLED) if not one[p]]
fail += [f"{p}: {one[p]} -> {two[p]}" for p in sorted(set(one) | set(two))
         if two[p] != one[p] * (2 if p in DOUBLED else 1)]

# Every seed-42 model task of run 1 must be reused, matched by hash: a cached
# task keeps its run-1 name (no _seed42 tag), so a name match would miss a rerun.
status = {r["hash"]: r["status"] for r in two_rows}
reused = [r for r in one_rows if process(r) in MODELS]
fail += [f"not reused: {r['name']} {r['hash']} -> {status.get(r['hash'], 'absent')}"
         for r in reused if status.get(r["hash"]) != "CACHED"]

dbs = sorted(p.parent.name for p in pathlib.Path("one").glob("*/nn_output"))
if not dbs:
    fail.append("no <db>/nn_output in the --seeds 42 run")
for db in dbs:
    for sub in MODELS.values():
        flat = listing("one", db, sub)
        if not flat:
            fail.append(f"one/{db}/{sub}: {flat}")
        if listing("two", db, sub) != ["seed42", "seed43"]:
            fail.append(f"two/{db}/{sub}: {listing('two', db, sub)}, want [seed42, seed43]")
        for seed in ("seed42", "seed43"):
            if listing("two", db, sub, seed) != flat:
                fail.append(f"two/{db}/{sub}/{seed}: {listing('two', db, sub, seed)} != {flat}")
    dummies = sorted(p.name for p in pathlib.Path("one", db, "data").glob("dummy__*.h5"))
    if not dummies:
        fail.append(f"one/{db}/data has no dummy__*.h5")
    for seed in ("seed42", "seed43"):
        if listing("two", db, "data", seed) != dummies:
            fail.append(f"two/{db}/data/{seed}: {listing('two', db, 'data', seed)} != {dummies}")
    if list(pathlib.Path("two", db, "data").glob("dummy__*.h5")):
        fail.append(f"two/{db}/data holds a flat dummy file")

if fail:
    sys.exit("FAIL:\n" + "\n".join(fail))
print("OK:", ", ".join(f"{p} {one[p]}->{two[p]}" for p in sorted(one)),
      f"| {len(reused)} seed-42 model tasks reused by hash | dbs: {', '.join(dbs)}")
EOF
