#!/usr/bin/env bash
# W-6 seeds: a stub run with --seeds 42, then the same resumed with --seeds 42,43.
# Every seed-dependent task (NN/RF/SVM, DEGREE_BASELINE, EVAL_ONE) must run
# twice in total, every other task once (no encoder is seeded any more, so
# identity extraction runs once and stays flat in data/), the outputs must move
# under seed<N>/, and resuming must re-run no seed-42 model, baseline or EVAL_ONE
# task. No containers needed.
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

MODELS = {"NEURAL_NETWORK": "nn_output", "RANDOM_FOREST": "rf_output", "SVM": "svm_output",
          "DEGREE_BASELINE": "baseline_output"}
DOUBLED = set(MODELS) | {"EVAL_ONE"}

def process(r):
    p = r["name"].split(" (")[0].split(":")[-1]
    return f"{p}[identity]" if p == "FEATURE_EXTRACTION_ONE" and "_identity_" in r["name"] else p

def listing(*parts):
    d = pathlib.Path(*parts)
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else None

one_rows = list(csv.DictReader(open("one.tsv"), delimiter="\t"))
two_rows = list(csv.DictReader(open("two.tsv"), delimiter="\t"))
one = collections.Counter(map(process, one_rows))
two = collections.Counter(map(process, two_rows))
fail = [f"{p}: no task in the --seeds 42 run" for p in sorted(DOUBLED | {"FEATURE_EXTRACTION_ONE[identity]"}) if not one[p]]
fail += [f"{p}: {one[p]} -> {two[p]}" for p in sorted(set(one) | set(two))
         if two[p] != one[p] * (2 if p in DOUBLED else 1)]

# Every seed-42 model, baseline and EVAL_ONE task of run 1 must be reused, matched by
# hash: a cached task keeps its run-1 name (no _seed42 tag), so a name match
# would miss a rerun. EVAL_ONE stages its model's model_parameters.json, which
# must not change its hash while the model task stays cached.
status = {r["hash"]: r["status"] for r in two_rows}
reused = [r for r in one_rows if process(r) in set(MODELS) | {"EVAL_ONE"}]
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
    # identity is seed-free: flat in data/ in both runs, no data/seed<N>/.
    for run in ("one", "two"):
        if not list(pathlib.Path(run, db, "data").glob("identity__*.h5")):
            fail.append(f"{run}/{db}/data has no identity__*.h5")
        seeded = sorted(p.name for p in pathlib.Path(run, db, "data").glob("seed*"))
        if seeded:
            fail.append(f"{run}/{db}/data has seed dirs {seeded}")

if fail:
    sys.exit("FAIL:\n" + "\n".join(fail))
print("OK:", ", ".join(f"{p} {one[p]}->{two[p]}" for p in sorted(one)),
      f"| {len(reused)} seed-42 model/baseline/EVAL_ONE tasks reused by hash | dbs: {', '.join(dbs)}")
EOF
