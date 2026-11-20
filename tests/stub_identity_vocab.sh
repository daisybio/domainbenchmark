#!/usr/bin/env bash
# W-15: identity's vocabulary comes from the database's train + validation
# sqlite, staged as inputs of their own (a directory input is hashed by the
# directory's own path/size/mtime, not its children). A copy of the fixture,
# a stub run, then validation.sqlite3 of the copy touched and the run resumed:
# every identity extraction must re-run, aacomp must stay cached. No containers needed.
# usage: tests/stub_identity_vocab.sh [workdir]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-$(mktemp -d)}
mkdir -p "$dir/data/databases" && cd "$dir"
cp -r "$repo/tests/data/databases/random" data/databases/
args=(-profile test -stub-run -ansi-log false --input data/databases
      --structures "$repo/tests/data/structures/structures.h5" --outdir results)
nextflow run "$repo" "${args[@]}" -with-trace one.tsv > run1.log
sleep 1
touch data/databases/random/validation.sqlite3
nextflow run "$repo" "${args[@]}" -with-trace two.tsv -resume > run2.log
python3 - <<'EOF'
import csv, sys

def rows(path, feature):
    return [r for r in csv.DictReader(open(path), delimiter="\t")
            if r["name"].split(" (")[0].endswith(":FEATURE_EXTRACTION_ONE") and f"_{feature}_" in r["name"]]

fail = []
for feature, want in (("identity", "COMPLETED"), ("aacomp", "CACHED")):
    one, two = rows("one.tsv", feature), rows("two.tsv", feature)
    if not one or len(two) != len(one):
        fail.append(f"{feature}: {len(one)} tasks in run 1, {len(two)} in run 2")
    bad = [f"{r['name']}: {r['status']}" for r in two if r["status"] != want]
    fail += [f"{feature} should be {want}: {b}" for b in bad]
if fail:
    sys.exit("FAIL:\n" + "\n".join(fail))
print(f"OK: {len(rows('two.tsv', 'identity'))} identity extractions re-ran, "
      f"{len(rows('two.tsv', 'aacomp'))} aacomp extractions cached")
EOF
