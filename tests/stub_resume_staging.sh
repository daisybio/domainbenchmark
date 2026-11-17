#!/usr/bin/env bash
# S-12 regression test: a model task stages only its own combo's feature files,
# so appending a feature and resuming re-runs only that feature's singletons and
# the `_all` combos. Two stub runs of the test profile; no containers needed.
# usage: tests/stub_resume_staging.sh [workdir]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-$(mktemp -d)}
mkdir -p "$dir" && cd "$dir"
nextflow run "$repo" -profile test -stub-run --outdir results -ansi-log false > run1.log
nextflow run "$repo" -profile test -stub-run --outdir results -ansi-log false -resume \
    --machine_learning_features aacomp,dummy,esm3_embeddings,aaencode > run2.log
# Model tasks submitted in the resumed run, by combo (tag ends in _<combo>).
rerun=$(grep -E ':(NEURAL_NETWORK|RANDOM_FOREST|SVM) \(' run2.log | grep -oE '\([^)]*\)' | sort -u)
bad=$(echo "$rerun" | grep -vE '_(aaencode|all)\)$' || true)
if [ -z "$rerun" ] || [ -n "$bad" ]; then
    echo "FAIL: expected only aaencode/_all model tasks to re-run; re-ran:" >&2
    echo "$rerun" >&2
    exit 1
fi
echo "OK: re-ran only $(echo "$rerun" | wc -l) aaencode/_all model tasks"
