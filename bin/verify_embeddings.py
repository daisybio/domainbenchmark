#!/usr/bin/env python3
"""Verify a published domainsplit embedding file against the split databases it will be paired with.

`domain.id` is a **surrogate integer**. domainsplit's `SUBSET_SPLIT_DB` copies it
verbatim and `PRUNE_UNREPRESENTED_DDIS` deletes without renumbering, so one
published `<model>_domain_embeddings.h5` is valid across every split database
*of the same run* and silently wrong across runs -- the ids still exist, they
just name different domains.

Nothing raises on its own when that happens. `bin/features/embeddings.py` keys on
`COALESCE(instance_id, 'r' || rowid)` and `machine_learning.load_embedding_data`
*skips* any instance pair it cannot resolve, so a drifted key layout yields zero
training rows, which is indistinguishable from a model that found no data. This
script asserts the join resolves instead of assuming it, before a single GPU hour
is spent.

Two guards, in order of strength:

1. **Structural.** For every `*.sqlite3` in the database directory, take the
   `(domain_id, instance_key)` set the benchmark will look up and count how many
   the HDF5 actually carries. A file from the same run resolves ~100%; a file
   from a foreign run resolves ~0%, because the surrogate ids collide only by
   accident. Below `--min-coverage` is a hard failure.

   Coverage is not required to be exactly 1.0: domainsplit's own exporter warns
   about instances with no embedding ("sequences dropped by --max-len or an OOM
   at batch size 1"), so a legitimate pairing can sit a little under 100%. The
   default threshold sits far below any such loss and far above the noise floor
   of a mismatched run.

2. **Declared run id.** `--expect-run` is compared against each file's
   `domainsplit_run` root attribute (domainsplit writes its `workflow.sessionId`
   there). Every file must also agree with every other file, whether or not
   `--expect-run` was given -- embeddings from two different runs in one
   `--embeddings` directory is the same hazard one level up.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import h5py

from determinism import seed_everything

# What `bin/features/embeddings.py` reads and what domainsplit's
# export_domain_embeddings.py records in the `key_layout` root attribute.
EXPECTED_KEY_LAYOUT = "{domain_id}/{instance_id}"

# The instance key the benchmark looks up, from the database's own point of view.
# Must stay identical to `embeddings.instance_key_sql()`.
INSTANCE_QUERY = (
    "SELECT domain_id, COALESCE(instance_id, 'r' || rowid) "
    "FROM domain_protein_map ORDER BY domain_id, rowid"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-dir",
        type=Path,
        required=True,
        help="Directory of split databases (train/validation/test*.sqlite3).",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="FEATURE=PATH",
        help="Feature name and the published HDF5 backing it. Repeatable.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Minimum fraction of a split's domain instances the HDF5 must "
        "resolve. Below this the pairing is rejected. Default: 0.5.",
    )
    parser.add_argument(
        "--expect-run",
        default=None,
        help="Required value of the `domainsplit_run` root attribute. When "
        "omitted the files are only checked against each other.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Unused; kept for uniformity.")
    return parser.parse_args()


def parse_pairs(raw_pairs):
    pairs = []
    for item in raw_pairs:
        if "=" not in item:
            sys.exit(f"--pair expects FEATURE=PATH, got {item!r}")
        feature, _, path = item.partition("=")
        pairs.append((feature, Path(path)))
    if not pairs:
        sys.exit("no --pair given; nothing to verify")
    return sorted(pairs)


def read_instances(db_path: Path):
    """`[(domain_id, instance_key)]` for one split database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            (str(domain_id), str(key)) for domain_id, key in conn.execute(INSTANCE_QUERY)
        ]
    finally:
        conn.close()


def coverage(h5_file: h5py.File, instances):
    """`(resolved, total, domains_resolved, domains_total, samples_missing)`.

    Looked up per instance rather than by traversing the HDF5: the published
    file is global to the run while a split database is a subset of it, so
    walking every group would read far more metadata than the question needs.
    """
    resolved = 0
    domains_seen = set()
    domains_resolved = set()
    groups = {}
    samples_missing = []

    for domain_id, instance_key in instances:
        domains_seen.add(domain_id)
        if domain_id not in groups:
            groups[domain_id] = h5_file.get(domain_id)
        group = groups[domain_id]
        if group is not None and instance_key in group:
            resolved += 1
            domains_resolved.add(domain_id)
        elif len(samples_missing) < 5:
            samples_missing.append(f"{domain_id}/{instance_key}")

    return (
        resolved,
        len(instances),
        len(domains_resolved),
        len(domains_seen),
        samples_missing,
    )


def check_attrs(pairs, expect_run):
    """Read root attributes, enforce run agreement. Returns {feature: attrs}."""
    all_attrs = {}
    runs = {}

    for feature, path in pairs:
        if not path.exists():
            sys.exit(f"[verify] {feature}: {path} does not exist")
        with h5py.File(path, "r") as h5_file:
            attrs = {k: h5_file.attrs[k] for k in h5_file.attrs}
        all_attrs[feature] = attrs

        described = ", ".join(f"{k}={attrs[k]}" for k in sorted(attrs))
        print(f"[verify] {feature}: {path.name} -- {described or 'no root attributes'}")

        if "domainsplit_run" not in attrs:
            sys.exit(
                f"[verify] {feature}: no `domainsplit_run` root attribute. This "
                "file was not written by domainsplit's EXPORT_DOMAIN_EMBEDDINGS, "
                "so there is no way to tell which run's surrogate domain ids it "
                "is keyed by. Refusing to pair it with a database."
            )
        runs[feature] = str(attrs["domainsplit_run"])

        layout = str(attrs.get("key_layout", ""))
        if layout != EXPECTED_KEY_LAYOUT:
            sys.exit(
                f"[verify] {feature}: key_layout is {layout!r}, expected "
                f"{EXPECTED_KEY_LAYOUT!r}. The benchmark reads "
                "h5[domain_id][instance_key]; a different layout would resolve "
                "nothing and look like an empty training set."
            )

    distinct = sorted(set(runs.values()))
    if len(distinct) > 1:
        listing = "\n".join(f"    {f}: {runs[f]}" for f, _ in pairs)
        sys.exit(
            "[verify] embedding files come from different domainsplit runs:\n"
            f"{listing}\n"
            "    domain.id is a surrogate integer, so mixing runs concatenates "
            "features of unrelated domains into one vector. Re-export them from "
            "a single run."
        )

    observed = distinct[0]
    if expect_run is not None and observed != expect_run:
        sys.exit(
            f"[verify] domainsplit_run mismatch: files carry {observed!r}, "
            f"--domainsplit_run declared {expect_run!r}. Either the embeddings "
            "or the databases are from the wrong run."
        )

    print(f"[verify] domainsplit_run = {observed}")
    return all_attrs


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    pairs = parse_pairs(args.pair)
    check_attrs(pairs, args.expect_run)

    split_dbs = sorted(args.db_dir.glob("*.sqlite3"))
    if not split_dbs:
        sys.exit(f"[verify] no *.sqlite3 under {args.db_dir}")

    failures = []
    for feature, path in pairs:
        with h5py.File(path, "r") as h5_file:
            for db_path in split_dbs:
                instances = read_instances(db_path)
                split = db_path.stem
                if not instances:
                    print(f"[verify] {feature} x {split}: no domain instances -- skipped")
                    continue

                resolved, total, dom_hit, dom_total, missing = coverage(h5_file, instances)
                frac = resolved / total
                print(
                    f"[verify] {feature} x {split}: {resolved}/{total} instances "
                    f"({frac:.2%}), {dom_hit}/{dom_total} domains"
                )
                if frac < args.min_coverage:
                    failures.append(
                        f"  {feature} x {split}: {frac:.2%} < {args.min_coverage:.2%} "
                        f"(e.g. missing {', '.join(missing)})"
                    )

    if failures:
        sys.exit(
            "[verify] embedding files do not match these databases:\n"
            + "\n".join(failures)
            + "\n"
            "    domain.id is a surrogate integer that domainsplit copies "
            "verbatim into every split database and never renumbers, so a file "
            "from a different run resolves almost nothing while raising no "
            "error of its own -- the loader would simply skip every pair and "
            "train on zero rows. Point --embeddings at the run that produced "
            "these databases."
        )

    print(f"[verify] OK -- {len(pairs)} feature(s) x {len(split_dbs)} split(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
