#!/usr/bin/env python3
"""Verify a published domainsplit embedding file against the split databases it will be paired with.

The published files are keyed `h5[pfam_id][instance_key]`, and this script asserts
that the join actually resolves before a single GPU hour is spent on it.

For every `*.sqlite3` in the database directory, take the
`(pfam_id, instance_key)` set the benchmark will look up and count how many the
HDF5 carries. Below `--min-coverage` is a hard failure. This catches an export
that does not cover these databases: the wrong dataset, a stale export made
before domains were added, a truncated file.

Coverage is not required to be exactly 1.0: domainsplit's own exporter warns
about instances with no embedding ("sequences dropped by --max-len or an OOM at
batch size 1"), so a legitimate pairing can sit a little under 100%.

What this cannot tell you is which *run* an export came from. Pfam accessions are
stable across runs, so a file exported by a different run over the same domain
universe resolves like a native one (its run-local instance ids are a partial and
unreliable backstop at best). Pairing the right export with the right databases is
the caller's job -- and with `--embeddings` left unset the pipeline derives it from
`--input`'s own directory, which gets it right by construction.

Why check at all rather than letting the models find out:
`machine_learning.load_embedding_data` *skips* every instance pair it cannot
resolve, so a file that does not fit yields zero training rows, which is
indistinguishable from a database that genuinely holds no data.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import h5py

from determinism import seed_everything

# What `bin/features/embeddings.py` reads and what domainsplit's
# export_domain_embeddings.py records in the `key_layout` root attribute.
EXPECTED_KEY_LAYOUT = "{pfam_id}/{instance_id}"

#: The layout published before the surrogate-id key was retired. Named so the
#: failure can say *why* the file is unusable instead of reporting 0% coverage.
LEGACY_KEY_LAYOUT = "{domain_id}/{instance_id}"

# The (group, dataset) key the benchmark looks up, from the database's own point
# of view. The group half must stay identical to `embeddings.domain_key_sql()`
# and the dataset half to `embeddings.instance_key_sql()`.
INSTANCE_QUERY = (
    "SELECT domain.pfam_id, "
    "COALESCE(domain_protein_map.instance_id, 'r' || domain_protein_map.rowid) "
    "FROM domain_protein_map "
    "JOIN domain ON domain_protein_map.domain_id = domain.id "
    "ORDER BY domain.pfam_id, domain_protein_map.rowid"
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
    """`[(pfam_id, instance_key)]` for one split database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            (str(pfam_id), str(key)) for pfam_id, key in conn.execute(INSTANCE_QUERY)
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

    for pfam_id, instance_key in instances:
        domains_seen.add(pfam_id)
        if pfam_id not in groups:
            groups[pfam_id] = h5_file.get(pfam_id)
        group = groups[pfam_id]
        if group is not None and instance_key in group:
            resolved += 1
            domains_resolved.add(pfam_id)
        elif len(samples_missing) < 5:
            samples_missing.append(f"{pfam_id}/{instance_key}")

    return (
        resolved,
        len(instances),
        len(domains_resolved),
        len(domains_seen),
        samples_missing,
    )


def check_layout(pairs) -> None:
    """Read root attributes and enforce the key layout.

    Nothing here looks at which domainsplit run a file came from. Pfam
    accessions are stable across runs, so the coverage check below cannot see
    through a foreign export either -- pairing the right export with the right
    databases is the caller's job, and the derived `--embeddings` default does it
    without being asked.
    """
    for feature, path in pairs:
        if not path.exists():
            sys.exit(f"[verify] {feature}: {path} does not exist")
        with h5py.File(path, "r") as h5_file:
            attrs = {k: h5_file.attrs[k] for k in h5_file.attrs}

        described = ", ".join(f"{k}={attrs[k]}" for k in sorted(attrs))
        print(f"[verify] {feature}: {path.name} -- {described or 'no root attributes'}")

        layout = str(attrs.get("key_layout", ""))
        if layout == LEGACY_KEY_LAYOUT:
            sys.exit(
                f"[verify] {feature}: key_layout is {layout!r}. This file is "
                "keyed by `domain.id`, a per-run surrogate integer the benchmark "
                "no longer speaks -- every CSV, feature h5 and prediction file "
                "is written under the Pfam accession so the report can be "
                "compared across runs. Re-export the embeddings with "
                f"key_layout={EXPECTED_KEY_LAYOUT!r}."
            )
        if layout != EXPECTED_KEY_LAYOUT:
            sys.exit(
                f"[verify] {feature}: key_layout is {layout!r}, expected "
                f"{EXPECTED_KEY_LAYOUT!r}. The benchmark reads "
                "h5[pfam_id][instance_key]; a different layout would resolve "
                "nothing and look like an empty training set."
            )


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    pairs = parse_pairs(args.pair)
    check_layout(pairs)

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
            "    `machine_learning.load_embedding_data` *skips* every "
            "(pfam_id, instance_key) it cannot resolve rather than raising, so "
            "an export that does not cover these databases trains on zero rows "
            "and looks exactly like a database holding no data. Point "
            "--embeddings at the run that produced these databases."
        )

    print(f"[verify] OK -- {len(pairs)} feature(s) x {len(split_dbs)} split(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
