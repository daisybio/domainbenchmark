#!/usr/bin/env python3
"""Verify a published domainsplit embedding file against the split databases it will be paired with.

The published files are keyed `h5[pfam_id][instance_key]`, and this script asserts
that the join actually resolves before a single GPU hour is spent on it.

For every split database named with `--split`, take the domain-instance pairs
`ddi_split_membership` assigns to it -- exactly what DDI_EXTRACTION writes to
`<split>_instances.csv` and what `machine_learning.load_embedding_data` will
instantiate -- and count the pairs whose *both* sides the HDF5 carries with an
all-finite vector. Below `--min-coverage` is a hard failure. This catches an
export that does not cover these databases: the wrong dataset, a stale export
made before domains were added, a truncated file, or vectors gone NaN.

Pairs rather than instances, because that is the unit the loader drops: one
missing side, or a NaN on either, loses the row. With instance coverage c the
pair coverage is roughly c**2, so an instance-level gate overstated what
reached the model. The DDI-level figure (DDIs keeping at least one pair) is
logged too: that is what the report actually scores.

Coverage is not required to be exactly 1.0: domainsplit's own exporter warns
about instances with no embedding ("sequences dropped by --max-len or an OOM at
batch size 1"), so a legitimate pairing can sit a little under 100%.

What this cannot tell you is which *run* an export came from. Pfam accessions and
domainsplit's instance ids (`<family>_<accession>_<start>_<end>`) are both stable
across runs, so a file exported by a different run over the same domain universe
resolves like a native one. Pairing the right export with the right databases is
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
import numpy as np

from determinism import seed_everything

# What `bin/features/embeddings.py` reads and what domainsplit's
# export_domain_embeddings.py records in the `key_layout` root attribute.
EXPECTED_KEY_LAYOUT = "{pfam_id}/{instance_id}"

#: The layout published before the surrogate-id key was retired. Named so the
#: failure can say *why* the file is unusable instead of reporting 0% coverage.
LEGACY_KEY_LAYOUT = "{domain_id}/{instance_id}"

# The instance pairs the loader will look up, from the database's own point of
# view: DDI_EXTRACTION's INSTANCE_QUERY, minus the orientation it resolves (a
# pair is covered or not whichever way round it is read). Each side's Pfam
# accession comes from its own `domain_protein_map` row, as in
# `embeddings.domain_key_sql()`. The instance half is the raw `instance_id`,
# which is what `<split>_instances.csv` carries and the loader looks up; it is
# equal to the encoders' `embeddings.instance_key_sql()` here because a
# membership row can only name an instance through a non-NULL instance_id.
#
# LEFT JOINs so a membership row naming an instance the database does not hold
# surfaces as a NULL accession instead of vanishing -- DDI_EXTRACTION fails on
# the same rows.
PAIR_QUERY = (
    "SELECT m.ddi_id, da.pfam_id, m.instance_id_a, db.pfam_id, m.instance_id_b "
    "FROM ddi_split_membership AS m "
    "LEFT JOIN domain_protein_map AS pa ON pa.instance_id = m.instance_id_a "
    "LEFT JOIN domain AS da ON pa.domain_id = da.id "
    "LEFT JOIN domain_protein_map AS pb ON pb.instance_id = m.instance_id_b "
    "LEFT JOIN domain AS db ON pb.domain_id = db.id "
    "ORDER BY m.ddi_id, m.instance_id_a, m.instance_id_b"
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
        "--split",
        action="append",
        default=[],
        help="Split database to check, as `<db-dir>/<split>.sqlite3`. Repeatable. "
        "Defaults to the train/validation/test* set discoverSplits() picks; "
        "other *.sqlite3 in the directory are never read.",
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
        default=0.95,
        help="Minimum fraction of a split's membership instance pairs whose two "
        "sides the HDF5 resolves to finite vectors. Below this the pairing is "
        "rejected. Default: 0.95.",
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


def default_splits(db_dir: Path):
    """train, validation and every test / test_* split -- discoverSplits()'s rule."""
    names = sorted(p.stem for p in db_dir.glob("*.sqlite3"))
    return [n for n in names if n in ("train", "validation") or n == "test" or n.startswith("test_")]


def read_pairs(db_path: Path):
    """`[(ddi_id, pfam_a, instance_a, pfam_b, instance_b)]` for one split database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        has_membership = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ddi_split_membership'"
        ).fetchone()
        if not has_membership:
            sys.exit(
                f"[verify] {db_path}: no ddi_split_membership table. Coverage is "
                "measured over the instance pairs it assigns, which is all the "
                "loader instantiates."
            )
        rows = conn.execute(PAIR_QUERY).fetchall()
    finally:
        conn.close()

    dangling = [r for r in rows if r[1] is None or r[3] is None]
    if dangling:
        sys.exit(
            f"[verify] {db_path}: {len(dangling)} membership row(s) name an "
            f"instance missing from domain_protein_map (e.g. {dangling[:3]}). "
            "DDI_EXTRACTION rejects these too; the database is inconsistent."
        )
    return [(ddi, str(pa), str(ia), str(pb), str(ib)) for ddi, pa, ia, pb, ib in rows]


class InstanceLookup:
    """Per-(pfam_id, instance_key) verdict, cached across pairs and splits.

    The vector is read, not just the key probed: the loader drops a row with
    a NaN in it, so a present-but-NaN vector is as lost as a missing one. Each
    instance is read once however many pairs and splits it appears in.
    """

    def __init__(self, h5_file: h5py.File):
        self.h5_file = h5_file
        self.groups = {}
        self.verdicts = {}

    def __call__(self, pfam_id, instance_key):
        """Return "ok", "missing" or "non-finite"."""
        key = (pfam_id, instance_key)
        if key not in self.verdicts:
            if pfam_id not in self.groups:
                self.groups[pfam_id] = self.h5_file.get(pfam_id)
            group = self.groups[pfam_id]
            dataset = group.get(instance_key) if group is not None else None
            if not isinstance(dataset, h5py.Dataset):
                verdict = "missing"
            elif not np.isfinite(dataset[()]).all():
                verdict = "non-finite"
            else:
                verdict = "ok"
            self.verdicts[key] = verdict
        return self.verdicts[key]


def coverage(lookup: InstanceLookup, pairs):
    """Pair- and DDI-level coverage of one split's membership pairs."""
    resolved = 0
    ddis, ddis_resolved = set(), set()
    instances_bad = {}
    samples = []

    for ddi_id, pfam_a, instance_a, pfam_b, instance_b in pairs:
        ddis.add(ddi_id)
        verdict_a = lookup(pfam_a, instance_a)
        verdict_b = lookup(pfam_b, instance_b)
        if verdict_a == verdict_b == "ok":
            resolved += 1
            ddis_resolved.add(ddi_id)
            continue
        for pfam_id, instance_key, verdict in (
            (pfam_a, instance_a, verdict_a),
            (pfam_b, instance_b, verdict_b),
        ):
            if verdict != "ok" and (pfam_id, instance_key) not in instances_bad:
                instances_bad[(pfam_id, instance_key)] = verdict
                if len(samples) < 5:
                    samples.append(f"{pfam_id}/{instance_key} ({verdict})")

    return {
        "pairs": len(pairs),
        "pairs_resolved": resolved,
        "ddis": len(ddis),
        "ddis_resolved": len(ddis_resolved),
        "missing": sum(v == "missing" for v in instances_bad.values()),
        "non_finite": sum(v == "non-finite" for v in instances_bad.values()),
        "samples": samples,
    }


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

    # Only the splits the benchmark reads. A domainsplit `databases/<name>/`
    # directory can hold many more *.sqlite3 than train/validation/test*, and
    # checking those would cost time and could fail on a database nothing uses.
    splits = sorted(set(args.split)) or default_splits(args.db_dir)
    if not splits:
        sys.exit(f"[verify] no train/validation/test*.sqlite3 under {args.db_dir}")
    split_dbs = [args.db_dir / f"{split}.sqlite3" for split in splits]
    absent = [str(p) for p in split_dbs if not p.is_file()]
    if absent:
        sys.exit(f"[verify] split database(s) not found: {', '.join(absent)}")
    split_pairs = {db_path.stem: read_pairs(db_path) for db_path in split_dbs}

    failures = []
    for feature, path in pairs:
        with h5py.File(path, "r") as h5_file:
            lookup = InstanceLookup(h5_file)
            for split, membership in split_pairs.items():
                if not membership:
                    print(f"[verify] {feature} x {split}: no membership pairs -- skipped")
                    continue

                c = coverage(lookup, membership)
                frac = c["pairs_resolved"] / c["pairs"]
                print(
                    f"[verify] {feature} x {split}: {c['pairs_resolved']}/{c['pairs']} "
                    f"instance pairs ({frac:.2%}); "
                    f"{c['ddis_resolved']}/{c['ddis']} DDIs keep >=1 pair "
                    f"({c['ddis_resolved'] / c['ddis']:.2%}); "
                    f"{c['missing']} instance(s) missing, {c['non_finite']} non-finite"
                )
                if frac < args.min_coverage:
                    failures.append(
                        f"  {feature} x {split}: {frac:.2%} < {args.min_coverage:.2%} "
                        f"(e.g. {', '.join(c['samples'])})"
                    )

    if failures:
        sys.exit(
            "[verify] embedding files do not match these databases:\n"
            + "\n".join(failures)
            + "\n"
            "    `machine_learning.load_embedding_data` *skips* every "
            "instance pair it cannot resolve, or that carries a NaN, rather than "
            "raising, so an export that does not cover these databases trains "
            "on a fraction of the split -- or zero rows, which looks exactly "
            "like a database holding no data. Point "
            "--embeddings at the run that produced these databases."
        )

    print(f"[verify] OK -- {len(pairs)} feature(s) x {len(split_dbs)} split(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
