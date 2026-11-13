#!/usr/bin/env python3
"""Per-database checks for everything that would otherwise fail *after* ML training.

PREFLIGHT runs beside DDI_EXTRACTION and gates NN/RF/SVM and SASA_CACHE_SHARD,
so a database that would crash EVALUATION or ENRICHMENT fails in minutes rather
than after ~15 h of training. Every check calls the function the late stage
itself calls, so the two cannot diverge:

  ppi        load_data_gm.load_ppi on every split (EVALUATION always calls it)
  combos     every feature combo resolves >= 1 membership instance pair on every
             split; train/validation hold both classes at DDI level (a
             single-class test split only warns). Key presence only: finiteness
             and coverage fractions are VERIFY_EMBEDDINGS' job.
  structure  structure_utils.iter_structures over every split (coverage +
             parse_complex on every complex), plus disulfide_bond_count.encode
             when that encoder is enabled (every CYS has SG)
  sequence   embeddings.clean_sequence over each enabled sequence encoder's own
             SEQUENCE_QUERY with its own MIN_SEQUENCE_LENGTH
  metadata   create_metadata.build_table + eval_enrichment.join_predictions over
             every DDI of every test split (with --metadata)
  toolchain  multiqc on PATH, and the post-training scripts import

All failures are collected and reported together; any failure exits 1.
"""
import argparse
import importlib
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import h5py
import pandas as pd

import load_data_gm
import verify_embeddings
from features import embeddings, structure_utils

TOOLCHAIN = [
    "eval_one", "eval_multiqc", "eval_multiqc_functions", "combine_eval",
    "eval_enrichment", "enrichment_plots", "create_metadata", "statsmodels",
]
TRAINING_SPLITS = ("train", "validation")


def name_value(items):
    """`["a=b", ...]` -> `{"a": "b", ...}`."""
    return dict(item.split("=", 1) for item in items)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db-dir", type=Path, required=True)
    p.add_argument("--split", action="append", default=[], help="Split to check. Repeatable.")
    p.add_argument("--combo", action="append", default=[], help="Comma-separated feature combo. Repeatable.")
    p.add_argument("--published", action="append", default=[], metavar="FEATURE=H5")
    p.add_argument("--extracted", action="append", default=[], metavar="FEATURE=MODULE",
                   help="Extracted (non-structure) feature and its encoder module.")
    p.add_argument("--structure", action="append", default=[], metavar="FEATURE=MODULE")
    p.add_argument("--structures", type=Path, help="domainsplit's structures.h5")
    p.add_argument("--metadata", type=Path, help="Directory with domain_metadata.csv and ddi_metadata.csv")
    p.add_argument("--report", type=Path, default=Path("preflight.txt"))
    return p.parse_args(argv)


def connect(db):
    return closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True))


def ddi_labels(db):
    """{ddi_id: interaction (1 = positive)}; DDI_EXTRACTION has validated `negative`."""
    with connect(db) as conn:
        return {ddi: 1 - int(neg) for ddi, neg in conn.execute("SELECT id, negative FROM domain_domain_interaction")}


class KeyPresence:
    """`(pfam, instance) in` one published h5, cached. No vector is read."""

    def __init__(self, path):
        self.h5 = h5py.File(path, "r")
        self.memo = {}

    def __contains__(self, key):
        if key not in self.memo:
            group = self.h5.get(key[0])
            self.memo[key] = group is not None and key[1] in group
        return self.memo[key]


def check_structures(db, structures, check_sg):
    """Instance pairs (frozensets) with a parsed complex; raises like SASA/extraction would."""
    disulfide = importlib.import_module("features.disulfide_bond_count") if check_sg else None
    covered = set()
    with connect(db) as conn:
        for (_pa, _pb, inst_a, inst_b), cx in structure_utils.iter_structures(conn, structures):
            if disulfide:
                disulfide.encode(cx)
            covered.add(frozenset((inst_a, inst_b)))
    return covered


def check_sequences(db, module_name):
    """Count of instances `clean_sequence` would reject for one encoder, plus examples."""
    module = importlib.import_module(f"features.{module_name}")
    bad = []
    with connect(db) as conn:
        for domain_key, instance_key, sequence in conn.execute(module.SEQUENCE_QUERY):
            try:
                embeddings.clean_sequence(sequence, domain_key, instance_key, min_length=module.MIN_SEQUENCE_LENGTH)
            except ValueError as e:
                bad.append(str(e))
    return bad


def combo_resolution(pairs, labels, combo, published, structure_pairs):
    """{label: DDIs} over the membership pairs every feature of `combo` resolves.

    Extracted encoders cover every pair or fail outright (checked separately),
    so only published keys and, for a combo with a structure feature, the
    structure-covered pairs (`structure_pairs`, None without one) can thin it.
    """
    by_label = {0: set(), 1: set()}
    for ddi, pfam_a, inst_a, pfam_b, inst_b in pairs:
        if any((pfam_a, inst_a) not in published[f] or (pfam_b, inst_b) not in published[f]
               for f in combo if f in published):
            continue
        if structure_pairs is not None and frozenset((inst_a, inst_b)) not in structure_pairs:
            continue
        by_label[labels[ddi]].add(ddi)
    return by_label


def instances_frame(pairs, labels):
    """What DDI_EXTRACTION's `<split>_instances.csv` will hold, from the membership rows."""
    return pd.DataFrame(
        [(pa, pb, ia, ib, labels[ddi]) for ddi, pa, ia, pb, ib in pairs],
        columns=["domain_1", "domain_2", "instance_1", "instance_2", "interaction"],
    )


def check_metadata(metadata_dir, pairs, labels):
    """Every test DDI gets a metadata row: the two calls ENRICHMENT would fail in."""
    import create_metadata
    import eval_enrichment

    instances = instances_frame(pairs, labels)
    table, _ = create_metadata.build_table(
        pd.read_csv(metadata_dir / "domain_metadata.csv"),
        pd.read_csv(metadata_dir / "ddi_metadata.csv"),
        instances,
    )
    canon = {load_data_gm.canonical_pair(a, b) + (y,) for a, b, y in
             zip(instances.domain_1, instances.domain_2, instances.interaction)}
    pred = pd.DataFrame(sorted(canon), columns=["domain_a", "domain_b", "true_interaction"])
    eval_enrichment.join_predictions(pred, table)
    return len(pred)


def check_toolchain():
    if shutil.which("multiqc") is None:
        raise ValueError("multiqc is not on PATH (EVALUATION / COMBINE_EVAL call it)")
    for module in TOOLCHAIN:
        importlib.import_module(module)


def main(argv=None):
    args = parse_args(argv)
    published_paths = name_value(args.published)
    extracted = name_value(args.extracted)
    structure = name_value(args.structure)
    combos = [c.split(",") for c in args.combo]
    splits = sorted(set(args.split)) or verify_embeddings.default_splits(args.db_dir)

    lines, failures, warnings = [], [], []

    def log(msg):
        print(f"[preflight] {msg}", flush=True)
        lines.append(msg)

    def check(what, fn, *a):
        """Run one check; a failure is recorded, not raised, so all are reported."""
        try:
            return fn(*a)
        except (ValueError, KeyError, OSError, ImportError, sqlite3.Error, SystemExit) as e:
            failures.append(f"{what}: {e}")
            return None

    check("toolchain", check_toolchain)
    log(f"toolchain: multiqc + {len(TOOLCHAIN)} module import(s)")

    published = {f: KeyPresence(p) for f, p in sorted(published_paths.items())}
    sequence_modules = sorted({m for m in extracted.values()
                               if hasattr(importlib.import_module(f"features.{m}"), "SEQUENCE_QUERY")})
    check_sg = "disulfide_bond_count" in structure.values()
    if structure and args.structures is None:
        failures.append("structure features enabled but no --structures given")

    for split in splits:
        db = args.db_dir / f"{split}.sqlite3"
        ppi = check(f"{split} ppi", load_data_gm.load_ppi, db)
        if ppi is not None:
            log(f"{split}: {len(ppi)} PPIs, score numeric")

        for module in sequence_modules:
            bad = check(f"{split} {module} sequences", check_sequences, db, module)
            if bad:
                failures.append(f"{split} {module}: {len(bad)} sequence(s) rejected, e.g. {bad[:3]}")
            elif bad is not None:
                log(f"{split}: {module} sequences clean (min_length "
                    f"{importlib.import_module(f'features.{module}').MIN_SEQUENCE_LENGTH})")

        structure_pairs = None
        if structure and args.structures is not None:
            structure_pairs = check(f"{split} structures", check_structures, db, args.structures, check_sg)
            if structure_pairs is not None:
                log(f"{split}: {len(structure_pairs)} complex(es) parsed"
                    + (", every CYS has SG" if check_sg else ""))

        if not combos and args.metadata is None:
            continue
        pairs = check(f"{split} membership", verify_embeddings.read_pairs, db)
        labels = check(f"{split} labels", ddi_labels, db)
        if pairs is None or labels is None:
            continue

        for combo in combos:
            if any(f in structure for f in combo):
                if structure_pairs is None:
                    continue  # the structure failure is already recorded
                by_label = combo_resolution(pairs, labels, combo, published, structure_pairs)
            else:
                by_label = combo_resolution(pairs, labels, combo, published, None)
            n_pos, n_neg = len(by_label[1]), len(by_label[0])
            name = "+".join(combo)
            if n_pos + n_neg == 0:
                failures.append(f"{split} [{name}]: no membership instance pair resolves in every feature")
            elif min(n_pos, n_neg) == 0:
                msg = f"{split} [{name}]: one class only ({n_pos} positive, {n_neg} negative DDIs)"
                (failures if split in TRAINING_SPLITS else warnings).append(msg)
            else:
                log(f"{split} [{name}]: {n_pos} positive / {n_neg} negative DDIs resolve")

        if args.metadata is not None and split not in TRAINING_SPLITS:
            n = check(f"{split} metadata", check_metadata, args.metadata, pairs, labels)
            if n is not None:
                log(f"{split}: metadata row for all {n} DDIs")

    for w in warnings:
        log(f"WARNING {w}")
    for f in failures:
        log(f"FAILED {f}")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        print(f"[preflight] {len(failures)} check(s) failed for {args.db_dir}; nothing was trained.", file=sys.stderr)
        return 1
    log(f"OK: {len(splits)} split(s), {len(combos)} combo(s)")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
