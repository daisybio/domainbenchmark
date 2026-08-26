#!/usr/bin/env python3
"""Generate the in-repo test fixture: a tiny, synthetic `databases/` tree.

The fixture mirrors what daisybio/domainsplit publishes, at a scale that runs
in seconds:

    <out>/random/         train, validation, test_balanced, test_realistic
    <out>/external_test/  train, validation, test

so `-profile test` exercises both the two-variant path (one training run
scored against two test sets) and the single-test path.

Everything is synthesised — no real database is needed, and the generator is
deterministic (fixed seed), so re-running it produces byte-identical content
for the non-blob columns.

Deliberate properties the pipeline depends on:

* the schema is domainsplit's own (see its `INIT_DOMAINSPLIT_DB` module),
  including `domain_protein_map.instance_id` and `ddi_split_membership`;
* one protein carries **two instances of the same domain family**, which
  collide under the old `h5[domain_id][protein_id]` layout and only work with
  instance-level keys;
* `ddi_split_membership` names the exact instance pairs of each split, which is
  what the ML loader instantiates;
* embedding columns hold pickled numpy arrays in the shapes the encoders
  expect: 2-D `(residues, dim)` per-residue, 1-D `(dim,)` per-domain.

Usage:

    python tests/create_test_data.py --out tests/data/databases
"""

import argparse
import pickle
import random
import sqlite3
from pathlib import Path

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
EMBED_DIM = 8
SEED = 42

# Per split: how many domain families, proteins, and DDI pairs to synthesise.
N_DOMAINS = 12
N_PROTEINS = 10
N_DDI = 20  # half positive, half negative

SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE domain (id INTEGER PRIMARY KEY, pfam_id, name, UNIQUE(pfam_id));
CREATE TABLE domain_go_terms(
    domain_id REFERENCES domain ON DELETE CASCADE,
    go_accession
);
CREATE TABLE domain_domain_interaction (
    id INTEGER PRIMARY KEY,
    domain_id_a, domain_id_b, negative,
    source VARCHAR(255),
    FOREIGN KEY(domain_id_a) REFERENCES domain ON DELETE CASCADE,
    FOREIGN KEY(domain_id_b) REFERENCES domain ON DELETE CASCADE,
    UNIQUE(domain_id_a, domain_id_b)
);
CREATE TABLE protein (
    id INTEGER PRIMARY KEY,
    uniprot_id,
    sequence,
    prott5_per_residue,
    esm3_per_residue,
    esmc_per_residue,
    UNIQUE(uniprot_id)
);
CREATE TABLE protein_go_terms(
    protein_id REFERENCES protein ON DELETE CASCADE,
    go_accession
);
CREATE TABLE protein_protein_interaction (
    protein_id_a REFERENCES protein ON DELETE CASCADE,
    protein_id_b REFERENCES protein ON DELETE CASCADE,
    score,
    UNIQUE(protein_id_a, protein_id_b)
);
CREATE TABLE domain_protein_map (
    domain_id REFERENCES domain ON DELETE CASCADE,
    protein_id REFERENCES protein ON DELETE CASCADE,
    domain_sequence, start_pos, end_pos,
    esm3_per_domain, esmc_per_domain,
    instance_id, clan, taxon_id,
    UNIQUE(domain_id, protein_id, start_pos, end_pos)
);
CREATE TABLE ddi_split_membership (
    ddi_id REFERENCES domain_domain_interaction ON DELETE CASCADE,
    method, split,
    instance_id_a, instance_id_b,
    UNIQUE(ddi_id, method, split, instance_id_a, instance_id_b)
);
CREATE INDEX idx_domain_domain_interaction_domain_id_a
    ON domain_domain_interaction (domain_id_a);
CREATE INDEX idx_domain_domain_interaction_domain_id_b
    ON domain_domain_interaction (domain_id_b);
CREATE INDEX idx_domain_protein_map_domain_id ON domain_protein_map (domain_id);
CREATE INDEX idx_domain_protein_map_protein_id ON domain_protein_map (protein_id);
CREATE UNIQUE INDEX idx_domain_protein_map_instance_id
    ON domain_protein_map (instance_id) WHERE instance_id IS NOT NULL;
CREATE INDEX idx_ddi_split_membership_method_split
    ON ddi_split_membership (method, split);
CREATE INDEX idx_ddi_split_membership_ddi_id ON ddi_split_membership (ddi_id);
"""


def blob(rng, shape):
    """Pickled numpy array, matching what the embedding encoders unpickle."""
    return pickle.dumps(rng.standard_normal(shape).astype(np.float32))


def sequence(rng, length):
    return "".join(rng.choice(list(AA), size=length))


# Provenance labels, mirroring what domainsplit writes: `source` is a
# comma-joined list of every source that contributed the pair, positives and
# negatives come from disjoint sources, and one pair is left without a source at
# all so the report's `unknown` bucket has something to collect.
POSITIVE_SOURCES = ["3did", "single_domain_ppi,PPIDM,PPIDM_Gold", "PPIDM"]
NEGATIVE_SOURCES = ["sampled_negative", "negatome"]
UNSOURCED_DDI_INDEX = 4


def ddi_source(idx: int, negative: int):
    if idx == UNSOURCED_DDI_INDEX:
        return None
    pool = NEGATIVE_SOURCES if negative else POSITIVE_SOURCES
    return pool[(idx // 2) % len(pool)]


def build_split(path: Path, method: str, split: str, offset: int) -> None:
    """Write one split database.

    `offset` shifts the synthetic namespace so different splits describe
    disjoint domain families and proteins — which is what a split is.
    """
    rng = np.random.default_rng(SEED + offset)
    pyrng = random.Random(SEED + offset)

    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    # ---- domains ---------------------------------------------------------
    domains = [(i + 1, f"PF{offset + i:05d}", f"domain_{offset + i}") for i in range(N_DOMAINS)]
    conn.executemany("INSERT INTO domain VALUES (?, ?, ?)", domains)
    conn.executemany(
        "INSERT INTO domain_go_terms VALUES (?, ?)",
        [(d[0], f"GO:{offset + d[0]:07d}") for d in domains],
    )

    # ---- proteins --------------------------------------------------------
    proteins = []
    for i in range(N_PROTEINS):
        seq = sequence(rng, 60)
        proteins.append(
            (
                i + 1,
                f"P{offset + i:05d}",
                seq,
                blob(rng, (len(seq), EMBED_DIM)),  # prott5_per_residue
                blob(rng, (len(seq), EMBED_DIM)),  # esm3_per_residue
                blob(rng, (len(seq), EMBED_DIM)),  # esmc_per_residue
            )
        )
    conn.executemany("INSERT INTO protein VALUES (?, ?, ?, ?, ?, ?)", proteins)
    conn.executemany(
        "INSERT INTO protein_go_terms VALUES (?, ?)",
        [(p[0], f"GO:{offset + p[0] + 500:07d}") for p in proteins],
    )
    conn.executemany(
        "INSERT OR IGNORE INTO protein_protein_interaction VALUES (?, ?, ?)",
        [
            (a, b, round(pyrng.uniform(0.4, 1.0), 3))
            for a in range(1, N_PROTEINS + 1)
            for b in range(a + 1, N_PROTEINS + 1)
            if pyrng.random() < 0.35
        ],
    )

    # ---- domain instances ------------------------------------------------
    # Every family gets one instance in each of two proteins. Family 1 gets a
    # SECOND instance inside its first protein (a tandem repeat): two rows with
    # the same (domain_id, protein_id), distinguished only by start/end. That
    # pair is what collides unless the HDF5 key is the instance.
    dpm_rows = []
    instances_by_domain = {}
    for domain_id, _pfam, _name in domains:
        carriers = [
            ((domain_id - 1) % N_PROTEINS) + 1,
            (domain_id % N_PROTEINS) + 1,
        ]
        placements = [(protein_id, 0, 25) for protein_id in carriers]
        if domain_id == 1:
            placements.append((carriers[0], 30, 55))

        instance_ids = []
        for protein_id, start, end in placements:
            protein_seq = proteins[protein_id - 1][2]
            instance_id = f"inst_{offset}_{domain_id}_{protein_id}_{start}"
            instance_ids.append(instance_id)
            dpm_rows.append(
                (
                    domain_id,
                    protein_id,
                    protein_seq[start:end],
                    start,
                    end,
                    blob(rng, (EMBED_DIM,)),  # esm3_per_domain
                    blob(rng, (EMBED_DIM,)),  # esmc_per_domain
                    instance_id,
                    f"CL{domain_id:04d}",
                    "9606",
                )
            )
        instances_by_domain[domain_id] = instance_ids

    conn.executemany(
        "INSERT INTO domain_protein_map VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", dpm_rows
    )

    # ---- DDIs + split membership ----------------------------------------
    # Pairs are drawn deterministically and labelled half positive, half
    # negative. Each pair's membership rows are the cross-product of the two
    # families' instances, canonicalised the way domainsplit canonicalises it
    # (sorted instance ids), so the pipeline has to resolve sides through
    # domain_protein_map rather than trusting column order.
    pairs = []
    a = 1
    b = 2
    while len(pairs) < N_DDI and a <= N_DOMAINS:
        if b > N_DOMAINS:
            a += 1
            b = a + 1
            continue
        pairs.append((a, b))
        b += 2

    ddi_rows = []
    membership_rows = []
    for idx, (domain_a, domain_b) in enumerate(pairs):
        ddi_id = idx + 1
        negative = 1 if idx % 2 else 0
        ddi_rows.append((ddi_id, domain_a, domain_b, negative, ddi_source(idx, negative)))
        for instance_a in instances_by_domain[domain_a]:
            for instance_b in instances_by_domain[domain_b]:
                first, second = sorted((instance_a, instance_b))
                membership_rows.append((ddi_id, method, split, first, second))

    conn.executemany(
        "INSERT INTO domain_domain_interaction VALUES (?, ?, ?, ?, ?)", ddi_rows
    )
    conn.executemany(
        "INSERT OR IGNORE INTO ddi_split_membership VALUES (?, ?, ?, ?, ?)",
        membership_rows,
    )

    conn.commit()
    conn.close()
    print(
        f"  {path.name}: {len(domains)} domains, {len(proteins)} proteins, "
        f"{len(dpm_rows)} instances, {len(ddi_rows)} DDIs, "
        f"{len(membership_rows)} membership rows"
    )


# dataset -> its splits. `random` has an internal test set and therefore two
# test variants; `external_test` gets its single test from BUILD_EXTERNAL_TEST.
DATASETS = {
    "random": ["train", "validation", "test_balanced", "test_realistic"],
    "external_test": ["train", "validation", "test"],
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", default="tests/data/databases",
        help="Output directory for the databases/ tree (default: tests/data/databases)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    offset = 0
    for dataset, splits in DATASETS.items():
        print(f"\n[{dataset}]")
        dataset_dir = out / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            offset += 100
            build_split(dataset_dir / f"{split}.sqlite3", dataset, split, offset)

    print(f"\nWrote fixture to {out}")


if __name__ == "__main__":
    main()
