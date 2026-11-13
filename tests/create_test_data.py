#!/usr/bin/env python3
"""Generate the in-repo test fixture: a tiny, synthetic `databases/` tree.

The fixture mirrors what daisybio/domainsplit publishes, at a scale that runs
in seconds:

    <out>/random/         train, validation, test_balanced, test_realistic
    <out>/external_test/  train, validation, test

so `-profile test` exercises both the two-variant path (one training run
scored against two test sets) and the single-test path. Alongside it, the
per-run embedding files domainsplit publishes:

    <embeddings-out>/esm3_domain_embeddings.h5
    <embeddings-out>/esmc_domain_embeddings.h5
    <embeddings-out>/prott5_domain_embeddings.h5

and, beside `databases/`, the AF3 structures and the enrichment metadata:

    structures/structures.h5          domainsplit's ENRICH_STRUCTURAL layout
    metadata/domain_metadata.csv      per-instance numeric metadata
    metadata/ddi_metadata.csv         per-instance-pair af_* metadata

Everything is synthesised — no real database is needed, and the generator is
deterministic (fixed seed), so re-running it produces byte-identical content
for the non-blob columns.

Deliberate properties the pipeline depends on:

* the schema is domainsplit's own (see its `INIT_DOMAINSPLIT_DB` module),
  including `domain_protein_map.instance_id` and `ddi_split_membership`;
* one protein carries **two instances of the same domain family**, which
  collide under a `h5[domain][protein_id]` layout and only work with
  instance-level keys;
* `ddi_split_membership` names the exact instance pairs of each split, which is
  what the ML loader instantiates;
* instance ids are domainsplit's `<pfam>_<uniprot>_<start>_<end>`, because its
  structure writer derives the structures.h5 group from that prefix;
* one homodimer DDI (family 1 with itself) per split;
* test splits hold 151 DDIs (train/validation 21), enough for the enrichment
  regressions to fit rather than skip (see N_DDI_TEST);
* every CYS in a complex has an SG atom, and CYS pairs in reach are bridged
  into disulfides, so disulfide_bond_count has positive cases;
* structure coverage is partial: `has_structure = 0` on one pair of a positive
  (the DDI keeps its other pairs) and on every pair of a negative (the DDI has
  no structure at all), so the structure features' coverage logging and drop
  path both run;
* the databases carry **no embedding BLOBs at all**. domainsplit embeds the
  cut domain sequence outside the database now, so `protein.*_per_residue` and
  `domain_protein_map.*_per_domain` are gone from the schema and the encoders
  that read them are gone from `bin/features/`;
* `domain.id` is 1..N in *every* split -- a per-run surrogate integer, which is
  why nothing the pipeline writes is keyed on it. The published embedding files
  are keyed `h5[pfam_id][instance_id]` like everything else and written once over
  the union of all splits. Pfam accessions are offset-shifted per split, so the
  splits stay disjoint in the key space too; `instance_id` is globally unique.
  The `domainsplit_run` root attribute is written because the real exporter
  writes it, not because anything reads it -- the pipeline never checks it
  (the `--domainsplit_run` param that did is gone).

Usage:

    python tests/create_test_data.py --out tests/data/databases
"""

import argparse
import gzip
import hashlib
import random
import sqlite3
from pathlib import Path

import h5py
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
EMBED_DIM = 8
SEED = 42

# What domainsplit publishes under `results/embeddings/`, and the feature names
# the benchmark maps them onto (`params.published_features`).
EMBEDDING_MODELS = ["esm3", "esmc", "prott5"]
# Stands in for domainsplit's `workflow.sessionId`, written to the
# `domainsplit_run` root attribute for fidelity only -- nothing reads it.
FIXTURE_RUN_ID = "fixture-run-0001"

# Per split: how many domain families, proteins, and DDI pairs to synthesise.
N_DOMAINS = 12
N_PROTEINS = 10
N_DDI = 20  # half positive, half negative
# Test splits are larger so the enrichment fits run instead of skipping:
# eval_enrichment needs MIN_ROWS_PER_PARAMETER (10) complete rows per parameter,
# and the fixture metadata gives 6 non-constant covariates + intercept = 7, so
# >= 70 complete rows *per class* for error_positive / error_negative. 150 DDIs
# (+ the homodimer) leave 76 positives and 74 structure-covered negatives.
# Families other than 1 carry one instance there, so a DDI is one complex and
# the SASA/structure cost stays small.
N_DOMAINS_TEST = 25
N_DDI_TEST = 150

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
    instance_id, clan, taxon_id,
    UNIQUE(domain_id, protein_id, start_pos, end_pos)
);
CREATE TABLE ddi_split_membership (
    ddi_id REFERENCES domain_domain_interaction ON DELETE CASCADE,
    method, split,
    instance_id_a, instance_id_b,
    has_structure INTEGER NOT NULL DEFAULT 0,
    z_score REAL,
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


def sequence(rng, length):
    return "".join(rng.choice(list(AA), size=length))


# Provenance labels, mirroring what domainsplit writes: `source` is a
# comma-joined list of every source that contributed the pair, positives and
# negatives come from disjoint sources, and one pair is left without a source at
# all so the report's `unknown` bucket has something to collect.
POSITIVE_SOURCES = ["3did", "single_domain_ppi,PPIDM,PPIDM_Gold", "PPIDM"]
NEGATIVE_SOURCES = ["sampled_negative", "negatome"]
UNSOURCED_DDI_INDEX = 4
# Structure coverage holes (see build_split): a positive with one pair missing,
# a negative with every pair missing.
PARTIAL_STRUCTURE_DDI_INDEX = 0
NO_STRUCTURE_DDI_INDEX = 7


def ddi_source(idx: int, negative: int):
    if idx == UNSOURCED_DDI_INDEX:
        return None
    pool = NEGATIVE_SOURCES if negative else POSITIVE_SOURCES
    return pool[(idx // 2) % len(pool)]


def build_split(path: Path, method: str, split: str, offset: int) -> list:
    """Write one split database; return its `[(pfam_id, instance_key)]`.

    `offset` shifts the synthetic namespace so different splits describe
    disjoint domain families and proteins — which is what a split is. Note what
    it does *not* shift: `domain.id` is 1..N in every split, because it is a
    surrogate integer domainsplit assigns per run and copies verbatim into each
    subset. Which is why the returned key is the Pfam accession, not the id: it
    is what the feature files, the DDI CSVs and the predictions are all keyed on.
    """
    rng = np.random.default_rng(SEED + offset)
    pyrng = random.Random(SEED + offset)
    test = split.startswith("test")
    n_domains = N_DOMAINS_TEST if test else N_DOMAINS
    n_ddi = N_DDI_TEST if test else N_DDI

    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    # ---- domains ---------------------------------------------------------
    domains = [(i + 1, f"PF{offset + i:05d}", f"domain_{offset + i}") for i in range(n_domains)]
    conn.executemany("INSERT INTO domain VALUES (?, ?, ?)", domains)
    conn.executemany(
        "INSERT INTO domain_go_terms VALUES (?, ?)",
        [(d[0], f"GO:{offset + d[0]:07d}") for d in domains],
    )

    # ---- proteins --------------------------------------------------------
    proteins = []
    for i in range(N_PROTEINS):
        seq = sequence(rng, 60)
        proteins.append((i + 1, f"P{offset + i:05d}", seq))
    conn.executemany("INSERT INTO protein VALUES (?, ?, ?)", proteins)
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
    # pair is what collides unless the HDF5 key is the instance. In test splits
    # the other families keep only their first carrier (see N_DDI_TEST).
    dpm_rows = []
    instances_by_domain = {}
    instance_info = {}  # instance_id -> (pfam_id, uniprot_id, domain sequence)
    for domain_id, _pfam, _name in domains:
        carriers = [
            ((domain_id - 1) % N_PROTEINS) + 1,
            (domain_id % N_PROTEINS) + 1,
        ]
        if test and domain_id != 1:
            carriers = carriers[:1]
        placements = [(protein_id, 0, 25) for protein_id in carriers]
        if domain_id == 1:
            placements.append((carriers[0], 30, 55))

        instance_ids = []
        for protein_id, start, end in placements:
            protein_seq = proteins[protein_id - 1][2]
            # domainsplit's `<family>_<uniprot>_<start>_<end>`: its structure
            # writer derives the structures.h5 group from this prefix.
            instance_id = f"{_pfam}_{proteins[protein_id - 1][1]}_{start}_{end}"
            instance_ids.append(instance_id)
            instance_info[instance_id] = (
                _pfam, proteins[protein_id - 1][1], protein_seq[start:end]
            )
            dpm_rows.append(
                (
                    domain_id,
                    protein_id,
                    protein_seq[start:end],
                    start,
                    end,
                    instance_id,
                    f"CL{domain_id:04d}",
                    "9606",
                )
            )
        instances_by_domain[domain_id] = instance_ids

    conn.executemany(
        "INSERT INTO domain_protein_map VALUES (?, ?, ?, ?, ?, ?, ?, ?)", dpm_rows
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
    while len(pairs) < n_ddi and a <= n_domains:
        if b > n_domains:
            a += 1
            b = a + 1
            continue
        pairs.append((a, b))
        b += 2
    # One homodimer: family 1 with itself, over its distinct instances.
    pairs.append((1, 1))

    ddi_rows = []
    membership_rows = []
    for idx, (domain_a, domain_b) in enumerate(pairs):
        ddi_id = idx + 1
        negative = 1 if idx % 2 else 0
        ddi_rows.append((ddi_id, domain_a, domain_b, negative, ddi_source(idx, negative)))
        instance_pairs = sorted({
            tuple(sorted((instance_a, instance_b)))
            for instance_a in instances_by_domain[domain_a]
            for instance_b in instances_by_domain[domain_b]
            if instance_a != instance_b
        })
        for row, (first, second) in enumerate(instance_pairs):
            # Structure coverage is partial on purpose, as AF3 coverage is:
            # one positive keeps a structure on all but its first pair (the DDI
            # survives), one negative has none at all (the DDI drops out of
            # every structure-feature run).
            missing = (idx == PARTIAL_STRUCTURE_DDI_INDEX and row == 0) or (
                idx == NO_STRUCTURE_DDI_INDEX
            )
            has_structure = 0 if missing else 1
            z_score = None if missing else round(float(rng.normal()), 4)
            membership_rows.append(
                (ddi_id, method, split, first, second, has_structure, z_score)
            )

    conn.executemany(
        "INSERT INTO domain_domain_interaction VALUES (?, ?, ?, ?, ?)", ddi_rows
    )
    conn.executemany(
        "INSERT OR IGNORE INTO ddi_split_membership VALUES (?, ?, ?, ?, ?, ?, ?)",
        membership_rows,
    )

    conn.commit()
    conn.close()
    print(
        f"  {path.name}: {len(domains)} domains, {len(proteins)} proteins, "
        f"{len(dpm_rows)} instances, {len(ddi_rows)} DDIs, "
        f"{len(membership_rows)} membership rows"
    )
    # `(domain.pfam_id, COALESCE(instance_id, 'r' || rowid))` -- every fixture
    # row has an instance_id, so the second half is the instance_id itself.
    pfam_by_id = {d[0]: d[1] for d in domains}
    instances = [(str(pfam_by_id[row[0]]), str(row[5])) for row in dpm_rows]
    sources = {r[0]: r[4] for r in ddi_rows}
    memberships = [
        (instance_a, instance_b, has_structure, sources[ddi_id])
        for ddi_id, _m, _s, instance_a, instance_b, has_structure, _z in membership_rows
    ]
    return instances, instance_info, memberships


# dataset -> its splits. `random` has an internal test set and therefore two
# test variants; `external_test` gets its single test from BUILD_EXTERNAL_TEST.
DATASETS = {
    "random": ["train", "validation", "test_balanced", "test_realistic"],
    "external_test": ["train", "validation", "test"],
}


THREE_LETTER = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}


def _key_rng(*tokens) -> np.random.Generator:
    """RNG seeded on a string key -- sha256, because `hash()` is salted."""
    digest = hashlib.sha256("/".join(tokens).encode()).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:8], "big"))


SG_BOND = 1.81  # CB-SG
SS_BOND = 2.05  # SG-SG of a disulfide
SS_REACH = 2 * SG_BOND + SS_BOND  # CBs closer than this can be bridged


def _bridge(cb_p, cb_q):
    """SG positions 1.81 A from each CB and 2.05 A from each other."""
    d = float(np.linalg.norm(cb_q - cb_p))
    u = (cb_q - cb_p) / d
    a = (d - SS_BOND) / 2  # along the CB-CB axis
    w = np.array([0.0, 0.0, -1.0]) + u[2] * u  # -z, made perpendicular to u
    w /= np.linalg.norm(w)
    h = np.sqrt(SG_BOND ** 2 - a ** 2)
    return cb_p + a * u + h * w, cb_q - a * u + h * w


def pdb_text(chains) -> str:
    """Two-chain PDB: backbone + CB per residue (no CB on GLY), SG on CYS.

    `chains` is `[(chain_id, sequence, y_offset, rng)]`. Residues run along x
    with a little jitter, one chain per y offset, so the chains touch: enough
    for Shrake-Rupley, contacts and backbone angles to give non-trivial values.
    Same layout as domainsplit's tests/python/test_structure_scoring.py.

    Every CYS gets an SG 1.81 A below its CB. Two CYS whose CBs are in reach
    (adjacent in a chain, or facing across a narrow chain gap) are bridged
    instead, closest first, each CYS at most once: SG-SG 2.05 A, a disulfide.
    """
    residues = []
    for chain_id, seq, y, rng in chains:
        for resseq, aa in enumerate(seq, start=1):
            x = 3.8 * resseq
            jitter = rng.normal(0.0, 0.3, size=3)
            atoms = [("N", -1.2, 0.0), ("CA", 0.0, 0.0), ("C", 1.2, 0.0), ("O", 1.2, 1.2)]
            if aa != "G":
                atoms.append(("CB", 0.0, -1.5))
            coords = {name: np.array([x + dx, y, dz]) + jitter for name, dx, dz in atoms}
            residues.append((chain_id, resseq, aa, coords))

    cys = [coords for _c, _r, aa, coords in residues if aa == "C"]
    reach = sorted(
        (float(np.linalg.norm(p["CB"] - q["CB"])), i, j)
        for i, p in enumerate(cys) for j, q in enumerate(cys) if i < j
    )
    for d, i, j in reach:
        if d < SS_REACH and "SG" not in cys[i] and "SG" not in cys[j]:
            cys[i]["SG"], cys[j]["SG"] = _bridge(cys[i]["CB"], cys[j]["CB"])
    for coords in cys:
        coords.setdefault("SG", coords["CB"] + np.array([0.0, 0.0, -SG_BOND]))

    lines, serial, chain = [], 1, residues[0][0]
    for chain_id, resseq, aa, coords in residues:
        if chain_id != chain:
            lines.append("TER")
            chain = chain_id
        for name, (cx, cy, cz) in coords.items():
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} {THREE_LETTER[aa]} {chain_id}{resseq:4d}    "
                f"{cx:8.3f}{cy:8.3f}{cz:8.3f}"
                f"{1.0:6.2f}{90.0:6.2f}          {name[0]:>2s}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_structure_file(path: Path, instance_info, memberships) -> None:
    """`structures.h5` in domainsplit's ENRICH_STRUCTURAL layout.

    One AF3-like complex per instance pair with `has_structure = 1`:
    `h5["<pfamA>_<pfamB>"]["<instA>__<instB>"]` -> uint8 gzip'd PDB text,
    chain A = instance A, keyed in the membership row's own orientation
    (domainsplit's `structure_h5_keys`). gzip mtime is pinned so the file is
    byte-identical between generator runs.
    """
    if path.exists():
        path.unlink()
    n = 0
    with h5py.File(path, "w") as out_h5:
        for instance_a, instance_b, has_structure, _src in sorted(set(memberships)):
            if not has_structure:
                continue
            pfam_a, _uni_a, seq_a = instance_info[instance_a]
            pfam_b, _uni_b, seq_b = instance_info[instance_b]
            rng = _key_rng("structure", instance_a, instance_b)
            gap = float(rng.uniform(5.0, 10.0))
            text = pdb_text([("A", seq_a, 0.0, rng), ("B", seq_b, gap, rng)])
            blob = gzip.compress(text.encode(), mtime=0)
            out_h5.require_group(f"{pfam_a}_{pfam_b}").create_dataset(
                f"{instance_a}__{instance_b}",
                data=np.frombuffer(blob, dtype="uint8"), compression=None,
            )
            n += 1
    print(f"  {path.name}: {n} complexes")


def write_metadata(out: Path, instance_info, memberships) -> None:
    """The two CSVs `--metadata` expects (bin/create_metadata.py).

    Real ones are produced outside both repos; these follow the column contract
    the code reads: `domain_metadata.csv` is one row per instance with numeric
    domain-level columns, `ddi_metadata.csv` one row per instance pair with
    numeric `af_*` columns (only for pairs with a structure).
    """
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "domain_metadata.csv", "w") as fh:
        fh.write("instance_id,pfam_id,uniprot_id,length,hydrophobic_fraction,interpro_n_sites\n")
        for instance_id in sorted(instance_info):
            pfam_id, uniprot_id, seq = instance_info[instance_id]
            hydrophobic = sum(aa in "AVILMFWY" for aa in seq) / len(seq)
            n_sites = int(_key_rng("interpro", instance_id).integers(0, 6))
            fh.write(f"{instance_id},{pfam_id},{uniprot_id},{len(seq)},"
                     f"{hydrophobic:.4f},{n_sites}\n")
    rows = 0
    with open(out / "ddi_metadata.csv", "w") as fh:
        fh.write("pfam_id_a,pfam_id_b,uniprot_id_a,uniprot_id_b,instance_id_a,"
                 "instance_id_b,source,af_iptm,af_pae_mean\n")
        for instance_a, instance_b, has_structure, source in sorted(set(memberships)):
            if not has_structure:
                continue
            pfam_a, uni_a, _ = instance_info[instance_a]
            pfam_b, uni_b, _ = instance_info[instance_b]
            rng = _key_rng("ddi_metadata", instance_a, instance_b)
            fh.write(f"{pfam_a},{pfam_b},{uni_a},{uni_b},{instance_a},{instance_b},"
                     f"\"{source or ''}\",{rng.uniform(0.2, 0.95):.4f},"
                     f"{rng.uniform(2.0, 25.0):.4f}\n")
            rows += 1
    print(f"  domain_metadata.csv: {len(instance_info)} rows, ddi_metadata.csv: {rows} rows")


def write_embedding_file(path: Path, model: str, instances) -> None:
    """One `<model>_domain_embeddings.h5`, in domainsplit's published layout.

    `h5[pfam_id][instance_key]` -> one mean-pooled fp16 vector, plus the root
    attributes the real exporter writes. Only `key_layout` is checked;
    `domainsplit_run` is carried for fidelity with domainsplit's own output.
    Written over the union of every split, because the real export runs once per
    run against the pruned master database.

    Vectors are drawn from an RNG seeded on `(model, pfam_id, instance_key)`
    so the file is byte-identical however the fixture is generated, and so two
    models never agree by accident. The key is digested with sha256 rather than
    `hash()`: str hashing is salted per interpreter, so `hash()` would make the
    fixture differ between generator runs.
    """
    if path.exists():
        path.unlink()

    domains = set()
    with h5py.File(path, "w") as out_h5:
        for pfam_id, instance_key in sorted(set(instances)):
            digest = hashlib.sha256(
                f"{model}/{pfam_id}/{instance_key}".encode()
            ).digest()
            rng = np.random.default_rng(SEED + int.from_bytes(digest[:8], "big"))
            vector = rng.standard_normal(EMBED_DIM).astype(np.float16)
            out_h5.create_dataset(f"{pfam_id}/{instance_key}", data=vector)
            domains.add(pfam_id)

        out_h5.attrs["model"] = model
        out_h5.attrs["pooling"] = "mean"
        out_h5.attrs["dim"] = EMBED_DIM
        out_h5.attrs["dtype"] = "float16"
        out_h5.attrs["key_layout"] = "{pfam_id}/{instance_id}"
        out_h5.attrs["n_domains"] = len(domains)
        out_h5.attrs["n_instances"] = len(set(instances))
        out_h5.attrs["domainsplit_run"] = FIXTURE_RUN_ID

    print(
        f"  {path.name}: {len(set(instances))} instances under {len(domains)} domains"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", default="tests/data/databases",
        help="Output directory for the databases/ tree (default: tests/data/databases)",
    )
    parser.add_argument(
        "--embeddings-out", default=None,
        help="Output directory for the published embedding files "
             "(default: an `embeddings` directory beside --out)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    instances = []
    instance_info = {}
    memberships = []
    offset = 0
    for dataset, splits in DATASETS.items():
        print(f"\n[{dataset}]")
        dataset_dir = out / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in splits:
            offset += 100
            split_instances, split_info, split_memberships = build_split(
                dataset_dir / f"{split}.sqlite3", dataset, split, offset
            )
            instances += split_instances
            instance_info.update(split_info)
            memberships += split_memberships

    embeddings_out = Path(args.embeddings_out) if args.embeddings_out else out.parent / "embeddings"
    embeddings_out.mkdir(parents=True, exist_ok=True)
    print("\n[embeddings]")
    for model in EMBEDDING_MODELS:
        write_embedding_file(
            embeddings_out / f"{model}_domain_embeddings.h5", model, instances
        )

    print("\n[structures]")
    structures_out = out.parent / "structures"
    structures_out.mkdir(parents=True, exist_ok=True)
    write_structure_file(structures_out / "structures.h5", instance_info, memberships)

    print("\n[metadata]")
    write_metadata(out.parent / "metadata", instance_info, memberships)

    print(f"\nWrote fixture to {out.parent}")


if __name__ == "__main__":
    main()
