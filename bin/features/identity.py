#!/usr/bin/env python3
"""Family-identity encoder: a one-hot slot per Pfam family, pair = sum (W-15).

Measures what family identity alone explains. The vocabulary is the sorted set
of Pfam accessions of the instances `ddi_split_membership` assigns in the
database's train and validation splits (`--vocabulary_db`, staged explicitly by
FEATURE_EXTRACTION_ONE so both files are part of the task hash), indexed by
position. Every split, test included, is encoded against that one vocabulary:

* both sides known  -> 1 at each side's slot (a same-family pair -> 2.0),
* one side unseen   -> a single 1,
* both sides unseen -> all zeros.

On a family-disjoint test split (`minimal_leakage`) every test vector is zero,
so an identity model there cannot rank above chance: `n_pairs_zero == n_pairs`
in the file attrs is the exact check, the model's AUC only a reported one
(constant input does not guarantee bitwise-equal backend output).

Per instance *pair*, like protdcal (interaction layout, both orientations,
`attrs["layout"] = "interaction"`), so the vector is symmetric. Deterministic:
`seed` is part of the encoder ABI and unused.

The attrs bind the file to its vocabulary: `vocabulary_sha1` (sha1 of the
accessions joined by "\\n") is checked across splits by the ML loader and
recorded in `model_parameters.json` as `feature_digests`, so a test file built
from another vocabulary of the same width fails instead of scoring garbage.
"""
import hashlib
import sqlite3

import h5py
import numpy as np
import pandas as pd
from features import embeddings

#: The splits the vocabulary is built from; extract_features.py requires one
#: `--vocabulary_db` per entry, named `<split>.sqlite3`.
VOCABULARY_SPLITS = ("train", "validation")

PAIR_QUERY = f"""
    SELECT DISTINCT
           {embeddings.domain_key_sql("d1")} AS domain_1_key,
           {embeddings.instance_key_sql("dpm_1")} AS instance_1_key,
           {embeddings.domain_key_sql("d2")} AS domain_2_key,
           {embeddings.instance_key_sql("dpm_2")} AS instance_2_key
    FROM ddi_split_membership AS m
    JOIN domain_protein_map AS dpm_1 ON dpm_1.instance_id = m.instance_id_a
    JOIN domain_protein_map AS dpm_2 ON dpm_2.instance_id = m.instance_id_b
    JOIN domain AS d1 ON dpm_1.domain_id = d1.id
    JOIN domain AS d2 ON dpm_2.domain_id = d2.id
    ORDER BY domain_1_key, instance_1_key, domain_2_key, instance_2_key
"""

FAMILY_QUERY = f"""
    SELECT DISTINCT {embeddings.domain_key_sql("d")} AS domain_key
    FROM (SELECT instance_id_a AS iid FROM ddi_split_membership
          UNION SELECT instance_id_b FROM ddi_split_membership) AS m
    JOIN domain_protein_map AS dpm ON dpm.instance_id = m.iid
    JOIN domain AS d ON dpm.domain_id = d.id
"""


def _require_membership(conn, what):
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ddi_split_membership'"
    ).fetchone():
        raise ValueError(f"identity needs ddi_split_membership in {what}: it encodes the instance pairs the split assigns.")


def build_vocabulary(vocabulary_dbs):
    """Sorted Pfam accessions of the membership instances of every database."""
    families = set()
    for path in vocabulary_dbs:
        # Read-only: a wrong path must fail, not create an empty database.
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            _require_membership(conn, path)
            for (key,) in conn.execute(FAMILY_QUERY):
                if key is None:
                    raise ValueError(f"identity: {path} has a membership instance with a NULL pfam_id")
                families.add(str(key))
    return sorted(families)


def vocabulary_sha1(vocab):
    return hashlib.sha1("\n".join(vocab).encode()).hexdigest()


def pair_vector(index, a, b):
    """+1 at each known side's slot; a self-pair gets 2.0, unseen sides nothing."""
    v = np.zeros(len(index), dtype=np.float32)
    for side in (a, b):
        if side in index:
            v[index[side]] += 1
    return v


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, vocabulary_dbs=()):
    if not vocabulary_dbs:
        raise ValueError("identity needs --vocabulary_db (train and validation).")
    _require_membership(conn, "the split database")
    vocab = build_vocabulary(vocabulary_dbs)
    if not vocab:
        raise ValueError("identity: the vocabulary databases name no family.")
    index = {f: i for i, f in enumerate(vocab)}
    sha1 = vocabulary_sha1(vocab)

    pair_df = pd.read_sql(PAIR_QUERY, conn)
    counts = {"both_known": 0, "one_unseen": 0, "zero": 0}
    written = set()
    for d1, i1, d2, i2 in pair_df.itertuples(index=False):
        d1, i1, d2, i2 = str(d1), str(i1), str(d2), str(i2)
        known = (d1 in index) + (d2 in index)
        counts[("zero", "one_unseen", "both_known")[known]] += 1
        vec = pair_vector(index, d1, d2)
        # Both directions, once per key (a homodimer pair can repeat one).
        for key in ((f"{d1}_{d2}", f"{i1}_{i2}"), (f"{d2}_{d1}", f"{i2}_{i1}")):
            if key not in written:
                embeddings.write_instance(out_file, *key, vec, compression="gzip")
                written.add(key)

    out_file.attrs["layout"] = "interaction"
    out_file.attrs["vocabulary_size"] = len(vocab)
    out_file.attrs["vocabulary_sha1"] = sha1
    out_file.attrs["vocabulary_splits"] = ",".join(VOCABULARY_SPLITS)
    out_file.attrs["n_pairs"] = len(pair_df)
    out_file.attrs["n_pairs_both_known"] = counts["both_known"]
    out_file.attrs["n_pairs_one_unseen"] = counts["one_unseen"]
    out_file.attrs["n_pairs_zero"] = counts["zero"]
    print(
        f"identity: V={len(vocab)} sha1={sha1} first={vocab[0]} last={vocab[-1]}; "
        f"{len(pair_df)} membership pairs (both known {counts['both_known']}, "
        f"one unseen {counts['one_unseen']}, zero {counts['zero']}); "
        f"{len(written)} oriented entries"
    )
