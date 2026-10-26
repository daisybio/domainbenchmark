#!/usr/bin/env python3
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sqlite3
from features import embeddings


# The instance pairs `ddi_split_membership` assigns to this split -- the same
# pairs DDI_EXTRACTION's INSTANCE_QUERY writes to `<split>_instances.csv`, which
# is all the ML loader ever looks up. Each side's Pfam accession comes from its
# own `domain_protein_map` row, so the key is right whichever way round the
# membership table sorted the two instance ids. This used to be the full
# cross-product of every instance of domain_a with every instance of domain_b,
# per DDI: quadratic in family size, and almost all of it never read.
PAIR_QUERY = f"""
    SELECT DISTINCT
           {embeddings.domain_key_sql("d1")} AS domain_1_key,
           {embeddings.instance_key_sql("dpm_1")} AS instance_1_key,
           dpm_1.domain_sequence AS sequence_1,
           {embeddings.domain_key_sql("d2")} AS domain_2_key,
           {embeddings.instance_key_sql("dpm_2")} AS instance_2_key,
           dpm_2.domain_sequence AS sequence_2
    FROM ddi_split_membership AS m
    JOIN domain_protein_map AS dpm_1 ON dpm_1.instance_id = m.instance_id_a
    JOIN domain_protein_map AS dpm_2 ON dpm_2.instance_id = m.instance_id_b
    JOIN domain AS d1 ON dpm_1.domain_id = d1.id
    JOIN domain AS d2 ON dpm_2.domain_id = d2.id
    ORDER BY domain_1_key, instance_1_key, domain_2_key, instance_2_key
"""


def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int):
    # Deterministic encoder: `seed` is part of the encoder ABI so that a
    # sampling encoder cannot be added without one. Unused here.
    # Interaction encoding: the group is the domain *pair* and the subgroup the
    # instance *pair*. Keys are joined with "_" but are never split apart again
    # -- the ML loader looks them up whole, built from the same instance ids the
    # DDI instance CSV carries.
    #
    # The domain half of the pair key is the Pfam accession, matching the group
    # names every other encoder writes and the `domain_1`/`domain_2` columns of
    # DDI_EXTRACTION's CSVs. Pfam accessions carry no "_", so the joined key
    # stays unambiguous. See bin/features/embeddings.py for why not `domain.id`.
    has_membership = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ddi_split_membership'"
    ).fetchone()
    if not has_membership:
        # The loader cannot use an interaction encoding without
        # `<split>_instances.csv` either, so there is nothing to fall back to.
        raise ValueError(
            "protdcal needs ddi_split_membership: it encodes the instance pairs "
            "the split assigns, and this database names none."
        )

    pair_df = pd.read_sql(PAIR_QUERY, conn)

    # Per-instance means are reused by every pair the instance appears in.
    single_means = {}

    def single_mean(domain_key, instance_key, sequence):
        if instance_key not in single_means:
            cleaned = embeddings.clean_sequence(sequence, domain_key, instance_key)
            single_means[instance_key] = (cleaned, e_state_mean(cleaned))
        return single_means[instance_key]

    written = set()
    for (
        domain_key_1,
        instance_1,
        seq_1,
        domain_key_2,
        instance_2,
        seq_2,
    ) in pair_df.itertuples(index=False):
        domain_key_1, instance_1 = str(domain_key_1), str(instance_1)
        domain_key_2, instance_2 = str(domain_key_2), str(instance_2)
        clean_1, mean_1 = single_mean(domain_key_1, instance_1, seq_1)
        clean_2, mean_2 = single_mean(domain_key_2, instance_2, seq_2)
        encoding = protdcal_combine(clean_1, clean_2, mean_1, mean_2)

        # Both directions: the loader looks the pair up in whichever
        # orientation the DDI CSV names it. A homodimer pair can produce the
        # same key twice -- the vector is symmetric, so the first write stands.
        for key in (
            (f"{domain_key_1}_{domain_key_2}", f"{instance_1}_{instance_2}"),
            (f"{domain_key_2}_{domain_key_1}", f"{instance_2}_{instance_1}"),
        ):
            if key not in written:
                embeddings.write_instance(out_file, *key, encoding)
                written.add(key)

    print(f"protdcal: wrote {len(written)} instance-pair entries from {len(pair_df)} membership pairs")


protdcal_table = pd.read_csv(Path(__file__).parent / "protdcal_table.csv", index_col=0)
_PROPERTIES = protdcal_table.to_numpy(dtype=float)
_ROW_OF = {aa: row for row, aa in enumerate(protdcal_table.index)}


def protdcal_combine(seq1: str, seq2: str, mean_1: np.ndarray, mean_2: np.ndarray) -> np.ndarray:
    """Interaction vector for one pair of cleaned sequences.

    The four averages -- each sequence alone and both concatenations -- are
    combined as described in the paper:
    `avg(s1+s2) + avg(s2+s1) - 2 avg(s1) - 2 avg(s2)`.
    """
    return (
        e_state_mean(seq1 + seq2)
        + e_state_mean(seq2 + seq1)
        - 2 * mean_1
        - 2 * mean_2
    )


def e_state_mean(sequence: str) -> np.ndarray:
    """Mean over residues of the E-state-normalised property matrix."""
    return e_state(property_matrix(sequence)).mean(axis=0)


def property_matrix(sequence: str) -> np.ndarray:
    """`(len(sequence), n_properties)` ProtDCal property matrix.

    `sequence` must already be cleaned (`embeddings.clean_sequence`): the
    table holds the 20 standard residues only.
    """
    return _PROPERTIES[[_ROW_OF[aa] for aa in sequence]]


def e_state(matrix: np.ndarray) -> np.ndarray:
    """E-state operator along the residue axis, independently per property.

        E_i = D_i - sum_{j != i} (D_j - D_i) / (j - i)^2

    i and j are *residue positions*: each residue's property value is
    perturbed by every other residue's, weighted by inverse squared sequence
    distance. It used to run across the 16 property columns of one residue
    instead, mixing quantities in different units (a free energy minus a
    hydrophobicity) and never looking at a neighbour.

    With W the Toeplitz matrix of 1/(j-i)^2 (zero diagonal) and w its row sums,
    E = D - W D + w * D. W D is a convolution with that kernel, which keeps
    memory linear in sequence length instead of materialising W.

    Under the residue mean taken afterwards the correction cancels exactly --
    W is symmetric, so sum_i sum_j W_ij (D_j - D_i) = 0 -- and the mean of E
    equals the mean of D up to rounding. The operator is kept so the matrix is
    the one the method defines, should another aggregation be added.
    """
    n = len(matrix)
    offsets = np.arange(-(n - 1), n, dtype=float)
    kernel = np.zeros_like(offsets)
    nonzero = offsets != 0
    kernel[nonzero] = 1.0 / offsets[nonzero] ** 2
    # Full convolution, then the window where output index i lines up with
    # (W @ column)[i] = sum_j column[j] * kernel[(i - j) + n - 1].
    window = slice(n - 1, 2 * n - 1)
    weighted = np.stack(
        [np.convolve(column, kernel)[window] for column in matrix.T], axis=1
    )
    weight_sum = np.convolve(np.ones(n), kernel)[window]
    return matrix - weighted + weight_sum[:, None] * matrix
