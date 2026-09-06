#!/usr/bin/env python3
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sqlite3
from features import embeddings


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
    domain_sequence_df = pd.read_sql(
        f"""
                SELECT {embeddings.domain_key_sql("d1")} AS domain_1_key,
                       {embeddings.instance_key_sql("dpm_1")} AS instance_1_key,
                       UPPER(dpm_1.domain_sequence) AS sequence_1,
                       {embeddings.domain_key_sql("d2")} AS domain_2_key,
                       {embeddings.instance_key_sql("dpm_2")} AS instance_2_key,
                       UPPER(dpm_2.domain_sequence) AS sequence_2
                FROM domain_domain_interaction
                JOIN domain_protein_map AS dpm_1 ON domain_id_a = dpm_1.domain_id
                JOIN domain_protein_map AS dpm_2 ON domain_id_b = dpm_2.domain_id
                JOIN domain AS d1 ON domain_id_a = d1.id
                JOIN domain AS d2 ON domain_id_b = d2.id;
            """,
        conn,
    )

    for (
        domain_key_1,
        instance_1,
        seq_1,
        domain_key_2,
        instance_2,
        seq_2,
    ) in domain_sequence_df.itertuples(index=False):
        encoding = protdcal_encode(seq_1, seq_2)

        def write_to_h5(pair_key, instance_key):
            # create a group for each domain pair and put the instance pair
            # underneath it
            if pair_key not in out_file:
                pair_group = out_file.create_group(pair_key)
            else:
                pair_group = out_file[pair_key]

            if instance_key in pair_group:
                print(f"Warning: Duplicate entry for {instance_key} in {pair_key}.")
            else:
                pair_group[instance_key] = encoding

        # write both directions
        write_to_h5(f"{domain_key_1}_{domain_key_2}", f"{instance_1}_{instance_2}")
        write_to_h5(f"{domain_key_2}_{domain_key_1}", f"{instance_2}_{instance_1}")


protdcal_table = pd.read_csv(Path(__file__).parent / "protdcal_table.csv", index_col=0)


def protdcal_encode(seq1, seq2):
    seq1 = seq1.replace("U", "C")
    seq2 = seq2.replace("U", "C")

    seq_list = [seq1, seq2, seq1 + seq2, seq2 + seq1]

    norm_dict = generate_matrix(seq_list, protdcal_table)

    average_matrix = np.zeros((4, len(protdcal_table.columns)), dtype=float)

    for idx, matrix in enumerate(norm_dict):
        if matrix is None or not matrix.size:
            print(f"No features found for sequence {seq1} or {seq2}. Skipping pair.")
            return None
        average_matrix[idx] = np.mean(matrix, axis=0)

    # Now do the aggregation as described in the paper
    combined_features = (
        average_matrix[2]
        + average_matrix[3]
        - 2 * average_matrix[0]
        - 2 * average_matrix[1]
    )
    return combined_features


def generate_matrix(seq_list, protdcal: pd.DataFrame) -> dict[str, np.ndarray]:
    """Generate input matrix for PROTDCAL features for a set of sequences."""
    normalized_matrix_list = []

    for seq in seq_list:
        # Generate the PROTDCAL feature matrix for the sequence
        matrix = generate_protdcal_feature_matrix(seq, protdcal)
        if not matrix.size:
            print(f"No features found for sequence {id}.")
            continue
        # Normalize the feature matrix
        normalized_matrix = normalize_protcal_matrix(matrix)
        normalized_matrix_list.append(normalized_matrix)

    return normalized_matrix_list


def generate_protdcal_feature_matrix(
    sequence: str, protdcal: pd.DataFrame
) -> np.ndarray:
    """Generate PROTDCAL matrix for a sequence."""
    slist = list(sequence)
    matrix = np.zeros((len(slist), len(protdcal.columns)), dtype=float)
    for i in range(len(slist)):
        aa = slist[i]
        if aa not in protdcal.index:
            print(f"Amino acid '{aa}' not found in protdcal table.")
        matrix[i] = protdcal.loc[aa].values
    return matrix


def normalize_protdcal_row(feature_list: list[float]) -> list[float]:
    """
    NORMALIZING PROTDCAL FEATURES USING E-State operator
    D_es = D_i - Σ_j≠i (D_j - D_i) / (j - i)^2
    """
    e_state = []
    num = len(feature_list)
    for i in range(num):
        feature_value = feature_list[i]
        sum_ = 0.0
        for j in range(num):
            if i != j:
                dij = j - i
                sum_ += (feature_list[j] - feature_value) / (dij**2)
        e_state.append(feature_value - sum_)
    return e_state


def normalize_protcal_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of the feature matrix using the E-State operator."""
    normalized_matrix = np.zeros_like(feature_matrix)
    for i, row in enumerate(feature_matrix):
        normalized_row = normalize_protdcal_row(row)
        normalized_matrix[i] = normalized_row
    return normalized_matrix
