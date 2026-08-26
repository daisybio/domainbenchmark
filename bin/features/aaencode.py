#!/usr/bin/env python3
import h5py
import pandas as pd
import sqlite3
from features import embeddings


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    domain_sequence_df = pd.read_sql(
        f"""
                SELECT domain_id, {embeddings.INSTANCE_KEY_SQL},
                       UPPER(domain_sequence) AS sequence
                FROM domain_protein_map;
            """,
        conn,
    )

    domain_sequence_df["encoding"] = domain_sequence_df["sequence"].apply(aaencode)
    domain_sequence_df["domain_id"] = domain_sequence_df["domain_id"].astype(str)
    domain_sequence_df["instance_key"] = domain_sequence_df["instance_key"].astype(str)

    for (
        domain_id,
        instance_key,
        domain_sequence,
        encoding,
    ) in domain_sequence_df.itertuples(index=False):
        embeddings.write_instance(out_file, domain_id, instance_key, encoding)

    print(domain_sequence_df.head())


FEATURE_NUMBER = 7
ENCODE_MAP = {
    **dict.fromkeys(["A", "G", "V"], "0"),
    "C": "1",
    **dict.fromkeys(["F", "I", "L", "P"], "2"),
    **dict.fromkeys(["M", "S", "T", "Y"], "3"),
    **dict.fromkeys(["W", "H", "N", "Q"], "4"),
    **dict.fromkeys(["K", "R"], "5"),
    **dict.fromkeys(["D", "E"], "6"),
}


def aaencode(seq: str):
    deepiii_encoded_sequence = deepiii_encode(seq.upper())
    sequence_vector = get_triad_freq(deepiii_encoded_sequence)
    normalized_vector = normalize_triad_freq(sequence_vector)
    return normalized_vector


def deepiii_encode(seq: str) -> str:
    """Encode sequence using DeepIII encoding."""
    return "".join(ENCODE_MAP.get(char, " ") for char in seq)


def get_triad_freq(seq: str, feature_number=FEATURE_NUMBER) -> dict:
    """Calculate triad frequencies in a sequence."""
    triad_freq = {
        (i, j, k): 0
        for i in range(feature_number)
        for j in range(feature_number)
        for k in range(feature_number)
    }
    for i in range(len(seq) - 2):
        if seq[i] != " " and seq[i + 1] != " " and seq[i + 2] != " ":
            triad_freq[(int(seq[i]), int(seq[i + 1]), int(seq[i + 2]))] += 1
    return triad_freq


def normalize_triad_freq(triad_freq: dict, feature_number=FEATURE_NUMBER) -> list:
    """Normalize triad frequencies and return as a list."""
    min_freq = min(triad_freq.values())
    max_freq = max(triad_freq.values())
    # Normalize the frequencies
    triad_freq_normalized = {
        k: (v - min_freq) / (max_freq) for k, v in triad_freq.items() if max_freq > 0
    }
    # Extract the normalized frequencies in correct order
    triad_freq_list = [
        triad_freq_normalized.get((i, j, k), 0)
        for i in range(feature_number)
        for j in range(feature_number)
        for k in range(feature_number)
    ]
    return triad_freq_list
