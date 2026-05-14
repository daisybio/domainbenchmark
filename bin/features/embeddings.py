import h5py

import pandas as pd
import pickle


def write_embeddings_h5(out_file: h5py.File, embeddings_df: pd.DataFrame):
    # unpickle and select the embedding values between start_pos and end_pos
    # then take the mean of the embeddings to get per-domain embedding
    embeddings_df["embedding"] = embeddings_df["embedding"].apply(pickle.loads)

    # if start_pos and end_pos are in the dataframe, select the embedding values between start_pos and end_pos
    if "start_pos" in embeddings_df.columns and "end_pos" in embeddings_df.columns:
        embeddings_df["embedding"] = embeddings_df.apply(
            lambda row: row["embedding"][row["start_pos"] : row["end_pos"]], axis=1
        )
        embeddings_df["embedding"] = embeddings_df["embedding"].apply(
            lambda x: x.mean(axis=0)
        )

    embeddings_df["domain_id"] = embeddings_df["domain_id"].astype(str)

    # save to hdf5 file using h5py
    for _, row in embeddings_df.iterrows():
        out_file[f"{row['domain_id']}/{row['protein_id']}"] = row["embedding"]
