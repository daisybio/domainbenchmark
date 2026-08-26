import h5py

import pandas as pd
import pickle

# HDF5 layout is `h5[domain_id][instance_key]`. The key is the *instance*, not
# the protein: `domain_protein_map` is unique on
# (domain_id, protein_id, start_pos, end_pos), so a protein carrying two copies
# of one family yields two rows that would otherwise collide on protein_id.
# `instance_id` is domainsplit's own opaque identifier -- never parse it. The
# rowid fallback only has to be unique within the file the .h5 is built from,
# which is the same file the DDI CSVs come from.
def instance_key_sql(alias: str = "domain_protein_map") -> str:
    """The instance-key expression for one `domain_protein_map` alias."""
    return f"COALESCE({alias}.instance_id, 'r' || {alias}.rowid)"


INSTANCE_KEY_SQL = f"{instance_key_sql()} AS instance_key"


def write_instance(out_file: h5py.File, domain_id: str, instance_key: str, value):
    """Write one instance vector under its domain group, creating it if needed."""
    if domain_id not in out_file:
        domain_group = out_file.create_group(domain_id)
    else:
        domain_group = out_file[domain_id]

    domain_group[instance_key] = value


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
    embeddings_df["instance_key"] = embeddings_df["instance_key"].astype(str)

    # save to hdf5 file using h5py
    for _, row in embeddings_df.iterrows():
        out_file[f"{row['domain_id']}/{row['instance_key']}"] = row["embedding"]
