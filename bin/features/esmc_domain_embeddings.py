#!/usr/bin/env python3
import h5py
import pandas as pd
import sqlite3
from features import embeddings


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    # export domain embeddings as hdf using h5py
    embeddings_df = pd.read_sql(
        """
               SELECT domain_id, protein_id, esmc_per_domain as embedding
               FROM domain_protein_map
               WHERE embedding IS NOT NULL;
           """,
        conn,
    )

    embeddings.write_embeddings_h5(out_file, embeddings_df)
