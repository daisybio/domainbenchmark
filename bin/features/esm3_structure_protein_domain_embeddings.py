#!/usr/bin/env python3
import h5py
import pandas as pd
import sqlite3
from features import embeddings


def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    # export domain embeddings as hdf using h5py
    embeddings_chunks = pd.read_sql(
        """
               SELECT domain_id, protein_id, start_pos, end_pos, esm3_structure_per_residue as embedding
               FROM domain_protein_map, protein
               WHERE protein_id = protein.id AND
                   embedding IS NOT NULL;
           """,
        conn,
        chunksize=5000,
    )

    for embeddings_df in embeddings_chunks:
        embeddings.write_embeddings_h5(out_file, embeddings_df)
