#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure, calculate_sasa_structure_level
from features import embeddings


def calculate_bsa(struct):

    structA = struct[0]["A"]
    structB = struct[0]["B"]
    
    sasa_A = calculate_sasa_structure_level(structA)
    sasa_B = calculate_sasa_structure_level(structB)

    sasa_complex = calculate_sasa_structure_level(struct)

    bsa = (sasa_A + sasa_B) - sasa_complex
    # Normalize by the total SASA of the two domains to get a relative BSA
    if (sasa_A + sasa_B) > 0:
        bsa /= (sasa_A + sasa_B)
    
    return bsa



def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.
    
    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
                optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                    pair, grouped by ddi_id.
        struct_file: Readable HDF5 file holding the DDI structures, keyed
                        as embeddings.interaction_group_name /
                        interaction_dataset_name describe.
    """
    ddi_df = embeddings.load_structure_data(conn)

    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return

    n_written = 0
    for (ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        bsa = calculate_bsa(structure)
        feature_vector = np.array([bsa], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1
        
    print(f"bsa: wrote {n_written} entries")
        