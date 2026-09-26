#!/usr/bin/env python3
import h5py
import sqlite3
import numpy as np
from numpy._typing._array_like import NDArray
from features import embeddings

from .structure_utils import bytes_to_pdb_structure, calculate_sasa_residue_level, pool_residue_feature



def calculate_bsa_residue_level(struct):

    structA = struct[0]["A"]
    structB = struct[0]["B"]
 
    sasa_A_residue = calculate_sasa_residue_level(structA)
    sasa_B_residue = calculate_sasa_residue_level(structB)
    sasa_complex_residue = calculate_sasa_residue_level(struct)
 
    bsa_residue = {}

    for key, sasa_isolated in {**sasa_A_residue, **sasa_B_residue}.items():
        bsa_residue[key] = sasa_isolated - sasa_complex_residue.get(key, 0.0)
 
    return bsa_residue
 


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
    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):


        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}") # pyright: ignore[reportAttributeAccessIssue]
        

        bsa_residue = calculate_bsa_residue_level(structure)
        feature_vector = pool_residue_feature(np.array(list(bsa_residue.values()), dtype=np.float32))

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, 
                                            instance_id_a, instance_id_b, 
                                            feature_vector)
        n_written += 1

    print(f"bsa_residue: wrote {n_written} entries")
