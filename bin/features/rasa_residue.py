#!/usr/bin/env python3
import h5py
import sqlite3
import numpy as np
from features import embeddings

from .structure_utils import bytes_to_pdb_structure, calculate_sasa_residue_level, pool_residue_feature


def calculate_rasa_residue_level(domain):
    sasa_residue = calculate_sasa_residue_level(domain)

    # MAxSASA values by Tien et al. 2013, "Maximum allowed solvent accessibilities of residues in proteins" (https://doi.org/10.1002/prot.24286)
    max_sasa_values = {
        'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
        'GLN': 223.0, 'GLU': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
        'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
        'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
    }

    rsa_residue = np.zeros(len(list(domain.get_residues())), dtype=np.float32)
    for i, residue in enumerate(domain.get_residues()):
        resname = residue.get_resname()
        rid = residue.get_id()
        chain_id = residue.get_parent().get_id()
        key = (chain_id, rid)
        sasa_value = sasa_residue.get(key, 0)
        max_sasa = max_sasa_values.get(resname, None)
        if max_sasa is not None and max_sasa > 0:
            rsa_residue[i] = sasa_value / max_sasa


    return rsa_residue



def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File, complex: bool = False):
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

        # Initialize empty vector of length 0
        feature_vector = np.zeros(0, dtype=np.float32)
        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}") # pyright: ignore[reportAttributeAccessIssue]

        if complex:
            rsa_residue = calculate_rasa_residue_level(structure)
            feature_vector = pool_residue_feature(np.array(rsa_residue, dtype=np.float32))
        else:
            rsa_residue_A = calculate_rasa_residue_level(structure[0]["A"]) # pyright: ignore[reportOptionalSubscript]
            rsa_residue_B = calculate_rasa_residue_level(structure[0]["B"]) # pyright: ignore[reportOptionalSubscript]
            feature_vector_A = np.array(rsa_residue_A, dtype=np.float32)
            feature_vector_B = np.array(rsa_residue_B, dtype=np.float32)
            feature_vector = pool_residue_feature(np.concatenate([feature_vector_A, feature_vector_B]))
        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b,
                                            instance_id_a, instance_id_b,
                                            feature_vector)
        n_written += 1
        

    print(f"rasa_residue: wrote {n_written} entries")