#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


DISULPHIDE_CUTOFFSQ = 5.0625

def identify_ss_bonds(domain):
    # Identify disulfide bonds in the domain structure
    # If the distance between the SG atoms of two cysteine residues is less than 2.25 Å, we consider it a disulfide bond.
    ss_bonds = []
    cysteines = [residue for residue in domain.get_residues() if residue.get_resname() == 'CYS']
    for i, res1 in enumerate(cysteines):
        for j, res2 in enumerate(cysteines):
            if i < j:  # Avoid double counting
                sg1 = res1['SG']
                sg2 = res2['SG']
                distance_sq = (sg1 - sg2) ** 2
                if distance_sq < DISULPHIDE_CUTOFFSQ:
                    ss_bonds.append((res1.get_id(), res2.get_id()))
    return ss_bonds


def count_disulfide_bonds(domain):
    ss_bonds = identify_ss_bonds(domain)
    return len(ss_bonds)



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

        ssbond_count = count_disulfide_bonds(structure)
        feature_vector = np.array([ssbond_count], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"disulfide_bond_count: wrote {n_written} entries")