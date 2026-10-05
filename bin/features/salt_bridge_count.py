#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3

from .structure_utils import bytes_to_pdb_structure
from features import embeddings


def salt_bridge_count(structure, threshold=5.5):
    structA = structure[0]["A"]
    structB = structure[0]["B"]
    # Count the number of salt bridges between two protein structures.
    # Salt bridges are typically defined as interactions between oppositely charged residues (e.g., Lys/Arg and Asp/Glu) within a certain distance threshold.
    residuesA = list(structA.get_residues())
    residuesB = list(structB.get_residues())
    count = 0
    for resA in residuesA:
        for resB in residuesB:
            if (resA.get_resname() in ['LYS', 'ARG'] and resB.get_resname() in ['ASP', 'GLU']) or \
               (resA.get_resname() in ['ASP', 'GLU'] and resB.get_resname() in ['LYS', 'ARG']):
                coord_resA = resA['CA'].get_coord() if 'CA' in resA else None
                coord_resB = resB['CA'].get_coord() if 'CA' in resB else None
                if coord_resA is not None and coord_resB is not None:
                    dist = np.linalg.norm(coord_resA - coord_resB)
                    if dist <= threshold:
                        count += 1
    return count




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
    n_skipped = 0
    for (ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b) in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]
        if len(structure[0]) < 2:
            print(f"Warning: Structure for ddi_id {ddi_id} has less than 2 chains. Skipping. (instance_id_a: {instance_id_a}, instance_id_b: {instance_id_b}, pfam_id_a: {pfam_id_a}, pfam_id_b: {pfam_id_b})")
            n_skipped += 1
            continue



        count = salt_bridge_count(structure, threshold=5.5)
        feature_vector = np.array([count], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1
        
    print(f"salt_bridge_count: wrote {n_written} entries, skipped {n_skipped} entries")
