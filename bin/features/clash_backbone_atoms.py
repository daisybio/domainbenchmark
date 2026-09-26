#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
from Bio.PDB.NeighborSearch import NeighborSearch

from .structure_utils import bytes_to_pdb_structure
from features import embeddings

vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52}
backbone_atoms = {'N', 'CA', 'C', 'O'}


def calculate_clash_backbone_atoms(structure, overlap_tolerance = 0.4):
    
    chain_a_atoms = [a for a in structure[0]['A'].get_atoms() if a.element.strip() in backbone_atoms]
    chain_b_atoms = [a for a in structure[0]['B'].get_atoms() if a.element.strip() in backbone_atoms]

    ns = NeighborSearch(chain_b_atoms)

    clashes = 0
        
    for atom_a in chain_a_atoms:
        r_a = vdw_radii.get(atom_a.element.strip(), 1.7)
        close_atoms = ns.search(atom_a.coord, r_a + max(vdw_radii.values()) - overlap_tolerance)
        for atom_b in close_atoms:
            r_b = vdw_radii.get(atom_b.element.strip(), 1.7)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportGeneralTypeIssues]
            dist = atom_a - atom_b
            if dist < (r_a + r_b - overlap_tolerance):
                clashes += 1

    return clashes



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
        clashes = calculate_clash_backbone_atoms(structure, 0.5)

        feature_vector = np.array([clashes], dtype=np.float32)

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1
        
    print(f"clash_backbone_atoms: wrote {n_written} entries")
