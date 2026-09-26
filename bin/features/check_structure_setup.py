#!/usr/bin/env python3

import sys

import sqlite3

import h5py
 
import embeddings
from structure_utils import bytes_to_pdb_structure
 
 
def check_residue_id_collision(struct, label=""):

    model = struct[0]
 
    if "A" not in model or "B" not in model:

        print(f"[{label}] Missing chain A or B. Chains found: {[c.id for c in model]}")

        return
 
    ids_A = {res.get_id() for res in model["A"].get_residues()}

    ids_B = {res.get_id() for res in model["B"].get_residues()}

    overlap = ids_A & ids_B
 
    print(f"[{label}] Chain A residues: {len(ids_A)}")

    print(f"[{label}] Chain B residues: {len(ids_B)}")

    print(f"[{label}] Overlapping residue IDs (A ∩ B): {len(overlap)}")
 
    if overlap:

        print(f"[{label}] ⚠️  Collision confirmed:")

        for rid in sorted(overlap):

            print(f"   {rid}")

    else:

        print(f"[{label}] ✅ No collision — residue IDs unique across chains.")
 
 
def main(db_path, struct_path, n_check=1):

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    struct_file = h5py.File(struct_path, "r")
 
    ddi_df = embeddings.load_structure_data(conn)

    if ddi_df.empty:

        print("No entries found in ddi_split_membership.")

        return
 
    checked = 0

    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):

        pdb_gz = embeddings.read_interaction_instance(

            struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b

        )

        if pdb_gz is None:

            continue
 
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")

        check_residue_id_collision(structure, label=f"ddi_{ddi_id}")
 
        checked += 1

        if checked >= n_check:

            break
 
    if checked == 0:

        print("No structures with available data found to check.")
 
    conn.close()

    struct_file.close()
 
 
if __name__ == "__main__":

    if len(sys.argv) < 3:

        print("Usage: python check_residue_ids.py <db_path.sqlite3> <struct_file.h5> [n_check]")

        sys.exit(1)
 
    db_path = sys.argv[1]

    struct_path = sys.argv[2]

    n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    main(db_path, struct_path, n_check)
 