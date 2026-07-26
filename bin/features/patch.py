#!/usr/bin/env python3
"""Template for adding a new feature encoding to the benchmark pipeline.

Steps to add a new feature:
1. Copy this file to bin/features/<your_feature>.py
2. Implement extract_features() below
3. Add '<your_feature>' to params.machine_learning_features in nextflow.config
4. If your feature needs GPU or large memory, also add it to params.large_features

The pipeline auto-discovers features by name: extract_features.py calls
importlib.import_module(f"features.{feature_name}").extract_features(conn, out_file).

Database schema (domain_protein_map table):
    domain_id       TEXT    -- Pfam domain ID (e.g. PF00001)
    protein_id      TEXT    -- UniProt protein ID (e.g. P12345)
    domain_sequence TEXT    -- amino acid sequence of the domain
    start_pos       INT    -- domain start position in protein sequence
    end_pos         INT    -- domain end position in protein sequence
    (+ embedding columns like esm3_per_domain, esmc_per_residue, etc.)

    CREATE TABLE IF NOT EXISTS domain_structure (
        id         INTEGER PRIMARY KEY AUTOINCREMENT, 
        ddi_id INTEGER NOT NULL REFERENCES domain_domain_interaction(id),
        domain_id_a TEXT NOT NULL REFERENCES domain(id),
        domain_id_b TEXT NOT NULL REFERENCES domain(id),
        protein1 INTEGER NOT NULL REFERENCES protein(id),
        protein2 INTEGER NOT NULL REFERENCES protein(id),
        source     TEXT    NOT NULL,
        pdb_gz     BLOB    NOT NULL,
        UNIQUE (ddi_id, protein1, protein2, source)
    );

HDF5 output structure (required by downstream ML models):
    /<domain_id_a _ domain_id_b>/<protein_id_a protein_id_b> = numpy array of shape (feature_dim,)

Features are saved double, once for af, once for rf


"""

import h5py
import numpy as np
import pandas as pd
import sqlite3

from structure_utils import bytes_to_pdb_structure


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [1] * size
        self.source = list(range(size))  # Track the source domain for each residue (0 for structA, 1 for structB)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1


def build_interchain_patches(structA, structB, threshold):
    all_residues = list(structA.get_residues()) + list(structB.get_residues())
    uf = UnionFind(len(all_residues))
    # Mark source for each residue by its index in `all_residues` (0..n-1).
    structA_count = len(list(structA.get_residues()))
    for idx in range(len(all_residues)):
        uf.source[idx] = 0 if idx < structA_count else 1

    # Build Union Find strucutre for all possible contacts within & between domains
    for i, res in enumerate(all_residues):
        for j in range(i + 1, len(all_residues)):
            coord_i = res['CA'].get_coord() if 'CA' in res else None
            coord_j = all_residues[j]['CA'].get_coord() if 'CA' in all_residues[j] else None
            if coord_i is not None and coord_j is not None:
                dist = np.linalg.norm(coord_i - coord_j)
                if dist <= threshold:
                    uf.union(i, j)

    return uf


def calculate_largest_patch(uf, all_residues):
    patch_sizes = {}
    for i in range(len(all_residues)):
        root = uf.find(i)
        if root not in patch_sizes:
            patch_sizes[root] = {'size': 0, 'source': set()}
        patch_sizes[root]['size'] += 1
        patch_sizes[root]['source'].add(uf.source[i])

    largest_patch_both = max((size['size'] for size in patch_sizes.values() if size['source'] == {0, 1}), default=0)

    return largest_patch_both



def extract_features(conn: sqlite3.Connection, out_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                  pair, grouped by ddi_id.
    """
    domain_structure = pd.read_sql(
        """
        SELECT domain_id_a, domain_id_b, protein_id_a, protein_id_b, source, pdb_gz
        FROM domain_structure;
        """,
        conn,
    )


    domain_structure["domain_id_a"] = domain_structure["domain_id_a"].astype(str)
    domain_structure["domain_id_b"] = domain_structure["domain_id_b"].astype(str)
    domain_structure["protein_id_a"] = domain_structure["protein_id_a"].astype(str)
    domain_structure["protein_id_b"] = domain_structure["protein_id_b"].astype(str)


    for domain_id_a, domain_id_b, protein_id_a, protein_id_b, source, pdb_gz in domain_structure.itertuples(index=False):

        feature_vector = np.zeros(1, dtype=np.float32)  # 20 amino acids

        structure = bytes_to_pdb_structure(pdb_gz)

        structA = structure[0]["A"]
        structB = structure[0]["B"]

        uf = build_interchain_patches(structA, structB, threshold=5.0)
        largest_patch = calculate_largest_patch(uf, list(structA.get_residues()) + list(structB.get_residues()))
        feature_vector = np.array([largest_patch], dtype=np.float32)

        def write_to_h5(domain_key, protein_key):
            # create a group for each pfam_id and put uniprot_id as a subgroup
            if domain_key not in out_file:
                pfam_group = out_file.create_group(domain_key)
            else:
                pfam_group = out_file[domain_key]

            if protein_key in pfam_group:
                print(f"Warning: Duplicate entry for {protein_key} in {domain_key}.")
            else:
                pfam_group[protein_key] = feature_vector  # pyright: ignore[reportIndexIssue]

        # write both directions
        write_to_h5(f"{domain_id_a}_{domain_id_b}", f"{protein_id_a}_{protein_id_b}")
        write_to_h5(f"{domain_id_b}_{domain_id_a}", f"{protein_id_b}_{protein_id_a}")

        

    print(f"patch: wrote {len(domain_structure)} entries")
