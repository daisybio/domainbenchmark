#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
import torch
from collections import defaultdict

from .structure_utils import (
    ProteinProteinInteractionPrediction,
    bytes_to_pdb_structure,
)
from features import embeddings

device = torch.device('cpu')

# Shared across the whole extract_features() run -- see note above. Kept
# separate from charge_map.py's FINGERPRINT_DICT: joint-graph fingerprints
# are a different vocabulary from intra-chain-only shell fingerprints.
FINGERPRINT_DICT = defaultdict(lambda: len(FINGERPRINT_DICT))

CONTACT_DISTANCE_THRESHOLD = 5.0  # Angstrom, CA-CA


def get_residue_ca_coords(chain):
    """One row per residue with a CA atom, in chain order.

    Returns:
        residues: list of Bio.PDB residues (same order as coords)
        coords:   np.ndarray, shape (n_residues, 3)
    """
    residues = []
    coords = []
    for residue in chain:
        if 'CA' in residue:
            residues.append(residue)
            coords.append(residue['CA'].get_coord())
    return residues, np.array(coords, dtype=float)


def build_joint_contact_graph(coords_a, coords_b, distance_threshold=CONTACT_DISTANCE_THRESHOLD):
    """Binary residue-level contact graph over the WHOLE complex: chain A's
    residues, then chain B's residues, as one set of nodes. An edge exists
    (weight 1.0) wherever two residues' CA atoms are within
    distance_threshold, regardless of whether they're on the same chain or
    across the interface.

    Returns:
        graph:        np.ndarray, shape (n_a + n_b, n_a + n_b)
        chain_labels: list of 'A'/'B', length n_a + n_b, aligned to graph rows
    """
    n_a = coords_a.shape[0]
    n_b = coords_b.shape[0]
    n = n_a + n_b

    all_coords = np.vstack([coords_a, coords_b]) if n_a and n_b else (
        coords_a if n_b == 0 else coords_b
    )

    graph = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(all_coords[i] - all_coords[j])
            if dist <= distance_threshold:
                graph[i, j] = 1.0
                graph[j, i] = 1.0

    chain_labels = ['A'] * n_a + ['B'] * n_b
    return graph, chain_labels


def create_fingerprints(adjacency_matrix):
    """Same neighborhood-hashing scheme as charge_map.py/struct2graph.py,
    written against this module's own SHARED FINGERPRINT_DICT. Works
    unchanged on a binary (0/1) adjacency matrix -- the neighbor signature is
    just less granular (all contact weights equal 1.0) than a weighted graph.
    """
    adjacency = np.array(adjacency_matrix, dtype=float)
    n = adjacency.shape[0]

    fingerprints = []
    for i in range(n):
        neighbors_idx = np.where(adjacency[i] > 0.0001)[0]
        neighbor_weights = adjacency[i][neighbors_idx]

        if len(neighbors_idx) > 0:
            sorted_idx = np.argsort(-neighbor_weights)
            neighbors_sorted = neighbors_idx[sorted_idx]
            weights_sorted = neighbor_weights[sorted_idx]
            neighbor_sig = tuple(
                (int(idx), round(float(w), 4)) for idx, w in zip(neighbors_sorted, weights_sorted)
            )
            fingerprint = (i, neighbor_sig)
        else:
            fingerprint = (i,)

        fingerprints.append(FINGERPRINT_DICT[fingerprint])

    return np.array(fingerprints)


def embed_joint_graph_pool_by_chain(fingerprint, adjacency, chain_labels, model):
    """Run the GNN message-passing layers (model.W_gnn) over the JOINT
    adjacency -- so a residue's embedding is informed by its neighbors
    regardless of which chain they're on -- then split the resulting node
    embeddings by chain_labels and mean-pool each chain separately.

    Returns:
        vec_a, vec_b: np.ndarray, each shape (dim,) -- dim(=20) zero-vector
                      if that chain has no residues in this instance.
    """
    x = model.embed_fingerprint(torch.LongTensor(fingerprint).to(device))
    A = torch.FloatTensor(adjacency).to(device)

    for layer in model.W_gnn:
        h = torch.relu(layer(x))
        x = torch.matmul(A, h)

    x = x.detach().cpu().numpy()
    chain_labels = np.array(chain_labels)
    dim = x.shape[1] if x.size else 0

    mask_a = chain_labels == 'A'
    mask_b = chain_labels == 'B'
    vec_a = x[mask_a].mean(axis=0) if mask_a.any() else np.zeros(dim, dtype=np.float32)
    vec_b = x[mask_b].mean(axis=0) if mask_b.any() else np.zeros(dim, dtype=np.float32)

    return vec_a, vec_b



def extract_features(conn: sqlite3.Connection, out_file: h5py.File, seed: int, struct_file: h5py.File):
    """Extract features from the database and write them to the HDF5 file.

    Args:
        conn: SQLite connection to one of train.sqlite3 / test.sqlite3 /
              optimization.sqlite3. Read-only — do not write.
        out_file: Writable HDF5 file. Write one dataset per (ddi, ppi)
                  pair, grouped by ddi_id.
        seed: unused here, kept for signature parity with struct2graph.py.
        struct_file: Readable HDF5 file holding the DDI structures, keyed
                     as embeddings.interaction_group_name /
                     interaction_dataset_name describe.
    """
    ddi_df = embeddings.load_structure_data(conn)

    if ddi_df.empty:
        print("Warning: No entries found in ddi_split_membership. Skipping feature extraction.")
        return

    features = {} 
    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):
        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue

        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}")  # pyright: ignore[reportAttributeAccessIssue]

        structA = structure[0]["A"]  # type: ignore[reportGeneralTypeIssues]
        structB = structure[0]["B"]  # type: ignore[reportGeneralTypeIssues]

        _, coords_a = get_residue_ca_coords(structA)
        _, coords_b = get_residue_ca_coords(structB)

        if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
            print(f"Warning: Missing CA coordinates for ddi {ddi_id}. Skipping.")
            continue

        joint_graph, chain_labels = build_joint_contact_graph(coords_a, coords_b)
        fingerprints = create_fingerprints(joint_graph)

        features[(pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)] = (fingerprints, joint_graph, chain_labels)

    # ---- Pass 2: FINGERPRINT_DICT is now final -- size and build ONE model,
    # then reuse it for every embedding call. ----
    n_fingerprint = len(FINGERPRINT_DICT) + 100
    model = ProteinProteinInteractionPrediction(n_fingerprint).to(device)
    model.eval()

    n_written = 0
    for key, (fingerprints, joint_graph, chain_labels) in features.items():
        pfam_id_a, pfam_id_b, instance_id_a, instance_id_b = key

        vec_a, vec_b = embed_joint_graph_pool_by_chain(fingerprints, joint_graph, chain_labels, model)

        # A/B-distinguishing combination by default -- swap for e.g.
        # np.mean([vec_a, vec_b], axis=0) if the feature should be symmetric.
        feature_vector = np.concatenate([vec_a, vec_b])

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"contact_map: wrote {n_written} entries")
