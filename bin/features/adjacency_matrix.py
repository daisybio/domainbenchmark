#!/usr/bin/env python3
import h5py
import numpy as np
import sqlite3
import torch
from collections import defaultdict, OrderedDict

from .structure_utils import (
    ProteinProteinInteractionPrediction,
    bytes_to_pdb_structure,
    calculate_rsa_residue_level,
    normalize_weighted_adjacency,
)
from features import embeddings

device = torch.device('cpu')

# Shared across the whole extract_features() run -- see note above.
FINGERPRINT_DICT = defaultdict(lambda: len(FINGERPRINT_DICT))


def calculate_min_heavy_atom_distance(res1, res2):
    min_dist = float('inf')
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            dist = atom1 - atom2
            if dist < min_dist:
                min_dist = dist
    return min_dist


def define_shell(domain, rsa_threshold=0.2):# -> list[Any]:
    rsa_residue = calculate_rsa_residue_level(domain)
    residue_lookup = OrderedDict({
        (residue.get_resname(), residue.get_parent().get_id(), residue.get_id()): residue
        for residue in domain.get_residues()
    })
 
    shell_residues = [
        residue_lookup[key]
        for key, rsa in rsa_residue.items()
        if rsa is not None and rsa > rsa_threshold
    ]
    return shell_residues


def build_shell_adjacency_graph(shell_residues, distance_threshold=14.0, d0=4.0):
    shell_residues = list(shell_residues)  # fix iteration order for indices
    n = len(shell_residues)
    graph = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, res1 in enumerate(shell_residues):
        for j, res2 in enumerate(shell_residues):
            if i == j:
                graph[i][j] = 1.0
            if i < j:
                min_dist = calculate_min_heavy_atom_distance(res1, res2)
                if min_dist <= distance_threshold:
                    graph[i][j] = (2 * d0) / (d0 + max(d0, min_dist))
                    graph[j][i] = graph[i][j]
    return graph


def create_fingerprints(adjacency_matrix):
    """Same neighborhood-hashing scheme as struct2graph.create_fingerprints /
    structure_utils.create_node_fingerprints, but written against the
    module-level, SHARED FINGERPRINT_DICT so identical local neighborhoods
    get the same id everywhere in the run, not just within one call.
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


def embed_graph_mean_pool(fingerprint, adjacency, model):
    """Run the GNN message-passing layers only (no attention head, no
    self-pairing) on one chain's shell graph, and mean-pool the resulting
    node embeddings into a single dim-sized (20) vector.

    fingerprint: array-like of node fingerprint ids, shape (n_nodes,)
    adjacency:   normalized adjacency matrix, shape (n_nodes, n_nodes)
    model:       the ONE shared ProteinProteinInteractionPrediction instance
                 for this run (reused across every call, never rebuilt here)
    """
    x = model.embed_fingerprint(torch.LongTensor(fingerprint).to(device))
    A = torch.FloatTensor(adjacency).to(device)

    for layer in model.W_gnn:
        h = torch.relu(layer(x))
        x = torch.matmul(A, h)

    return x.mean(dim=0).detach().cpu().numpy()



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
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}") # pyright: ignore[reportAttributeAccessIssue]

        structA = structure[0]["A"]  # type: ignore[reportGeneralTypeIssues]
        structB = structure[0]["B"]  # type: ignore[reportGeneralTypeIssues]

        entries = []
        for chain in(structA, structB):
            shell_matrix = build_shell_adjacency_graph(define_shell(chain))
            fingerprints = create_fingerprints(shell_matrix)
            adjacency_norm = normalize_weighted_adjacency(shell_matrix)
            entries.append((fingerprints, adjacency_norm))

        features[(pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)] = entries



    n_fingerprint = len(FINGERPRINT_DICT) + 100
    model = ProteinProteinInteractionPrediction(n_fingerprint).to(device)
    model.eval()

    n_written = 0
    for key, entries in features.items():
        pfam_id_a, pfam_id_b, instance_id_a, instance_id_b = key
        (fp_a, adj_a), (fp_b, adj_b) = entries

        vec_a = embed_graph_mean_pool(fp_a, adj_a, model)
        vec_b = embed_graph_mean_pool(fp_b, adj_b, model)

        feature_vector = np.concatenate([vec_a, vec_b])
        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

    print(f"adjacency_matrix: wrote {n_written} entries")