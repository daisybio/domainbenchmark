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

isoelectric_points = {
    'ALA': 6.0, 'ARG': 10.76, 'ASN': 5.41, 'ASP': 2.77,
    'CYS': 5.07, 'GLN': 5.65, 'GLU': 3.22, 'GLY': 5.97,
    'HIS': 7.59, 'ILE': 6.02, 'LEU': 5.98, 'LYS': 9.74,
    'MET': 5.74, 'PHE': 5.48, 'PRO': 6.30, 'SER': 5.68,
    'THR': 5.60, 'TRP': 5.89, 'TYR': 5.66, 'VAL': 5.98
}


def define_shell(domain, rsa_threshold=0.2):
    rsa_residue = calculate_rsa_residue_level(domain)
    shell_residues = OrderedDict()
    for (resname, chain_id, rid), rsa in rsa_residue.items():
        if rsa is not None and rsa > rsa_threshold:
            shell_residues[(resname, chain_id, rid)] = rsa
    return shell_residues


def build_shell_cci_graph(shell_residues):
    graph = [[0.0 for _ in shell_residues] for _ in shell_residues]
    for i, res1 in enumerate(shell_residues):
        for j, res2 in enumerate(shell_residues):
            if i <= j:
                # res1 and res2 are tuples of (resname, rid)
                pi1 = isoelectric_points.get(res1[0], 0)
                pi2 = isoelectric_points.get(res2[0], 0)
                cci = 11 - abs(((pi1 - 7) * (pi2 - 7)) * 19 / 33.8)
                graph[i][j] = cci
                graph[j][i] = cci  # Symmetric graph
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
            shell_matrix = build_shell_cci_graph(define_shell(chain))
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

    print(f"charge_map: wrote {n_written} entries")