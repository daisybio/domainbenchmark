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


hydrophobic_moments = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 4.5, 'LYS': -3.9, 
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 
    'THR': -0.7, 'TRP': 1.6, 'TYR': -1.3, 'VAL': 4.2
}


def define_shell(domain, rsa_threshold=0.2):
    rsa_residue = calculate_rsa_residue_level(domain)
    shell_residues = OrderedDict()
    for (resname, chain_id, rid), rsa in rsa_residue.items():
        if rsa is not None and rsa > rsa_threshold:
            shell_residues[(resname, chain_id, rid)] = rsa
    return shell_residues

def build_shell_hci_graph(shell_residues):
    graph = [[0.0 for _ in shell_residues] for _ in shell_residues]
    for i, res1 in enumerate(shell_residues):
        for j, res2 in enumerate(shell_residues):
            if i <= j:
                hm1 = hydrophobic_moments.get(res1[0], 0)
                hm2 = hydrophobic_moments.get(res2[0], 0)
                hci = 20 - abs((hm1 - hm2) * 19 / 10.6)
                graph[i][j] = hci
                graph[j][i] = hci  # Symmetric graph
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
            shell_matrix = build_shell_hci_graph(define_shell(chain))
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

    print(f"hydrophobic_map: wrote {n_written} entries")