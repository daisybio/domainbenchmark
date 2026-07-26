import numpy as np
from Bio.PDB.PDBParser import PDBParser
from collections import defaultdict
import gzip

from Bio.PDB.SASA import ShrakeRupley


def calculate_sasa_residue_level(domain):
    sr = ShrakeRupley()
    sr.compute(domain, level="R")  # Compute SASA at the residue level
    sasa_values = {}
    for residue in domain.get_residues():
        sasa_values[residue.get_id()] = residue.sasa
    return sasa_values


def calculate_rsa_residue_level(domain):
    sasa_residue = calculate_sasa_residue_level(domain)

    # MAxSASA values by Tien et al. 2013, "Maximum allowed solvent accessibilities of residues in proteins" (https://doi.org/10.1002/prot.24286)
    max_sasa_values = {
        'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
        'GLN': 223.0, 'GLU': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
        'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
        'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
    }

    rsa_residue = {}
    for residue in domain.get_residues():
        resname = residue.get_resname()
        rid = residue.get_id()
        sasa_value = sasa_residue.get(rid, 0)
        max_sasa = max_sasa_values.get(resname, None)
        if max_sasa is not None and max_sasa > 0:
            rsa_residue[rid] = sasa_value / max_sasa
        else:
            rsa_residue[rid] = None

    return rsa_residue


def read_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    return structure


def build_residue_graph(domain, distance_threshold=5.0):
    # Build a graph where nodes are residues
    # Edges exist if residues are within a certain distance threshold (e.g., 5 Å)
    # Return 2D distance matrix
    residues = list(domain.get_residues())
    graph = [[0.0 for _ in residues] for _ in residues]
    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues):
            if i < j:  # Avoid double counting and self-comparison
                dist = float('inf')
                coord_res1 = res1['CA'].get_coord() if 'CA' in res1 else None
                coord_res2 = res2['CA'].get_coord() if 'CA' in res2 else None
                if coord_res1 is not None and coord_res2 is not None:
                    dist = np.linalg.norm(coord_res1 - coord_res2)
                if dist <= distance_threshold:
                    graph[i][j] = 1

    return graph


def build_adj_residue_graph(domain, distance_threshold=5.0):
    # Same as build_residue_graph but with distance values instead of binary edges
    residues = list(domain.get_residues())
    graph = [[float('inf') for _ in residues] for _ in residues]
    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues):
            if i < j:  # Avoid double counting and self-comparison

                coord_res1 = res1['CA'].get_coord() if 'CA' in res1 else None
                coord_res2 = res2['CA'].get_coord() if 'CA' in res2 else None
                if coord_res1 is not None and coord_res2 is not None:
                    dist = np.linalg.norm(coord_res1 - coord_res2)
                    if dist <= distance_threshold:
                        graph[i][j] = float(dist)
                        graph[j][i] = float(dist)  # Symmetric graph
    return graph


def build_interchain_adjacency_graph(domain1, domain2, distance_threshold=5.0):
    # Same as build_residue_graph but with distance values instead of binary edges
    residues1 = list(domain1.get_residues())
    residues2 = list(domain2.get_residues())
    graph = [[float('inf') for _ in residues2] for _ in residues1]
    for i, res1 in enumerate(residues1):
        for j, res2 in enumerate(residues2):
            coord_res1 = res1['CA'].get_coord() if 'CA' in res1 else None
            coord_res2 = res2['CA'].get_coord() if 'CA' in res2 else None
            if coord_res1 is not None and coord_res2 is not None:
                dist = np.linalg.norm(coord_res1 - coord_res2)
                if dist <= distance_threshold:
                    graph[i][j] = float(dist)
    return graph


def normalize_weighted_adjacency(adjacency_matrix):
    """
    Normalize weighted adjacency matrix using degree matrix normalization.
    Formula: A_norm = D^(-1/2) * A * D^(-1/2)
    where D is the degree matrix (sum of weights per node)
    """
    adjacency = np.array(adjacency_matrix, dtype=float)
    n = adjacency.shape[0]
    
    # Calculate degree (sum of weights for each node)
    degree = np.sum(adjacency, axis=1)
    
    # Avoid division by zero
    degree = np.where(degree == 0, 1, degree)
    d_half = np.sqrt(degree)
    d_half_inv = 1.0 / d_half
    d_half_inv = np.diag(d_half_inv)
    
    # Apply normalization: D^(-1/2) * A * D^(-1/2)
    normalized = np.matmul(d_half_inv, np.matmul(adjacency, d_half_inv))
    
    return normalized


def create_node_fingerprints(adjacency_matrix, node_features=None, radius=1):
    """
    Create r-radius fingerprints for each node in a weighted graph.
    Similar to Weisfeiler-Lehman algorithm but for weighted graphs.
    
    Args:
        adjacency_matrix: weighted adjacency matrix (n x n)
        node_features: optional node feature vector (n,) or (n x f)
        radius: neighborhood radius (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
    
    Returns:
        fingerprints: array of fingerprint IDs for each node
    """
    adjacency = np.array(adjacency_matrix, dtype=float)
    n = adjacency.shape[0]
    
    # Initialize fingerprint dictionary for hashing
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    
    # Start with node features or node indices
    if node_features is None:
        current_features = np.arange(n)
    else:
        current_features = np.array(node_features)
    
    fingerprints = []
    
    # For each node, create a fingerprint based on neighborhood
    for i in range(n):
        # Get neighbors (non-zero adjacency)
        neighbors_idx = np.where(adjacency[i] > 0.0001)[0]
        neighbor_weights = adjacency[i][neighbors_idx]
        
        # Create signature: (node_feature, sorted_neighbor_features_with_weights)
        if len(neighbors_idx) > 0:
            # Sort neighbors by weight (descending) for consistent ordering
            sorted_idx = np.argsort(-neighbor_weights)
            neighbors_sorted = neighbors_idx[sorted_idx]
            weights_sorted = neighbor_weights[sorted_idx]
            
            # Create tuple signature
            neighbor_sig = tuple((int(n_idx), round(float(w), 4)) for n_idx, w in zip(neighbors_sorted, weights_sorted))
            fingerprint = (int(current_features[i]), neighbor_sig)
        else:
            fingerprint = (int(current_features[i]),)
        
        fingerprints.append(fingerprint_dict[fingerprint])
    
    return np.array(fingerprints), fingerprint_dict


def encode_weighted_graph(adjacency_matrix, node_features=None, radius=1, normalize=True):
    """
    Encode a weighted graph using struct2graph approach:
    1. Create fingerprints (node identity + neighborhood)
    2. Normalize adjacency matrix
    
    Args:
        adjacency_matrix: weighted adjacency matrix (n x n)
        node_features: optional node labels/features
        radius: neighborhood radius for fingerprints
        normalize: whether to apply normalization
    
    Returns:
        fingerprints: array of fingerprint IDs
        adjacency_normalized: normalized adjacency matrix
        fingerprint_dict: mapping of fingerprints to IDs
    """
    fingerprints, fp_dict = create_node_fingerprints(adjacency_matrix, node_features, radius)
    
    if normalize:
        adjacency_norm = normalize_weighted_adjacency(adjacency_matrix)
    else:
        adjacency_norm = np.array(adjacency_matrix, dtype=float)
    
    return fingerprints, adjacency_norm, fp_dict



def bytes_to_pdb_structure(blob: bytes) -> object:
    """
    Decompress gzip-compressed and load into PDB structure object without writing to disk.
    """

    pdb_text = gzip.decompress(blob).decode("utf-8")
    structure = read_pdb(pdb_text)
    
    
    return structure



"""
Minimal single-protein embedding function (untrained weights,
random projection). Import and call embed_fingerprints() directly
from another script.
"""

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# DIM = 20
# LAYER_GNN = 2


# class _SingleProteinEncoder(nn.Module):
#     def __init__(self, n_fingerprint, dim=DIM, layer_gnn=LAYER_GNN):
#         super(_SingleProteinEncoder, self).__init__()
#         self.layer_gnn = layer_gnn
#         self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
#         self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
#         self.W_attention = nn.Linear(dim, dim)
#         self.context = nn.Parameter(torch.zeros(dim, 1))

#     def gnn(self, xs, A):
#         for i in range(self.layer_gnn):
#             hs = torch.relu(self.W_gnn[i](xs))
#             xs = torch.matmul(A, hs)
#         return xs

#     def self_attention(self, h):
#         u = torch.tanh(self.W_attention(h))
#         scores = torch.matmul(u, self.context).view(-1)
#         alpha = torch.softmax(scores, dim=0)
#         s = torch.matmul(torch.t(h), alpha).view(1, -1)
#         return s

#     def forward(self, fingerprint, adjacency):
#         x = self.embed_fingerprint(fingerprint)
#         x = self.gnn(x, adjacency)
#         return self.self_attention(x)


MODEL_DEFAULTS = {
    'radius': 1,
    'dim': 20,
    'layer_gnn': 2,
    'lr': 1e-3,
    'lr_decay': 0.5,
    'decay_interval': 10,
    'iteration': 100,
    'num_trials': 5
}


class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, MODEL_DEFAULTS['dim'])
        self.W_gnn             = nn.ModuleList([nn.Linear(MODEL_DEFAULTS['dim'], MODEL_DEFAULTS['dim'])
                                    for _ in range(MODEL_DEFAULTS['layer_gnn'])])
        self.W1_attention      = nn.Linear(MODEL_DEFAULTS['dim'], MODEL_DEFAULTS['dim'])
        self.W2_attention      = nn.Linear(MODEL_DEFAULTS['dim'], MODEL_DEFAULTS['dim'])
        self.w                 = nn.Parameter(torch.zeros(MODEL_DEFAULTS['dim'],1))
        
        self.W_out             = nn.Linear(2*MODEL_DEFAULTS['dim'], 2)
        
    def gnn(self, xs1, A1, xs2, A2):
        for i in range(MODEL_DEFAULTS['layer_gnn']):
            hs1 = torch.relu(self.W_gnn[i](xs1))            
            hs2 = torch.relu(self.W_gnn[i](xs2))
            
            xs1 = torch.matmul(A1, hs1)
            xs2 = torch.matmul(A2, hs2)
        
        return xs1, xs2
        
    
    def mutual_attention(self, h1, h2):
        x1 = self.W1_attention(h1)
        x2 = self.W2_attention(h2)
        
        m1 = x1.size()[0]
        m2 = x2.size()[0]
        
        c1 = x1.repeat(1,m2).view(m1, m2, MODEL_DEFAULTS['dim'])
        c2 = x2.repeat(m1,1).view(m1, m2, MODEL_DEFAULTS['dim'])

        d = torch.tanh(c1 + c2)
        alpha = torch.matmul(d,self.w).view(m1,m2)
        
        b1 = torch.mean(alpha,1)
        p1 = torch.softmax(b1,0)
        s1 = torch.matmul(torch.t(x1),p1).view(-1,1)
        
        b2 = torch.mean(alpha,0)
        p2 = torch.softmax(b2,0)
        s2 = torch.matmul(torch.t(x2),p2).view(-1,1)
        
        return torch.cat((s1,s2),0).view(1,-1), p1, p2

    
    def forward(self, inputs):

        fingerprints1, adjacency1, fingerprints2, adjacency2 = inputs
        
        """Protein vector with GNN."""
        x_fingerprints1        = self.embed_fingerprint(fingerprints1)
        x_fingerprints2        = self.embed_fingerprint(fingerprints2)
        
        x_protein1, x_protein2 = self.gnn(x_fingerprints1, adjacency1, x_fingerprints2, adjacency2)
        
        """Protein vector with mutual-attention."""
        y, p1, p2     = self.mutual_attention(x_protein1, x_protein2)

        return y, p1, p2




# Single shared (untrained) encoder instance, reused across calls so
# repeated calls in a loop don't reinitialize random weights each time.
_encoder_cache = {}


def embed_fingerprints_single(fingerprint, adjacency, n_fingerprint, dim=DIM, layer_gnn=LAYER_GNN):
    """
    fingerprint: array-like of atom/residue indices, shape (n_nodes,)
    adjacency:   array-like adjacency matrix, shape (n_nodes, n_nodes)
    n_fingerprint: size of the fingerprint vocabulary (needed to size the
                   embedding table consistently across calls)

    Returns: numpy array, shape (dim,)
    """
    key = (n_fingerprint, dim, layer_gnn)
    if key not in _encoder_cache:
        model = ProteinProteinInteractionPrediction(n_fingerprint).to(device)
        model.eval()
        _encoder_cache[key] = model
    model = _encoder_cache[key]

    # Self-pairing: treat the same fingerprint and adjacency as both inputs to the model.
    protein1 = torch.LongTensor(fingerprint)
    adjacency1 = torch.FloatTensor(adjacency)
    protein2 = torch.LongTensor(fingerprint)
    adjacency2 = torch.FloatTensor(adjacency)

    inputs = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device))
    y, att1, att2 = model.forward(inputs)

    return y.detach().cpu().numpy()



def embed_fingerprints(fingerprint1, fingerprint2, adjacency1, adjacency2, n_fingerprint, dim=DIM, layer_gnn=LAYER_GNN):
    """
    fingerprint: array-like of atom/residue indices, shape (n_nodes,)
    adjacency:   array-like adjacency matrix, shape (n_nodes, n_nodes)
    n_fingerprint: size of the fingerprint vocabulary (needed to size the
                   embedding table consistently across calls)

    Returns: numpy array, shape (dim,)
    """
    key = (n_fingerprint, dim, layer_gnn)
    
    if key not in _encoder_cache:
        model = ProteinProteinInteractionPrediction(n_fingerprint).to(device)
        model.eval()
        _encoder_cache[key] = model
    model = _encoder_cache[key]

    protein1 = torch.LongTensor(fingerprint1)
    adjacency1 = torch.FloatTensor(adjacency1)
    protein2 = torch.LongTensor(fingerprint2)
    adjacency2 = torch.FloatTensor(adjacency2)

    inputs = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device))
    y, att1, att2 = model.forward(inputs)

    return y.detach().cpu().numpy()







def pad_vector(vector, max_length):
    """
    Pads a 1D vector with zeros to ensure it has a specified maximum length.
    
    Parameters:
    vector (list): The input 1D vector to be padded.
    max_length (int): The desired length of the output vector after padding.
    
    Returns:
    list: A new vector that is padded with zeros to the specified length.
    """
    if len(vector) > max_length:
        raise ValueError(f"Input vector length {len(vector)} exceeds the maximum length of {max_length}.")
    
    # Calculate the number of zeros needed for padding
    padding_length = max_length - len(vector)
    
    # Create a new vector with the original values followed by the required number of zeros
    padded_vector = vector + [0] * padding_length
    
    return padded_vector


# Second possible padding, which puts vector in the middle of the padded vector, with zeros on both sides
def pad_vector_middle(vector, max_length):
    """
    Pads a 1D vector with zeros to ensure it has a specified maximum length, centering the original vector.
    
    Parameters:
    vector (list): The input 1D vector to be padded.
    max_length (int): The desired length of the output vector after padding.
    
    Returns:
    list: A new vector that is padded with zeros to the specified length, with the original vector centered.
    """
    if len(vector) > max_length:
        raise ValueError(f"Input vector length {len(vector)} exceeds the maximum length of {max_length}.")
    
    # Calculate the number of zeros needed for padding
    total_padding = max_length - len(vector)
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    
    # Create a new vector with the original values centered and padded with zeros on both sides
    padded_vector = [0] * left_padding + vector + [0] * right_padding
    
    return padded_vector