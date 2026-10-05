import numpy as np
import gzip

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley


def calculate_sasa_structure_level(domain):
    sr = ShrakeRupley()
    sr.compute(domain, level="C")  # Compute SASA at the structure level
    return sum(atom.sasa for residue in domain for atom in residue)


def calculate_sasa_residue_level(domain):
    from collections import OrderedDict
    sr = ShrakeRupley()
    sr.compute(domain, level="R")  # Compute SASA at the residue level
    sasa_values = OrderedDict()
    for residue in domain.get_residues():
        chain_id = residue.get_parent().get_id()
        key = (chain_id, residue.get_id())  # Use chain ID and residue ID as key
        sasa_values[key] = residue.sasa
    return sasa_values


def calculate_sasa_values(domain):
    """Compute SASA once per domain and return both residue-level and
    structure-level results, instead of running ShrakeRupley twice
    (once per aggregation level) on the same atoms.
 
    Returns:
        (residue_sasa, structure_sasa) where residue_sasa maps
        residue.get_id() -> summed SASA over that residue's atoms, and
        structure_sasa is the SASA summed over the whole domain.
    """
    sr = ShrakeRupley()
    sr.compute(domain, level="A")  # atom-level is the finest; aggregate ourselves
 
    residue_sasa = {}
    structure_sasa = 0.0
    for residue in domain.get_residues():
        r_sasa = sum(atom.sasa for atom in residue.get_atoms())
        chain_id = residue.get_parent().get_id()
        key = (chain_id, residue.get_id())
        residue_sasa[key] = r_sasa
        structure_sasa += r_sasa
 
    return residue_sasa, structure_sasa


def calculate_rsa_residue_level(domain):
    from collections import OrderedDict
    sasa_residue = calculate_sasa_residue_level(domain)

    # MAxSASA values by Tien et al. 2013, "Maximum allowed solvent accessibilities of residues in proteins" (https://doi.org/10.1002/prot.24286)
    max_sasa_values = {
        'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
        'GLN': 223.0, 'GLU': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
        'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
        'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
    }

    rsa_residue = OrderedDict()
    for residue in domain.get_residues():
        resname = residue.get_resname()
        chain_id = residue.get_parent().get_id()
        rid = residue.get_id()
        key = (chain_id, rid)
        sasa_value = sasa_residue.get(key, 0)
        max_sasa = max_sasa_values.get(resname, None)
        combined_key = (resname, chain_id, rid)
        if max_sasa is not None and max_sasa > 0:
            rsa_residue[combined_key] = sasa_value / max_sasa
        else:
            rsa_residue[combined_key] = None
    return rsa_residue


"""Fixed-size, per-entry encoding of ONE variable-size 2D matrix.

Independent of every other entry (no vocabulary, no fitted state, no model),
so it can be called inside the extraction loop and written out immediately.
numpy only; the pooling follows the semantics of torch's AdaptiveAvgPool2d /
AdaptiveMaxPool2d (cell i spans [floor(i*n/k), ceil((i+1)*n/k)) ).

Output layout (N_FEATURES = 56 floats):
    6   scalars      log1p(rows), log1p(cols), mean, std, max, density (>0)
    21  avg pyramid  adaptive average pooling to 1x1, 2x2, 4x4
    4   max pooling  adaptive max pooling to 2x2
    12  marginals    mean/std/min/max/median/sum of row sums and of column sums
    7   band profile mean value per sequence-separation band |i-j| (square
                     "intra-chain" matrices only; zeros if banded=False/non-square)
"""


import numpy as np

AVG_LEVELS = (2, 4)   # k=1 dropped: it is M.mean(), derivable from the row-sum mean
MAX_LEVELS = (2,)
SEP_EDGES = (1, 3, 5, 9, 17, 33, 65)  # bands [1,3) [3,5) [5,9) ... [65,inf)


def n_features(symmetric, banded=False):
    core = 9 if symmetric else 14
    levels = list(AVG_LEVELS) + list(MAX_LEVELS)
    grids = sum(k * (k + 1) // 2 if symmetric else k * k for k in levels)
    return core + grids + (len(SEP_EDGES) if banded else 0)


def _adaptive_pool(M, k, reduce, upper_only):
    r, c = M.shape
    P = np.empty((k, k))
    for i in range(k):
        r0, r1 = (i * r) // k, -((-(i + 1) * r) // k)
        for j in range(k):
            c0, c1 = (j * c) // k, -((-(j + 1) * c) // k)
            P[i, j] = reduce(M[r0:r1, c0:c1])
    return P[np.triu_indices(k)] if upper_only else P.ravel()


def _stats5(v):
    return np.array([v.mean(), v.std(), v.min(), v.max(), np.median(v)])


def _stats4(v):
    return np.array([v.std(), v.min(), v.max(), np.median(v)])


def encode_matrix(M, symmetric, banded=False, fill=0.0):
    """Encode a 2D matrix into n_features(symmetric, banded) floats.

    symmetric: True  -> square symmetric matrices (one row-sum pool, upper-triangle grids)
               False -> rectangular / non-symmetric matrices (row- and column-sum pools)
               Must be constant per feature.
    banded:    add the |i-j| band profile; only for symmetric intra-chain maps.
    fill:      replacement for NaN/inf.
    """
    if banded and not symmetric:
        raise ValueError("banded=True requires symmetric=True")
    nf = n_features(symmetric, banded)

    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {M.shape}")
    if symmetric and M.shape[0] != M.shape[1]:
        raise ValueError(f"symmetric=True needs a square matrix, got {M.shape}")
    if M.size == 0:
        return np.zeros(nf, dtype=np.float32)
    M = np.nan_to_num(M, nan=fill, posinf=fill, neginf=fill)
    r, c = M.shape

    rs = M.sum(axis=1)
    if symmetric:
        # col sums == row sums, log1p(c) == log1p(r)
        core = np.concatenate([
            [np.log1p(r), M.std(), M.max(), (M > 0).mean()],
            _stats5(rs),
        ])
    else:
        cs = M.sum(axis=0)
        core = np.concatenate([
            [np.log1p(r), np.log1p(c), M.std(), M.max(), (M > 0).mean()],
            _stats5(rs),
            _stats4(cs),  # col mean derivable from rs.mean(), r, c
        ])

    parts = [core]
    for k in AVG_LEVELS:
        parts.append(_adaptive_pool(M, k, np.mean, symmetric))
    for k in MAX_LEVELS:
        parts.append(_adaptive_pool(M, k, np.max, symmetric))

    if banded:
        band = np.zeros(len(SEP_EDGES))
        idx = np.arange(r)
        D = np.abs(idx[:, None] - idx[None, :])
        edges = list(SEP_EDGES) + [np.inf]
        for b in range(len(SEP_EDGES)):
            mask = (D >= edges[b]) & (D < edges[b + 1])
            if mask.any():
                band[b] = M[mask].mean()
        parts.append(band)

    out = np.concatenate(parts).astype(np.float32)
    # assert out.shape[0] == nf, (out.shape[0], nf)
    return out



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


def bytes_to_pdb_structure(pdb_gz: bytes, structure_id: str):
    import io
    """Inverse of utils_struct.ddi_pair_to_bytes.

    `pdb_gz` is gzip-compressed PDB text, as stored in structures.h5 (see
    the note on ddi_pair_to_bytes: already-compressed, so the dataset itself
    uses compression=None). Decompress, then parse with the same PDBParser
    settings used to build it (QUIET=True) so warnings are suppressed the
    same way on the round trip.

    `structure_id` is just Bio.PDB's internal label for the returned
    Structure object -- purely cosmetic, doesn't affect parsing.
    """
    pdb_text = gzip.decompress(pdb_gz).decode("utf-8")

    parser = PDBParser(QUIET=True)
    
    return parser.get_structure(structure_id, io.StringIO(pdb_text))



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


import torch
import torch.nn as nn
device = torch.device('cpu')

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



def pool_residue_feature(values: np.ndarray) -> np.ndarray:
    """Collapse a variable-length residue-wise feature vector into a fixed-size summary."""
    if len(values) == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        values.mean(),
        values.std(),
        values.min(),
        values.max(),
        np.median(values),
        values.sum(),
    ], dtype=np.float32)



# def pad_vector(vector, max_length):
#     """
#     Pads a 1D numpy array with zeros to ensure it has a specified maximum length.

#     Parameters:
#     vector (np.ndarray): The input 1D vector to be padded.
#     max_length (int): The desired length of the output vector after padding.

#     Returns:
#     np.ndarray: A new array padded with zeros to the specified length.
#     """
#     vector = np.asarray(vector)

#     if vector.ndim != 1:
#         raise ValueError(f"Expected a 1D vector, got shape {vector.shape}.")

#     if len(vector) > max_length:
#         raise ValueError(f"Input vector length {len(vector)} exceeds the maximum length of {max_length}.")

#     padding_length = max_length - len(vector)

#     return np.pad(vector, (0, padding_length), mode='constant', constant_values=0)


# # Second possible padding, which puts vector in the middle of the padded vector, with zeros on both sides
# def pad_vector_middle(vector, max_length):
#     """
#     Pads a 1D vector with zeros to ensure it has a specified maximum length, centering the original vector.
    
#     Parameters:
#     vector (list): The input 1D vector to be padded.
#     max_length (int): The desired length of the output vector after padding.
    
#     Returns:
#     list: A new vector that is padded with zeros to the specified length, with the original vector centered.
#     """
#     if len(vector) > max_length:
#         raise ValueError(f"Input vector length {len(vector)} exceeds the maximum length of {max_length}.")
    
#     # Calculate the number of zeros needed for padding
#     total_padding = max_length - len(vector)
#     left_padding = total_padding // 2
#     right_padding = total_padding - left_padding
    
#     # Create a new vector with the original values centered and padded with zeros on both sides
#     padded_vector = [0] * left_padding + vector + [0] * right_padding
    
#     return padded_vector