#!/usr/bin/env python3

import h5py
import numpy as np
import sqlite3
from collections import defaultdict
import torch

from .structure_utils import ProteinProteinInteractionPrediction, bytes_to_pdb_structure
from features import embeddings


AMINO_ACIDS = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','PYL','SER','SEC','THR','TRP','TYR','VAL','ASX','GLX','XAA','XLE']
AA = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','O','S','U','T','W','Y','V']
AA3_1_DICT = {aa3: aa1 for aa3, aa1 in zip(AMINO_ACIDS, AA)}
FINGERPRINT_DICT = defaultdict(lambda: len(FINGERPRINT_DICT))
MAX_RESIDUES = 2000

device = torch.device('cpu')



def get_data_from_chain(structure_chainlevel):
    amino = []
    group = []
    coords = []
    
    for residue in structure_chainlevel:
        if residue.get_resname() in AMINO_ACIDS or residue.get_resname() == 'FME':
            amino.append(residue.get_resname())
            group.append(residue.id[1])
            coords.append(residue['CA'].get_coord())
    
    return amino, group, coords



def group_by_coords(group,amino,coords):
    uniq_group   = np.unique(group)
    group_coords = np.zeros((uniq_group.shape[0],3))
    
    group_amino  = []
    
    np_group     = np.array(group)
    
    for i,e in enumerate(uniq_group):
        inds = np.where(np_group==e)[0]
        group_coords[i,:] = np.mean(np.array(coords)[inds],axis=0)
        group_amino.append(amino[inds[0]])
    
    return group_coords, group_amino


def create_fingerprints(atoms, adjacency, radius):
    """Extract r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""
    
    fingerprints = []
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [FINGERPRINT_DICT[a] for a in atoms]
    else:
        for i in range(len(atoms)):
            vertex      = atoms[i]
            neighbors   = tuple(set(tuple(sorted(atoms[np.where(adjacency[i]>0.0001)[0]]))))
            fingerprint = (vertex, neighbors)
            fingerprints.append(FINGERPRINT_DICT[fingerprint])
    
    return np.array(fingerprints)


def create_amino_acids(acids):
    retval = [AA3_1_DICT[acid_name] if acid_name in AMINO_ACIDS else AA3_1_DICT['MET'] if acid_name=='FME' else AA3_1_DICT['TMP'] for acid_name in acids]
    retval = np.array(retval)
    
    return retval



def embed_fingerprints(fingerprint1, fingerprint2, adjacency1, adjacency2, model):
    """
    fingerprint: array-like of atom/residue indices, shape (n_nodes,)
    adjacency:   array-like adjacency matrix, shape (n_nodes, n_nodes)
    n_fingerprint: size of the fingerprint vocabulary (needed to size the
                   embedding table consistently across calls)

    Returns: numpy array, shape (dim,)
    """
    
    protein1 = torch.LongTensor(fingerprint1)
    adjacency1 = torch.FloatTensor(adjacency1)
    protein2 = torch.LongTensor(fingerprint2)
    adjacency2 = torch.FloatTensor(adjacency2)

    inputs = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device))
    y, att1, att2 = model.forward(inputs)

    return y.detach().cpu().numpy()




def get_graph_from_struct(group_coords, group_amino):
    num_residues = group_coords.shape[0]
    
    if (num_residues > MAX_RESIDUES):
        num_residues = MAX_RESIDUES
    
    residues = group_amino[:num_residues]
        
    struct2graph = [ [0.0 for _ in range(0, num_residues)] for _ in range(0, num_residues)]
    
    residue_type = []
    for i in range(0, num_residues):
        if residues[i] == 'FME':
            residues[i] = 'MET'
        elif residues[i] not in AMINO_ACIDS:
            residues[i] = 'TMP'
            
        residue_type.append(residues[i])
        
        for j in range(i+1, num_residues):
            x, y = group_coords[i], group_coords[j]
            struct2graph[i][j] = float(np.linalg.norm(x-y))
            struct2graph[j][i] = struct2graph[i][j]
    
    struct2graph = np.array(struct2graph)
    
    threshold = 9.5
    
    for i in range(0, num_residues):
        for j in range(0, num_residues):
            if (struct2graph[i,j] <= threshold):
                struct2graph[i,j] = 1
            else:
                struct2graph[i,j] = 0

    n          = struct2graph.shape[0]
    adjacency  = struct2graph + np.eye(n)
    degree     = sum(adjacency)
    d_half     = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))

    return struct2graph, adjacency





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


    features = defaultdict(list)

    for ddi_id, instance_id_a, instance_id_b, pfam_id_a, pfam_id_b in ddi_df.itertuples(index=False):
        
        pdb_gz = embeddings.read_interaction_instance(struct_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)
        if pdb_gz is None:
            # No structure for this instance pair -- skip, per the
            # documented behavior in embeddings.py.
            continue
        structure = bytes_to_pdb_structure(pdb_gz.tobytes(), f"ddi_{ddi_id}") # pyright: ignore[reportAttributeAccessIssue]

        structA = structure[0]["A"]  # type: ignore[reportGeneralTypeIssues]
        structB = structure[0]["B"]  # type: ignore[reportGeneralTypeIssues]


        for struct in [structA, structB]:

            amino, group, coords = get_data_from_chain(struct)
            group_coords, group_amino = group_by_coords(group, amino, coords)
            struct2graph, adjacency = get_graph_from_struct(group_coords, group_amino)

            fingerprints = create_fingerprints(np.array(group_amino), struct2graph, radius=1)

            features[(pfam_id_a, pfam_id_b, instance_id_a, instance_id_b)].append((fingerprints, adjacency))


    fingerprint_dict_length = len(FINGERPRINT_DICT)
    n_fingerprint = fingerprint_dict_length + 100

    model = ProteinProteinInteractionPrediction(n_fingerprint).to(device)
    model.eval()

    n_written = 0
    for key, value in features.items():
        pfam_id_a, pfam_id_b, instance_id_a, instance_id_b = key
        # print(value)
        fingerpint_a, adjacency_a = value[0]
        fingerpint_b, adjacency_b = value[1]
        
        feature_vector = embed_fingerprints(fingerpint_a, fingerpint_b, adjacency_a, adjacency_b, model)
        # Feature vector has doublev [[...]] shape, we need to flatten it
        feature_vector = feature_vector.flatten()

        embeddings.write_interaction_instance(out_file, pfam_id_a, pfam_id_b, instance_id_a, instance_id_b, feature_vector)
        n_written += 1

        
    print(f"struct2graph: wrote {n_written} entries")
