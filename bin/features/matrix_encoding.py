# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:36:25 2019

@author: mayank
"""

import pickle
import os
import glob
import argparse as ap

import numpy as np

import torch
import torch.nn as nn

import sys



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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

model_params = MODEL_DEFAULTS



class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self, n_fingerprint):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, model_params['dim'])
        self.W_gnn             = nn.ModuleList([nn.Linear(model_params['dim'], model_params['dim'])
                                    for _ in range(model_params['layer_gnn'])])
        self.W1_attention      = nn.Linear(model_params['dim'], model_params['dim'])
        self.W2_attention      = nn.Linear(model_params['dim'], model_params['dim'])
        self.w                 = nn.Parameter(torch.zeros(model_params['dim'],1))
        
        self.W_out             = nn.Linear(2*model_params['dim'], 2)
        
    def gnn(self, xs1, A1, xs2, A2):
        for i in range(model_params['layer_gnn']):
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
        
        c1 = x1.repeat(1,m2).view(m1, m2, model_params['dim'])
        c2 = x2.repeat(m1,1).view(m1, m2, model_params['dim'])

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
    

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


    




def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='input directory for pdb files')
    # parser.add_argument('--radius', required=True, type=int, help='radius for subgraph extraction')
    parser.add_argument('--out_dir', required=True, help='output directory for embeddings')
    parser.add_argument('--model_params', help='path to model parameters file')
    return parser.parse_args()



def main():

    args = parse_args()
    dir_input = args.input_dir
    # radius = args.radius
    out_dir = args.out_dir

    if not args.model_params:
        model_params = MODEL_DEFAULTS
    else:
        model_params = load_pickle(args.model_params)
    

    fingerprint_dict_length = np.load(dir_input + 'fingerprint_dict_length', allow_pickle=True)
    n_fingerprint = fingerprint_dict_length[0] + 100

    p_list = {}
    A_list = {}


    # File format: fingerprints_{pdb}.npy and adjacencies_{pdb}.npy
    fingerprints_files = glob.glob(os.path.join(dir_input, 'fingerprints_*.npy'))
    for f in fingerprints_files:
        pdb_name = os.path.basename(f).replace('fingerprints_', '').replace('.npy', '')
        fingerprints = np.load(f, allow_pickle=True)
        adjacency = np.load(os.path.join(dir_input, f'adjacencies_{pdb_name}.npy'), allow_pickle=True)

        p_list[pdb_name] = fingerprints
        A_list[pdb_name] = adjacency


    model = ProteinProteinInteractionPrediction(n_fingerprint).to(device)
    model.eval()

    keys = list(p_list.keys())
    if not keys:
        print('No proteins found in p_list to extract embeddings from')
        sys.exit(0)

    protein_pairs = [(k1, k2) for k1 in keys for k2 in keys]
    print(f"Extracting embeddings for {len(protein_pairs)} protein pairs...")


    for p1_key, p2_key in protein_pairs:
        protein1 = torch.LongTensor(p_list[p1_key])
        adjacency1 = torch.FloatTensor(A_list[p1_key])
        protein2 = torch.LongTensor(p_list[p2_key])
        adjacency2 = torch.FloatTensor(A_list[p2_key])

        inputs = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device))
        y, att1, att2 = model.forward(inputs)
        emb = y.detach().cpu().numpy()
        outpath = os.path.join(out_dir, f'{p1_key}_{p2_key}_embedding.npy')
        np.save(outpath, emb)

