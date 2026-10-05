#!/usr/bin/env python3
"""Compute SASA ONCE per unique DDI structure and cache it in one HDF5 file.

Every SASA-based feature (aacomp_interface, bsa_*, interface_area*, sasa_*,
rasa_*, hydrophobic_map, charge_map, ...) then reads from this cache instead of
re-parsing the PDB and re-running Shrake-Rupley itself.

Run it as its own Nextflow process, ONCE, before FEATURE_EXTRACTION (single
writer; HDF5 does not like concurrent writers):

    python -m features.sasa_cache \
        --db train.sqlite3 validation.sqlite3 test.sqlite3 ... \
        --struct structures.h5 --out sasa_cache.h5 --procs 32

Structures that occur in several splits / datasets (minimal_leakage, random)
are computed only once, because the key is (pfam_a, pfam_b, inst_a, inst_b).

Per structure it stores, aligned residue-by-residue (all chain-A residues,
then all chain-B residues):
    resname, chain, ca_xyz (NaN if no CA)
    sasa_iso   residue SASA with the chain ALONE
    sasa_cplx  residue SASA inside the A+B complex
    attrs total_A, total_B, total_AB   (structure-level SASA)
"""
import argparse
import sqlite3
from multiprocessing import Pool

import h5py
import numpy as np
from Bio.PDB.SASA import ShrakeRupley

from scipy.spatial.distance import cdist
from features import embeddings
from .structure_utils import bytes_to_pdb_structure

# Tien et al. 2013 (same values as structure_utils.calculate_rsa_residue_level)
MAX_SASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLN': 223.0, 'GLU': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}


def make_key(pfam_a, pfam_b, inst_a, inst_b):
    return f"{pfam_a}|{pfam_b}|{inst_a}|{inst_b}"


def atom_depth(atoms):
    """Distance of each atom to the nearest solvent-accessible atom (0 for accessible atoms).

    Expects atom.sasa to be set (complex-level Shrake-Rupley already run).
    Same logic as the old depth_index.calculate_depth.
    """
    n = len(atoms)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    sasa = np.array([a.sasa for a in atoms], dtype=np.float64)
    coords = np.array([a.coord for a in atoms], dtype=np.float64)

    accessible = sasa > 0
    if not accessible.any():
        return np.zeros(n, dtype=np.float32)

    min_dist = cdist(coords, coords[accessible]).min(axis=1)
    return np.where(accessible, 0.0, min_dist).astype(np.float32)


def compute_bundle(pdb_gz: bytes, n_points: int = 100):
    """3 Shrake-Rupley runs (A alone, B alone, A+B), aggregated per residue."""
    structure = bytes_to_pdb_structure(pdb_gz, "s")
    model = structure[0]
    if len(model) < 2:
        return None
    chains = [model["A"], model["B"]]
    sr = ShrakeRupley(n_points=n_points)

    chain_ids, resnames, ca, iso = [], [], [], []
    for ch in chains:
        sr.compute(ch, level="A")
        for res in ch.get_residues():
            chain_ids.append(ch.get_id())
            resnames.append(res.get_resname())
            ca.append(res["CA"].get_coord() if "CA" in res else [np.nan] * 3)
            iso.append(sum(a.sasa for a in res))

    sr.compute(structure, level="A")
    cplx = [sum(a.sasa for a in res) for ch in chains for res in ch.get_residues()]
    total_AB = float(sum(a.sasa for a in structure.get_atoms()))
    depth_atom = atom_depth(list(structure.get_atoms()))   # NEW

    chain_ids = np.array(chain_ids)
    iso = np.array(iso, dtype=np.float32)
    return {
        "resname": np.array(resnames, dtype="S3"),
        "chain": np.array(chain_ids, dtype="S1"),
        "ca_xyz": np.array(ca, dtype=np.float32),
        "sasa_iso": iso,
        "sasa_cplx": np.array(cplx, dtype=np.float32),
        "depth_atom": depth_atom,                           # NEW
        "total_A": float(iso[chain_ids == "A"].sum()),
        "total_B": float(iso[chain_ids == "B"].sum()),
        "total_AB": total_AB,
    }

class SasaCache:
    """Read-only access; safe to open from many feature processes at once."""

    def __init__(self, path):
        self.f = h5py.File(path, "r")

    def get(self, pfam_a, pfam_b, inst_a, inst_b):
        key = make_key(pfam_a, pfam_b, inst_a, inst_b)
        if key not in self.f:
            return None
        g = self.f[key]
        return {
            "resname": g["resname"][:].astype(str),
            "chain": g["chain"][:].astype(str),
            "ca": g["ca_xyz"][:],
            "sasa_iso": g["sasa_iso"][:],
            "sasa_cplx": g["sasa_cplx"][:],
            "total_A": float(g.attrs["total_A"]),
            "total_B": float(g.attrs["total_B"]),
            "total_AB": float(g.attrs["total_AB"]),
            "depth_atom": g["depth_atom"][:],  # NEW
        }


# ---------------------------------------------------------------- precompute
def _jobs(dbs, struct_file, done):
    seen = set(done)
    for db in dbs:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        df = embeddings.load_structure_data(conn)
        for _ddi_id, ia, ib, pa, pb in df.itertuples(index=False):
            key = make_key(pa, pb, ia, ib)
            if key in seen:
                continue
            pdb_gz = embeddings.read_interaction_instance(struct_file, pa, pb, ia, ib)
            if pdb_gz is None:
                continue
            seen.add(key)
            yield key, pdb_gz.tobytes()
        conn.close()


def _work(job):
    key, pdb_gz = job
    try:
        return key, compute_bundle(pdb_gz)
    except Exception as e:  # keep the run alive, report at the end
        return key, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+", required=True)
    ap.add_argument("--struct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()

    n_ok = n_skip = 0
    errors = []
    with h5py.File(a.struct, "r") as sf, h5py.File(a.out, "a") as out, Pool(a.procs) as pool:
        for key, res in pool.imap_unordered(_work, _jobs(a.db, sf, out.keys()), chunksize=16):
            if res is None:
                n_skip += 1          # < 2 chains
                continue
            if isinstance(res, str):
                errors.append((key, res))
                continue
            g = out.create_group(key)
            for name in ("resname", "chain", "ca_xyz", "sasa_iso", "sasa_cplx", "depth_atom"):
                g.create_dataset(name, data=res[name])
            for name in ("total_A", "total_B", "total_AB"):
                g.attrs[name] = res[name]
            n_ok += 1
            if n_ok % 5000 == 0:
                out.flush()
    print(f"sasa_cache: wrote {n_ok}, skipped {n_skip} (<2 chains), failed {len(errors)}")
    for key, msg in errors[:20]:
        print("  FAILED", key, msg)


if __name__ == "__main__":
    main()