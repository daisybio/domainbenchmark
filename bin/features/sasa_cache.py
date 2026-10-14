#!/usr/bin/env python3
"""Compute SASA ONCE per unique DDI structure and cache it in one HDF5 file.

Every SASA-based feature (aacomp_interface, bsa_*, interface_area*, sasa_*,
rasa_*, hydrophobic_map, charge_map, ...) then reads from this cache instead of
re-parsing the PDB and re-running Shrake-Rupley itself.

Run it as its own Nextflow process, once, before FEATURE_EXTRACTION:

    python -m features.sasa_cache \
        --db train.sqlite3 validation.sqlite3 test.sqlite3 ... \
        --struct structures.h5 --out sasa_cache.h5 --procs 12

Structures that occur in several splits / datasets (minimal_leakage, random)
are computed only once, because the key is (pfam_a, pfam_b, inst_a, inst_b).

Per structure it stores, aligned residue-by-residue (all chain-A residues,
then all chain-B residues):
    resname, chain, ca_xyz (NaN if no CA)
    sasa_iso   residue SASA with the chain ALONE
    sasa_cplx  residue SASA inside the A+B complex
    attrs total_A, total_B, total_AB   (structure-level SASA)
    attrs num_res_A, num_res_B, num_res_AB   (structure-level residue counts)
"""
import argparse
import sqlite3
from multiprocessing import Pool
import os


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
        "num_res_A": int((chain_ids == "A").sum()),
        "num_res_B": int((chain_ids == "B").sum()),
        "num_res_AB": int((chain_ids == "A").sum() + (chain_ids == "B").sum())
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
            "depth_atom": g["depth_atom"][:],
            "num_res_A": int((g["chain"][:] == b"A").sum()),
            "num_res_B": int((g["chain"][:] == b"B").sum()),
            "num_res_AB": int((g["chain"][:] == b"A").sum() + (g["chain"][:] == b"B").sum())
        }


# ---------------------------------------------------------------- precompute

_SF = None  # per-worker read-only handle to structures.h5


def _init_worker(struct_path):
    global _SF
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    _SF = h5py.File(struct_path, "r")


def _jobs(dbs, done):
    """Yield only (key, (pfam_a, pfam_b, inst_a, inst_b)) -- no structure bytes."""
    seen = set(done)
    for db in dbs:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        df = embeddings.load_structure_data(conn)
        conn.close()
        for _ddi_id, ia, ib, pa, pb in df.itertuples(index=False):
            key = make_key(pa, pb, ia, ib)
            if key in seen:
                continue
            seen.add(key)
            yield key, (pa, pb, ia, ib)


def _work(job):
    key, (pa, pb, ia, ib) = job
    try:
        pdb_gz = embeddings.read_interaction_instance(_SF, pa, pb, ia, ib)
        if pdb_gz is None:
            return key, False                      # structure missing
        return key, compute_bundle(pdb_gz.tobytes())
    except Exception as e:  # keep the run alive, report at the end
        return key, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", nargs="+", required=True)
    ap.add_argument("--struct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()

    n_ok = n_skip = n_missing = 0
    errors = []
    # Pool first (fork before any HDF5 file is open in the parent)
    with Pool(a.procs, initializer=_init_worker, initargs=(a.struct,),
              maxtasksperchild=2000) as pool:
        with h5py.File(a.out, "a") as out:
            done = list(out.keys())
            for key, res in pool.imap_unordered(_work, _jobs(a.db, done), chunksize=8):
                if res is False:
                    n_missing += 1
                    continue
                if res is None:
                    n_skip += 1          # < 2 chains
                    continue
                if isinstance(res, str):
                    errors.append((key, res))
                    continue
                g = out.create_group(key)
                for name in ("resname", "chain", "ca_xyz", "sasa_iso", "sasa_cplx", "depth_atom"):
                    g.create_dataset(name, data=res[name])
                for name in ("total_A", "total_B", "total_AB", "num_res_A", "num_res_B", "num_res_AB"):
                    g.attrs[name] = res[name]
                n_ok += 1
                if n_ok % 5000 == 0:
                    out.flush()
    print(f"sasa_cache: wrote {n_ok}, skipped {n_skip} (<2 chains), "
          f"missing {n_missing}, failed {len(errors)}")
    for key, msg in errors[:20]:
        print("  FAILED", key, msg)


if __name__ == "__main__":
    main()