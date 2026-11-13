"""Shrake-Rupley once per structure, shared by every SASA-based encoder.

Cache layout mirrors structures.h5: `cache[group][dataset][chain]` with
float64 datasets `iso` (per-residue SASA of the chain alone), `cplx`
(per-residue SASA inside the A+B complex) and `depth` (per-atom distance to the
nearest solvent-accessible atom in the complex), residues/atoms in PDB order.
`chain` is the stored chain id (A/B); encoders map it to their own chain order.
It holds exactly what `structure_utils.compute_sasa` returns, so an encoder
reads the same numbers with or without the cache.

`build_cache` covers every structure any of the given split databases needs;
`n_shards` partitions that set by sha256 of the key, so shards are disjoint
and independent of input order. `merge_shards` joins them.
"""
import hashlib
import multiprocessing
import sqlite3

import h5py

from features import structure_utils

_STRUCT = None


def _init_worker(structures):
    global _STRUCT
    _STRUCT = h5py.File(structures, "r")


def _work(key):
    group, dataset = key
    chain_a, chain_b = structure_utils.parse_complex(_STRUCT[group][dataset][()], f"{group}/{dataset}")
    return key, structure_utils.compute_sasa(chain_a, chain_b)


def shard_of(group, dataset, n_shards):
    return int.from_bytes(hashlib.sha256(f"{group}/{dataset}".encode()).digest()[:8], "big") % n_shards


def build_cache(dbs, structures, out, procs=1, shard_id=0, n_shards=1):
    """Write this shard's SASA entries to `out`; any failing structure raises."""
    if not 0 <= shard_id < n_shards:
        raise ValueError(f"shard_id {shard_id} out of range for n_shards {n_shards}")
    keys = set()
    with h5py.File(structures, "r") as struct_h5:
        for db in dbs:
            print(f"{db}:")
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = structure_utils.structure_rows(conn, struct_h5)
            finally:
                conn.close()
            keys.update((row[4], row[5]) for row in rows)
    jobs = sorted(k for k in keys if shard_of(*k, n_shards) == shard_id)

    with multiprocessing.get_context("spawn").Pool(procs, _init_worker, (structures,)) as pool, \
            h5py.File(out, "w") as out_h5:
        # imap keeps job order, so entries are written in sorted key order.
        for (group, dataset), by_chain in pool.imap(_work, jobs, chunksize=4):
            entry = out_h5.require_group(group).create_group(dataset)
            for chain_id in sorted(by_chain):
                for name in ("iso", "cplx", "depth"):
                    entry.create_dataset(f"{chain_id}/{name}", data=by_chain[chain_id][name])
    print(f"sasa_cache shard {shard_id}/{n_shards}: {len(jobs)} of {len(keys)} structures")


def merge_shards(shards, out):
    """Merge shard files into one cache; a key in two shards is an error."""
    sources = [h5py.File(path, "r") for path in shards]
    try:
        entries = sorted(
            (group, dataset, i) for i, src in enumerate(sources) for group in src for dataset in src[group]
        )
        with h5py.File(out, "w") as out_h5:
            for group, dataset, i in entries:
                if group in out_h5 and dataset in out_h5[group]:
                    raise ValueError(f"{group}/{dataset} occurs in more than one shard ({shards[i]})")
                sources[i].copy(sources[i][group][dataset], out_h5.require_group(group), name=dataset)
    finally:
        for src in sources:
            src.close()
    print(f"sasa_cache: merged {len(entries)} structures from {len(shards)} shards")
