#!/usr/bin/env python3
"""Build one shard of the SASA cache (see features/sasa_cache.py)."""
import argparse

from features.sasa_cache import build_cache

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", nargs="+", required=True)
    ap.add_argument("--struct", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    a = ap.parse_args()
    build_cache(a.db, a.struct, a.out, a.procs, a.shard_id, a.n_shards)
