#!/usr/bin/env python3
"""Merge SASA cache shards into one file (see features/sasa_cache.py)."""
import argparse

from features.sasa_cache import merge_shards

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge_shards(a.shards, a.out)
