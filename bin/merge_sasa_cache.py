#!/usr/bin/env python3
"""Merge per-shard sasa cache files into one sasa_cache.h5."""
import argparse
import os

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    n = 0
    with h5py.File(a.out, "w") as out:
        for path in sorted(a.shards):
            with h5py.File(path, "r") as src:
                for key in src.keys():
                    if key in out:          # cannot happen with crc32 partitioning; be safe
                        continue
                    src.copy(src[key], out, name=key)   # copies datasets + attrs
                    n += 1
    print(f"merge_sasa_cache: {n} structures from {len(a.shards)} shards -> {a.out}")


if __name__ == "__main__":
    main()