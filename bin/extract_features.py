#!/usr/bin/env python3
import argparse
import h5py
import importlib
import sqlite3
from pathlib import Path

from determinism import seed_everything


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", type=Path, required=True, help="Path to the database file"
    )
    parser.add_argument("--feature", required=True, help="Feature name to extract")
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed, passed to every encoder as its third argument. "
             "Only the `dummy` encoder draws from it today, but an encoder that "
             "samples must be reproducible like everything else downstream of "
             "it -- and must derive per-item child seeds rather than draw from "
             "the global RNG in row order.",
    )
    args = parser.parse_args()
    if args.run_id:
        print(f"[extract_features] run_id={args.run_id}")

    seed_everything(args.seed)

    print("Opening database and output file...")
    with (
        sqlite3.connect(args.db) as connection,
        h5py.File(args.out, "w") as output_file,
    ):
        print(f"Extracting feature '{args.feature}'...")
        feature_module = importlib.import_module(f"features.{args.feature}")
        # Every encoder takes the seed, whether or not it draws from it today:
        # an encoder that samples must not be able to reach the global RNG.
        feature_module.extract_features(connection, output_file, args.seed)
