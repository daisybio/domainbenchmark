#!/usr/bin/env python3
import argparse
import h5py
import importlib
import json
import sqlite3
from pathlib import Path

from determinism import derive_seed, seed_everything


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", type=Path, required=True, help="Path to the database file"
    )
    parser.add_argument("--feature", required=True, help="Feature name to extract")
    parser.add_argument(
        "--module", default=None,
        help="Encoder module under bin/features/ (default: --feature). Differs "
             "from --feature for a parameterised variant (assets/feature_variants.json).",
    )
    parser.add_argument(
        "--params", type=json.loads, default={},
        help="JSON object of keyword arguments for the encoder (a variant's params).",
    )
    parser.add_argument(
        "--structures", type=Path, default=None,
        help="domainsplit's structures.h5; passed to structure encoders only.",
    )
    parser.add_argument(
        "--sasa_cache", type=Path, default=None,
        help="Merged SASA cache (bin/merge_sasa_cache.py); structure encoders only.",
    )
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
        feature_module = importlib.import_module(f"features.{args.module or args.feature}")
        # Every encoder takes the seed, whether or not it draws from it today:
        # an encoder that samples must not be able to reach the global RNG.
        # The seed is keyed on the split (the db file's stem: train,
        # validation, test_realistic, ...). Keyed on --seed alone, `dummy` gave
        # a domain instance the identical vector in every split, so a model
        # could memorise instances in train and recognise them in test.
        split_seed = derive_seed(args.seed, "split", args.db.stem)
        kwargs = dict(args.params)
        if args.structures:
            kwargs.update(structures=str(args.structures), sasa_cache=args.sasa_cache and str(args.sasa_cache))
        feature_module.extract_features(connection, output_file, split_seed, **kwargs)
