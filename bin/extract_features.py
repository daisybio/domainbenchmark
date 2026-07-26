#!/usr/bin/env python3
import argparse
import h5py
import importlib
import json
import sqlite3
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", type=Path, required=True, help="Path to the database file"
    )
    parser.add_argument(
        "--feature", required=True,
        help="Feature name — used as the output identity (directory/model-input key)"
    )
    parser.add_argument(
        "--module", default=None,
        help="Feature module to import from features/. Defaults to --feature "
             "for backward compatibility (i.e. name == module, no params)."
    )
    parser.add_argument(
        "--params", default="{}",
        help="JSON dict of keyword arguments passed to extract_features()"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only)."
    )
    args = parser.parse_args()
    if args.run_id:
        print(f"[extract_features] run_id={args.run_id}")

    # module_name = args.module or args.feature
    feature_params = json.loads(args.params)

    print("Opening database and output file...")
    with (
        sqlite3.connect(args.db) as connection,
        h5py.File(args.out, "w") as output_file,
    ):
        print(
            f"Extracting feature '{args.feature}' "
            f"(module=features.{args.module}, params={feature_params})..."
        )
        
        feature_module = importlib.import_module(f"features.{args.module}")
        # If feature params is empty, call without params, else call with params.
        if not args.params or feature_params == {}:
            feature_module.extract_features(connection, output_file)
        else:
            feature_params = eval(feature_params) if isinstance(feature_params, str) else feature_params
            feature_module.extract_features(connection, output_file, **feature_params)