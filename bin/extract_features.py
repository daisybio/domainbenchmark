#!/usr/bin/env python3
import argparse
import contextlib
import h5py
import importlib
import inspect
import json
import sqlite3
from pathlib import Path

from determinism import seed_everything


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
        "--struct-file", type=Path, default=None,
        help="Optional path to the DDI structures HDF5 file (see "
             "embeddings.interaction_group_name/interaction_dataset_name). "
             "Passed to extract_features() as struct_file only for encoder "
             "modules that declare that parameter -- see the inspect check "
             "below. Structure-less encoders never see it, so this flag is "
             "safe to pass unconditionally whenever params.structures is set."
    )
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

    feature_params = json.loads(args.params)

    print("Opening database and output file...")
    with (
        sqlite3.connect(args.db) as connection,
        h5py.File(args.out, "w") as output_file,
        contextlib.ExitStack() as struct_stack,
    ):
        struct_file = None
        if args.struct_file is not None:
            struct_file = struct_stack.enter_context(h5py.File(args.struct_file, "r"))

        print(
            f"Extracting feature '{args.feature}' "
            f"(module=features.{args.module}, params={feature_params}, "
            f"structures={'yes' if struct_file is not None else 'no'})..."
        )

        feature_module = importlib.import_module(f"features.{args.module}")

        call_kwargs = feature_params if isinstance(feature_params, dict) else {}
        if isinstance(feature_params, str):
            call_kwargs = json.loads(feature_params)

        # Only pass struct_file to encoders that actually declare that
        # parameter -- e.g. aacomp_interface.extract_features(conn, out_file,
        # seed, struct_file), but not aacomp.extract_features(conn, out_file,
        # seed). This is what lets us stage the structures file into every
        # task unconditionally instead of threading a per-feature
        # "is this a structure encoder" flag through the Nextflow layer.
        sig = inspect.signature(feature_module.extract_features)
        if "struct_file" in sig.parameters:
            if struct_file is None:
                raise ValueError(
                    f"Feature module '{args.module}' requires struct_file but "
                    f"no --struct-file was given (is params.structures set?)."
                )
            call_kwargs["struct_file"] = struct_file

        feature_module.extract_features(connection, output_file, args.seed, **call_kwargs)