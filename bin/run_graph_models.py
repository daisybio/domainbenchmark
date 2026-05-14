#! /usr/bin/env python3

# Create dummy script for graph models

import os
from kgiddi import run_kgiddi
from ddiparsimony import run_ddiparsimony

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, help="Path to database file(s)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--params",
        required=True,
        help="Path to model JSON file, containing model parameters",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output model path for additional files"
    )
    parser.add_argument(
        "--out_predictions", required=False, help="Output predictions file path"
    )
    parser.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )
    args = parser.parse_args()
    if args.run_id:
        print(f"[run_graph_models] run_id={args.run_id}")

    if args.model == "kgiddi":
        json_file = os.path.join(args.params, "kgiddi.json")
        # Call kgiddi with appropriate parameters
        print(
            f"Run KGIDDI graph model with database {args.database} and parameters from {json_file}, output to {args.out_dir} and predictions to {args.out_predictions}"
        )
        run_kgiddi(args.database, json_file, args.out_dir, args.out_predictions)
    elif args.model == "kgiddi_random":
        json_file = os.path.join(args.params, "kgiddi_random.json")
        print(
            f"Run KGIDDI_RANDOM graph model with database {args.database} and parameters from {json_file}, output to {args.out_dir} and predictions to {args.out_predictions}"
        )
        run_kgiddi(args.database, json_file, args.out_dir, args.out_predictions)
    elif args.model == "ddiparsimony":
        json_file = os.path.join(args.params, "ddiparsimony.json")
        print(
            f"Run DDIParsimony graph model with database {args.database} and parameters from {json_file}, output to {args.out_dir} and predictions to {args.out_predictions}"
        )
        run_ddiparsimony(args.database, json_file, args.out_dir, args.out_predictions)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    with open(f"{args.out_dir}/model.txt", "w") as f:
        f.write(f"Run {args.model} model using database {args.database}\n")
