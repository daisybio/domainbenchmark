#! /usr/bin/env python3

# Entrypoint for the graph-based DDI models.
#
# A graph model is trained on the database's train split and then scored
# against every test split it ships (`test_balanced` + `test_realistic`, or a
# single `test`). Training is the expensive phase, so it happens once and only
# the scoring loop fans out -- one predictions file per variant.

import os

from determinism import seed_everything
from kgiddi import run_kgiddi
from ddiparsimony import run_ddiparsimony


def variant_of(test_split):
    """`test_balanced` -> `balanced`, `test` -> `test`."""
    return test_split[len("test_"):] if test_split.startswith("test_") else test_split


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
        "--out_predictions_dir",
        required=True,
        help="Directory to write predictions_<variant>.parquet into",
    )
    parser.add_argument(
        "--test_splits",
        nargs="+",
        default=["test"],
        help="Test split names to score (e.g. test_balanced test_realistic)",
    )
    parser.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Number of worker threads/processes (from task.cpus).",
    )
    parser.add_argument(
        "--ppi_score_cutoff", type=int, default=None,
        help="Minimum STRING combined_score a PPI must reach to enter the "
             "interactome, shared by every graph model so they all score the "
             "same network. Overrides the model JSON's own value; when omitted "
             "the JSON decides, and failing that DEFAULT_PPI_SCORE_CUTOFF.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed. Every randomised step derives a child seed from "
             "it and its own stable identity (split name, iteration index), so "
             "results do not depend on --threads or on worker scheduling.",
    )
    args = parser.parse_args()
    if args.run_id:
        print(f"[run_graph_models] run_id={args.run_id}")

    seed_everything(args.seed)

    os.makedirs(args.out_predictions_dir, exist_ok=True)
    test_splits = {
        variant_of(split): (
            split,
            os.path.join(args.out_predictions_dir, f"predictions_{variant_of(split)}.parquet"),
        )
        for split in args.test_splits
    }

    runners = {
        "kgiddi": (run_kgiddi, "kgiddi.json"),
        "kgiddi_random": (run_kgiddi, "kgiddi_random.json"),
        "ddiparsimony": (run_ddiparsimony, "ddiparsimony.json"),
    }
    if args.model not in runners:
        raise ValueError(f"Unknown model: {args.model}")

    runner, params_name = runners[args.model]
    json_file = os.path.join(args.params, params_name)
    print(
        f"Run {args.model.upper()} graph model with database {args.database} and "
        f"parameters from {json_file}, output to {args.out_dir} and predictions "
        f"to {sorted(path for _, path in test_splits.values())}"
    )
    runner(
        args.database, json_file, args.out_dir, test_splits,
        threads=args.threads, seed=args.seed,
        ppi_score_cutoff=args.ppi_score_cutoff,
    )

    with open(f"{args.out_dir}/model.txt", "w") as f:
        f.write(f"Run {args.model} model using database {args.database}\n")
        f.write(f"Test splits: {', '.join(args.test_splits)}\n")
