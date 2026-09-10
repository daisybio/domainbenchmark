#!/usr/bin/env python3

import argparse as ap
import logging
import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sklearn.metrics as metrics

from typing import List, Tuple, Dict
from tqdm import tqdm
import gc

from ddiparsimony_functions import (
    compute_lp_score,
    compute_p_values_from_matrix,
    count_witnesses,
    find_best_pw_cutoff_with_candidates,
    compute_fp_rate,
    compute_random_x_matrix_parallel,
)
from load_data_gm import (
    DEFAULT_PPI_SCORE_CUTOFF,
    check_file_existence,
    canonical_pair,
    load_ddi,
    load_pd_mapping,
    load_ppi,
)


# --- Training: grid search for best reliability and cutoff ---
def evaluate_reliability_train(
    args: Tuple[
        float,
        List[Tuple[str, str]],
        Dict[str, set],
        Dict[Tuple[str, str], int],
        List[Tuple[str, str]],
        np.ndarray,
        Dict[Tuple[str, str], int],
        Dict[Tuple[str, str], int],
        list,
        int,
    ],
) -> Tuple[float, float, float, float, Dict[Tuple[str, str], float]]:
    (
        r,
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        random_x_matrix,
        ddi_dict,
        ddi_score,
        candidate_cutoffs,
        seed,
    ) = args
    # `seed` rides along in the task tuple because this runs in a pool worker,
    # which inherits no usable RNG state. The token is keyed on the reliability
    # rate: that is what distinguishes one grid point from another, and the
    # `avg_x_vec` rerun in run_ddiparsimony passes the same token so it
    # reproduces exactly the vector these pw_scores were computed from.
    avg_score, n_runs, observed_x_avg = compute_lp_score(
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        reliability=r,
        num_runs=200,
        return_x=True,
        seed=seed,
        rng_token=f"train_grid:{r}",
    )
    if observed_x_avg is None:
        observed_x_avg = np.zeros(len(domain_pairs))
    p_values = compute_p_values_from_matrix(
        random_x_matrix, observed_x_avg, domain_pairs
    )
    witnesses = count_witnesses(ppi_list, protein_domains, domain_pairs)
    pw_scores = {}
    for pair in domain_pairs:
        w = witnesses.get(pair, 0)
        pval = float(p_values.get(pair, 1.0))
        pw = float(min(pval, (1 - r) ** w))
        pw_scores[pair] = pw
    cut, accuracy, fp_rate = find_best_pw_cutoff_with_candidates(
        pw_scores, ddi_dict, candidate_cutoffs
    )
    cut = float(cut) if cut is not None else -1.0
    accuracy = float(accuracy) if accuracy is not None else 0.0
    fp_rate = float(fp_rate) if fp_rate is not None else 0.0
    return float(r), cut, accuracy, fp_rate, pw_scores


# --- Testing: use given reliability and cutoff ---
def evaluate_reliability_test(
    args: Tuple[
        float,
        float,
        List[Tuple[str, str]],
        Dict[str, set],
        Dict[Tuple[str, str], int],
        List[Tuple[str, str]],
        np.ndarray,
        Dict[Tuple[str, str], Tuple[int, int]],
        Dict[Tuple[str, str], int],
        int,
        str,
    ],
) -> Tuple[float, float, float, float, Dict[Tuple[str, str], float]]:
    (
        r,
        cutoff,
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        random_x_matrix,
        ddi_dict,
        ddi_score,
        seed,
        variant,
    ) = args
    # Token carries the variant: two test splits of the same database are
    # scored at the same reliability, and they must not share a mask.
    avg_score, n_runs, observed_x_avg = compute_lp_score(
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        reliability=r,
        num_runs=200,
        return_x=True,
        seed=seed,
        rng_token=f"test_eval:{variant}:{r}",
    )
    if observed_x_avg is None:
        observed_x_avg = np.zeros(len(domain_pairs))
    p_values = compute_p_values_from_matrix(
        random_x_matrix, observed_x_avg, domain_pairs
    )
    witnesses = count_witnesses(ppi_list, protein_domains, domain_pairs)
    pw_scores = {}
    for pair in domain_pairs:
        w = witnesses.get(pair, 0)
        pval = float(p_values.get(pair, 1.0))
        pw = float(min(pval, (1 - r) ** w))
        pw_scores[pair] = pw
    # Use provided cutoff
    y_pred = [int(pw_scores[pair] <= cutoff) for pair in pw_scores]
    y_true = [ddi_dict[pair][0] == 1 for pair in pw_scores]
    accuracy = metrics.accuracy_score(y_true, y_pred)
    fp_rate = compute_fp_rate(y_pred, y_true)
    return float(r), float(cutoff), float(accuracy), float(fp_rate), pw_scores


def preprocessing(
    db_path: Path,
    output_dir: Path,
    threads: int = 1,
    seed: int = 42,
    ppi_score_cutoff: int = DEFAULT_PPI_SCORE_CUTOFF,
) -> Tuple[
    List[Tuple[str, str]],
    np.ndarray,
    Dict[Tuple[str, str], Tuple[int, int]],
    Dict[str, set],
    List[Tuple[str, str]],
    Dict[Tuple[str, str], int],
]:

    ddi_df = load_ddi(db_path)
    pd_df = load_pd_mapping(db_path)
    ppi_df = load_ppi(db_path)

    logging.info(
        f"Filtering PPI data by STRING confidence (score >= {ppi_score_cutoff})..."
    )
    n_ppi_raw = len(ppi_df)
    # Inclusive because STRING's confidence bands are closed at their lower
    # edge (>= 900 highest, >= 700 high, >= 400 medium), so a strict > drops
    # every row sitting exactly on the boundary. Kept identical to kgiddi's
    # cutoff so both graph models score the same interactome -- see the longer
    # note in bin/kgiddi.py for why 900 is too strict for a split database.
    ppi_df = ppi_df[ppi_df["score"] >= ppi_score_cutoff].reset_index(drop=True)
    # Same hazard as in kgiddi: an empty interactome does not raise here, it
    # produces an all-negative prediction set that only fails downstream in
    # EVAL_ONE (single-class y_true). Fail where the cause is visible.
    if ppi_df.empty:
        raise ValueError(
            f"{db_path}: no PPIs left after the confidence filter "
            f"(ppi_score_cutoff={ppi_score_cutoff}, {n_ppi_raw} PPI rows in the "
            "database). Either this split database ships no "
            "protein_protein_interaction rows, or its score column is on a "
            "different scale than the cutoff assumes -- check "
            "`SELECT count(*), min(score), max(score) FROM "
            "protein_protein_interaction`."
        )

    # Reduce ddi_df and pd_df based on proteins in ppi_df
    logging.info("Filtering DDI and PD mapping data based on PPI proteins...")
    proteins_in_ppi = set(ppi_df["protein_1"]).union(set(ppi_df["protein_2"]))
    pd_filtered_df = pd_df[pd_df["uniprot_id"].isin(proteins_in_ppi)].reset_index(
        drop=True
    )
    involved_domains = set(pd_filtered_df["pfam_id"].unique())
    ddi_filtered_df = ddi_df[
        ddi_df["domain_a"].isin(involved_domains)
        & ddi_df["domain_b"].isin(involved_domains)
    ].reset_index(drop=True)

    pd_df = pd_filtered_df
    ddi_df = ddi_filtered_df

    ddi_score = ddi_df.groupby(["domain_a", "domain_b"]).size().to_dict()

    ddi_df = ddi_df.drop_duplicates().reset_index(drop=True)

    # Write ddi scores to file
    with open(os.path.join(output_dir, "ddiparsimony_ddi_scores.json"), "w") as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in ddi_score.items()}, f, indent=4)  # type: ignore

    # Build adjacency matrix for randomization
    logging.info("Building adjacency matrix for PPI interactions...")
    # Get all proteins from ppi network
    proteins = sorted(set(ppi_df["protein_1"]).union(set(ppi_df["protein_2"])))
    protein_idx = {p: i for i, p in enumerate(proteins)}
    A = lil_matrix((len(proteins), len(proteins)))
    for p1, p2 in zip(ppi_df["protein_1"], ppi_df["protein_2"]):
        if p1 in protein_idx and p2 in protein_idx:
            A[protein_idx[p1], protein_idx[p2]] = 1
            A[protein_idx[p2], protein_idx[p1]] = 1

    # Build domain pairs and the corresponding index
    logging.info("Building domain pairs and their indices...")
    ddi_dict = {
        (row.domain_a, row.domain_b): (row.interaction, row.eval_relevant)
        for row in ddi_df.itertuples(index=False)
    }

    domain_pairs = list((d1, d2) for d1, d2 in ddi_dict.keys())
    domain_pair_to_idx = {pair: idx for idx, pair in enumerate(domain_pairs)}
    # Save domain pairs in .npy format
    np.save(
        os.path.join(output_dir, "ddiparsimony_domain_pairs.npy"),
        np.array(domain_pairs),
    )

    # Get protein_domains
    protein_domains = {}
    for protein, group in pd_df.groupby("uniprot_id"):
        protein_domains[protein] = set(group["pfam_id"].tolist())

    # Delete pd_df after use
    del pd_df, ddi_df
    gc.collect()

    # --- Parallelized computation ---
    # Cache random_x_matrix per dataset (db_path.stem = "train"/"test"). Train and
    # test share output_dir, so use dataset-suffixed filenames to avoid overwrite.
    # If a previous run already produced this matrix, skip the costly recompute.
    dataset_tag = db_path.stem
    rxm_path = os.path.join(
        output_dir, f"ddiparsimony_random_x_matrix_{dataset_tag}.npy"
    )
    expected_shape = (1000, len(domain_pairs))
    if os.path.exists(rxm_path):
        cached = np.load(rxm_path)
        if cached.shape == expected_shape:
            logging.info(f"Loaded cached random X matrix from {rxm_path}")
            random_x_matrix = cached
        else:
            logging.warning(
                f"Cached random X matrix shape {cached.shape} != expected {expected_shape}; recomputing."
            )
            random_x_matrix = None
    else:
        random_x_matrix = None

    if random_x_matrix is None:
        logging.info("Computing random X matrix in parallel...")
        random_x_matrix = compute_random_x_matrix_parallel(
            A,
            protein_domains,
            domain_pair_to_idx,
            domain_pairs,
            proteins,
            ddi_score,
            num_iterations=1000,
            max_workers=threads,
            seed=seed,
        )
        np.save(rxm_path, random_x_matrix)

    ppi_pairs = [(p1, p2) for p1, p2 in zip(ppi_df["protein_1"], ppi_df["protein_2"])]
    return (
        domain_pairs,
        random_x_matrix,
        ddi_dict,
        protein_domains,
        ppi_pairs,
        ddi_score,
    )  # type: ignore


def run_ddiparsimony(
    database: str,
    params_file: str,
    out_dir: str,
    test_splits: dict,
    threads: int = 1,
    seed: int = 42,
    ppi_score_cutoff: int | None = None,
):
    """Train once, score every test split.

    `test_splits` maps variant -> (split name, output predictions path). The
    reliability rate and pw-score cutoff are optimised on the train split once
    and reused for each test set.

    `ppi_score_cutoff` is the pipeline-level `params.ppi_score_cutoff`
    (`--ppi_score_cutoff` on the command line). When it is None the model JSON's
    own `parameter_list.ppi_score_cutoff` is used, and failing that
    DEFAULT_PPI_SCORE_CUTOFF -- KGIDDI resolves it identically, so both graph
    models always score the same interactome.
    """
    db_train = Path(os.path.join(database, "train.sqlite3"))
    check_file_existence(db_train)

    test_dbs = {
        variant: Path(os.path.join(database, f"{split}.sqlite3"))
        for variant, (split, _) in test_splits.items()
    }
    for db_test in test_dbs.values():
        check_file_existence(db_test)

    # Load json parameters
    with open(params_file) as f:
        params_json = json.load(f)
        # data_to_load = params_json.get("data", ["DDI", "PD", "PPI", "PGO"])
        reliability_rates = params_json.get("parameter_list", {}).get(
            "reliability_rate", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )
        pw_score_thresholds = params_json.get("parameter_list", {}).get(
            "pw_score_threshold", [0.7, 0.8, 0.9]
        )
        if ppi_score_cutoff is None:
            ppi_score_cutoff = params_json.get("parameter_list", {}).get(
                "ppi_score_cutoff", DEFAULT_PPI_SCORE_CUTOFF
            )

    opt_param = params_json.get("optimized", {})

    training = True
    best_r = 0.0
    best_cutoff = None
    if opt_param:
        training = False
        logging.info("Optimized parameters found, running in testing mode.")
        best_r = float(opt_param.get("reliability_rate", 0.9))
        best_cutoff = float(opt_param.get("pw_cutoff", 1.0))

    if training:
        logging.info(
            "Starting training to optimize reliability rate and pw-score cutoff..."
        )
        best_accuracy = -1
        best_fp = float("inf")

        # Preprocessing on training data
        (
            domain_pairs,
            random_x_matrix,
            ddi_dict,
            protein_domains,
            ppi_list,
            ddi_score,
        ) = preprocessing(db_train, Path(out_dir), threads, seed, ppi_score_cutoff)
        import gc

        domain_pair_to_idx = {pair: idx for idx, pair in enumerate(domain_pairs)}

        param_grid = [
            (
                r,
                ppi_list,
                protein_domains,
                domain_pair_to_idx,
                domain_pairs,
                random_x_matrix,
                ddi_dict,
                ddi_score,
                pw_score_thresholds,
                seed,
            )
            for r in reliability_rates
        ]

        train_logs = []
        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(
                tqdm(
                    executor.map(evaluate_reliability_train, param_grid),
                    total=len(param_grid),
                    desc="Reliability grid search",
                )
            )

        for idx, (r, cut, accuracy, fp_rate, pw_scores) in enumerate(results):
            logging.info(
                f"Reliability: {r}, Cutoff: {cut}, Accuracy: {accuracy}, FP-Rate: {fp_rate}, Score: {sum(pw_scores.values())}\n"
            )
            # Save log files for each reliability value
            suffix = f"train_{r}"
            # Save pw_scores
            with open(os.path.join(out_dir, f"pw_scores_{suffix}.json"), "w") as f:
                json.dump(
                    {f"{k[0]}_{k[1]}": v for k, v in pw_scores.items()}, f, indent=4
                )
            # Save observed_x_avg and avg_x_vec if available
            # For this, rerun compute_lp_score with return_x=True and get avg_x_vec, observed_x_avg
            # Same rng_token as the grid worker used for this `r`, so this
            # rerun reproduces that worker's observed_x_avg exactly. Under the
            # old global RNG the two diverged, and the avg_x_vec saved here did
            # not correspond to the pw_scores saved beside it.
            avg_score, n_runs, avg_x_vec = compute_lp_score(
                ppi_list,
                protein_domains,
                domain_pair_to_idx,
                domain_pairs,
                ddi_score,
                reliability=r,
                num_runs=200,
                return_x=True,
                seed=seed,
                rng_token=f"train_grid:{r}",
            )
            if avg_x_vec is not None:
                np.save(os.path.join(out_dir, f"avg_x_vec_{suffix}.npy"), avg_x_vec)
                observed_x_avg = (
                    avg_x_vec  # For training, avg_x_vec is the observed average
                )
                np.save(
                    os.path.join(out_dir, f"observed_x_avg_{suffix}.npy"),
                    observed_x_avg,
                )
            train_logs.append(
                {
                    "reliability": r,
                    "cutoff": cut,
                    "accuracy": accuracy,
                    "fp_rate": fp_rate,
                }
            )
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and fp_rate < best_fp
            ):
                best_r = r
                best_cutoff = cut
                best_accuracy = accuracy
                best_fp = fp_rate
        # Save train logs summary
        with open(
            os.path.join(out_dir, "ddiparsimony_train_log_summary.json"), "w"
        ) as f:
            json.dump(train_logs, f, indent=4)
        # Delete large training objects
        del (
            domain_pairs,
            random_x_matrix,
            ddi_dict,
            protein_domains,
            ppi_list,
            ddi_score,
            domain_pair_to_idx,
            param_grid,
            train_logs,
            results,
            pw_scores,
        )
        gc.collect()

    if best_cutoff is None:
        best_cutoff = 1.0  # default when no optimisation ran

    for variant, (split, out_predictions) in test_splits.items():
        logging.info(
            f"Testing split {split} (variant {variant}) with best reliability: "
            f"{best_r} and pw-score cutoff: {best_cutoff}"
        )
        score_test_split(
            test_dbs[variant],
            out_predictions,
            variant,
            best_r,
            best_cutoff,
            params_json,
            out_dir,
            threads,
            seed,
            ppi_score_cutoff,
        )


def score_test_split(
    db_test,
    out_predictions,
    variant,
    best_r,
    best_cutoff,
    params_json,
    out_dir,
    threads,
    seed=42,
    ppi_score_cutoff=DEFAULT_PPI_SCORE_CUTOFF,
):
    """Score one test split with the reliability/cutoff chosen on the train split."""
    # Preprocessing on test data
    domain_pairs, random_x_matrix, ddi_dict, protein_domains, ppi_list, ddi_score = (
        preprocessing(db_test, Path(out_dir), threads, seed, ppi_score_cutoff)
    )
    import gc

    domain_pair_to_idx = {pair: idx for idx, pair in enumerate(domain_pairs)}

    # Call function evaluate_reliability_test
    eval_rel_args = (
        best_r,
        best_cutoff,
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        random_x_matrix,
        ddi_dict,
        ddi_score,
        seed,
        variant,
    )
    rel, cut, accuracy, fp_rate, pw_scores = evaluate_reliability_test(eval_rel_args)
    logging.info(
        f"Test Results - Reliability: {best_r}, Cutoff: {cut}, Accuracy: {accuracy}, FP-Rate: {fp_rate}\n"
    )

    # Save test log files
    suffix = variant
    with open(os.path.join(out_dir, f"pw_scores_{suffix}.json"), "w") as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in pw_scores.items()}, f, indent=4)
    # Same token as evaluate_reliability_test above, for the same reason as on
    # the train side: the saved avg_x_vec must be the one the pw_scores came
    # from.
    avg_score, n_runs, avg_x_vec = compute_lp_score(
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        reliability=best_r,
        num_runs=200,
        return_x=True,
        seed=seed,
        rng_token=f"test_eval:{variant}:{best_r}",
    )
    if avg_x_vec is not None:
        np.save(os.path.join(out_dir, f"avg_x_vec_{suffix}.npy"), avg_x_vec)
        observed_x_avg = avg_x_vec
        np.save(os.path.join(out_dir, f"observed_x_avg_{suffix}.npy"), observed_x_avg)

    optimized = {
        "reliability_rate": best_r,
        "pw_cutoff": best_cutoff,
        "accuracy": accuracy,
        "fp_rate": fp_rate,
    }

    with open(os.path.join(out_dir, f"ddiparsimony_{variant}.json"), "w") as jf:
        json.dump(dict(params_json, optimized=optimized), jf, indent=4)

    # Prepare output
    output_rows = []
    for pair in domain_pairs:
        # Every DDI row in a split database belongs to that split by
        # construction (domainsplit's SUBSET_SPLIT_DB) -- nothing to filter.
        d1, d2 = pair
        observed, _ = ddi_dict.get(pair, (0, 0))
        if best_cutoff is not None:
            predicted = int(pw_scores.get(pair, 0) <= best_cutoff)
        else:
            predicted = 0  # or handle as appropriate for your use case
        # `pair` keys every lookup above; only the emitted orientation is
        # canonicalised, so every predictions file in the run agrees.
        out_a, out_b = canonical_pair(d1, d2)
        output_rows.append(
            {
                "domain_a": out_a,
                "domain_b": out_b,
                "true_interaction": observed,
                "predicted_interaction": predicted,
                "predicted_probability": 1 - pw_scores.get(pair, 0),
            }
        )

    out_df = pd.DataFrame(output_rows)
    if "true_interaction" in out_df.columns:
        out_df["true_interaction"] = out_df["true_interaction"].astype("int8")
    if "predicted_interaction" in out_df.columns:
        out_df["predicted_interaction"] = out_df["predicted_interaction"].astype("int8")
    if "predicted_probability" in out_df.columns:
        out_df["predicted_probability"] = out_df["predicted_probability"].astype("float32")
    if str(out_predictions).endswith(".csv"):
        out_df.to_csv(out_predictions, index=False)
    else:
        out_df.to_parquet(out_predictions, index=False, compression="zstd")
    del out_df
    # Delete large test objects
    del (
        domain_pairs,
        random_x_matrix,
        ddi_dict,
        protein_domains,
        ppi_list,
        ddi_score,
        domain_pair_to_idx,
        eval_rel_args,
        pw_scores,
        avg_score,
        n_runs,
        avg_x_vec,
        output_rows,
    )
    gc.collect()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "--database",
        required=True,
        help="path to database directory, containing sqlite3 files",
    )
    parser.add_argument("--params", required=True, help="JSON file with parameters")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--out_predictions", required=False, help="Output predictions file path"
    )
    parser.add_argument(
        "--test_split", default="test", help="Name of the test split to score"
    )
    args = parser.parse_args()
    variant = (
        args.test_split[len("test_"):]
        if args.test_split.startswith("test_")
        else args.test_split
    )
    run_ddiparsimony(
        args.database,
        args.params,
        args.out_dir,
        {variant: (args.test_split, args.out_predictions)},
    )
