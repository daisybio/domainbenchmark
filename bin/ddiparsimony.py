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
from load_data_gm import load_ppi, load_pd_mapping, load_ddi, check_file_existence


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
    ) = args
    avg_score, n_runs, observed_x_avg = compute_lp_score(
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        reliability=r,
        num_runs=200,
        return_x=True,
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
    ) = args
    avg_score, n_runs, observed_x_avg = compute_lp_score(
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        reliability=r,
        num_runs=200,
        return_x=True,
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
    db_path: Path, output_dir: Path, threads: int = 1
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

    # Reduce ppi_df to interactions with score > 900
    logging.info("Filtering PPI data for high-confidence interactions...")
    ppi_df = ppi_df[ppi_df["score"] > 900].reset_index(drop=True)

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
    database: str, params_file: str, out_dir: str, out_predictions: str, threads: int = 1
):
    db_train = Path(os.path.join(database, "train.sqlite3"))
    db_test = Path(os.path.join(database, "test.sqlite3"))
    check_file_existence(db_train)
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
        ) = preprocessing(db_train, Path(out_dir), threads)
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
            avg_score, n_runs, avg_x_vec = compute_lp_score(
                ppi_list,
                protein_domains,
                domain_pair_to_idx,
                domain_pairs,
                ddi_score,
                reliability=r,
                num_runs=200,
                return_x=True,
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

    logging.info(
        f"Testing with best reliability: {best_r} and pw-score cutoff: {best_cutoff}"
    )
    # Preprocessing on test data
    domain_pairs, random_x_matrix, ddi_dict, protein_domains, ppi_list, ddi_score = (
        preprocessing(db_test, Path(out_dir), threads)
    )
    import gc

    domain_pair_to_idx = {pair: idx for idx, pair in enumerate(domain_pairs)}

    # Call function evaluate_reliability_test
    if best_cutoff is None:
        best_cutoff = (
            1.0  # or another default float value appropriate for your use case
        )
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
    )
    rel, cut, accuracy, fp_rate, pw_scores = evaluate_reliability_test(eval_rel_args)
    logging.info(
        f"Test Results - Reliability: {best_r}, Cutoff: {cut}, Accuracy: {accuracy}, FP-Rate: {fp_rate}\n"
    )

    # Save test log files
    suffix = "test"
    with open(os.path.join(out_dir, f"pw_scores_{suffix}.json"), "w") as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in pw_scores.items()}, f, indent=4)
    avg_score, n_runs, avg_x_vec = compute_lp_score(
        ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        reliability=best_r,
        num_runs=200,
        return_x=True,
    )
    if avg_x_vec is not None:
        np.save(os.path.join(out_dir, f"avg_x_vec_{suffix}.npy"), avg_x_vec)
        observed_x_avg = avg_x_vec
        np.save(os.path.join(out_dir, f"observed_x_avg_{suffix}.npy"), observed_x_avg)

    params_json["optimized"] = {
        "reliability_rate": best_r,
        "pw_cutoff": best_cutoff,
        "accuracy": accuracy,
        "fp_rate": fp_rate,
    }

    with open(os.path.join(out_dir, "ddiparsimony.json"), "w") as jf:
        json.dump(params_json, jf, indent=4)

    # Prepare output
    output_rows = []
    for pair in domain_pairs:
        # Check if eval_relevant
        d1, d2 = pair
        observed, eval_relevant = ddi_dict.get(pair, (0, 0))
        if not eval_relevant:
            continue
        if best_cutoff is not None:
            predicted = int(pw_scores.get(pair, 0) <= best_cutoff)
        else:
            predicted = 0  # or handle as appropriate for your use case
        output_rows.append(
            {
                "domain_a": d1,
                "domain_b": d2,
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
    args = parser.parse_args()
    run_ddiparsimony(args.database, args.params, args.out_dir, args.out_predictions)
