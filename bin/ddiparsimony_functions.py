#!/usr/bin/env python3
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.optimize as opt
from sklearn import metrics


def build_domain_pairs(ppi_list, protein_domains, ddi_dict):
    domain_pairs = set()
    for p1, p2 in ppi_list:
        for d1 in protein_domains.get(p1, []):
            for d2 in protein_domains.get(p2, []):
                domain_pairs.add((d1, d2, ddi_dict.get((d1, d2), 1)))
    return list(domain_pairs)


def count_witnesses(ppi_list, protein_domains, domain_pairs):
    witnesses = {pair: 0 for pair in domain_pairs}
    for p1, p2 in ppi_list:
        for d1 in protein_domains.get(p1, []):
            for d2 in protein_domains.get(p2, []):
                pair = (d1, d2)
                if pair in witnesses:
                    witnesses[pair] += 1
    return witnesses


def compute_fp_rate(y_pred, y_true):
    false_positives = sum(
        (pred == 1 and true == 0) for pred, true in zip(y_pred, y_true)
    )
    return false_positives / len(y_pred) if y_pred else 0


def find_best_pw_cutoff_with_candidates(pw_scores, labels, candidate_cutoffs):
    """Finds the pw-score cutoff that maximizes accuracy and minimizes FP-rate score, only over candidate_cutoffs."""
    best_cut = None
    best_accuracy = -1
    best_fp = float("inf")
    for cut in sorted(candidate_cutoffs):
        y_pred = [int(pw_scores[pair] <= cut) for pair in pw_scores]
        y_true = [labels[pair][0] == 1 for pair in pw_scores]
        accuracy = metrics.accuracy_score(y_true, y_pred)
        fp_rate = compute_fp_rate(y_pred, y_true)
        print(f"Cutoff: {cut}, Accuracy: {accuracy:.4f}, FP-rate: {fp_rate:.4f}")
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and fp_rate < best_fp
        ):
            best_accuracy = accuracy
            best_fp = fp_rate
            best_cut = cut
    return best_cut, best_accuracy, best_fp


def compute_p_values_from_matrix(random_x_matrix, observed_x_avg, domain_pairs):
    p_values = {}
    for idx, pair in enumerate(domain_pairs):
        observed_val = observed_x_avg[idx]
        pval = np.mean(random_x_matrix[:, idx] >= observed_val)
        p_values[pair] = pval
    return p_values


def association_score_ddi(ddi_dict, ppi_list, protein_domains):
    ppi_set = set(ppi_list)
    ppi_set |= set((b, a) for (a, b) in ppi_list)
    from collections import defaultdict

    domain_to_proteins = defaultdict(set)
    for protein, domains in protein_domains.items():
        for d in domains:
            domain_to_proteins[d].add(protein)
    domain_counts = defaultdict(int)
    for domains in protein_domains.values():
        for d in domains:
            domain_counts[d] += 1
    association = {}
    for (d1, d2), dtype in ddi_dict.items():
        proteins1 = domain_to_proteins.get(d1, set())
        proteins2 = domain_to_proteins.get(d2, set())
        ddi_count = 0
        for p1 in proteins1:
            for p2 in proteins2:
                if p1 != p2 and ((p1, p2) in ppi_set):
                    ddi_count += 1
        denom = domain_counts[d1] + domain_counts[d2]
        association[(d1, d2)] = ddi_count / denom if denom > 0 else 0.0
    return association


def randomization(network, num_iterations=1000):
    randomized_networks = []
    for _ in range(num_iterations):
        G = nx.from_numpy_array(network)
        nswap = int(G.number_of_edges() * 0.8)
        max_tries = nswap * 20
        try:
            randomized_G = nx.double_edge_swap(G, nswap=nswap, max_tries=max_tries)
            randomized_networks.append(nx.to_numpy_array(randomized_G))
        except nx.NetworkXAlgorithmError:
            continue
    return randomized_networks


def log_msg(msg, log_file=None):
    if log_file is not None:
        with open(log_file, "a") as lf:
            lf.write(msg + "\n")
    else:
        print(msg)


def compute_lp_score(
    ppi_list,
    protein_domains,
    domain_pair_to_idx,
    domain_pairs,
    ddi_score,
    reliability,
    num_runs,
    return_x=False,
    log_file="log_lp_scores.txt",
):
    lp_scores = []
    x_vecs = []

    for i in range(num_runs):
        if (i + 1) % 100 == 0:
            log_msg(f"Run {i + 1}/{num_runs}...", log_file=log_file)
        # Build constraint matrix for LP relaxation
        A_ub = []
        b_ub = []
        constraint_count = 0
        for p1, p2 in ppi_list:
            if np.random.rand() < reliability:
                row = np.zeros(len(domain_pairs))
                for d1 in protein_domains.get(p1, []):
                    for d2 in protein_domains.get(p2, []):
                        idx = domain_pair_to_idx.get((d1, d2), None)
                        if idx is not None:
                            row[idx] = 1
                # Ensure each PPI must be explained by at least one DDI
                if np.sum(row) > 0:
                    A_ub.append(-row)
                    b_ub.append(-1)
                    constraint_count += 1
        if constraint_count == 0:
            log_msg(
                f"[LP] Skipping run {i + 1}: no constraints added.", log_file=log_file
            )
            continue
        # Diagnostics: log number of constraints added per run
        log_msg(
            f"[LP] Run {i + 1}: {constraint_count} constraints added.",
            log_file=log_file,
        )
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c = np.ones(len(domain_pairs))
        bounds = [(0, 1)] * len(domain_pairs)
        res = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            lp_scores.append(np.float32(res.fun))
            if return_x:
                x_vecs.append(res.x.astype(np.float32))
    if lp_scores:
        avg_score = np.mean(lp_scores)
        if return_x:
            x_vecs = np.array(x_vecs)
            avg_x_vec = np.mean(x_vecs, axis=0)
            return avg_score, len(lp_scores), avg_x_vec
        else:
            return avg_score, len(lp_scores), None
    else:
        zero_vec = np.zeros(len(domain_pairs), dtype=np.int8)
        return None, 0, zero_vec


# --- Parallelized version ---
def process_one_random(
    run_idx, A, protein_domains, domain_pair_to_idx, domain_pairs, proteins, ddi_score
):
    G = (
        nx.from_scipy_sparse_array(A)
        if hasattr(nx, "from_scipy_sparse_array")
        else nx.from_scipy_sparse_matrix(A)
    )  # type: ignore
    nswap = int(G.number_of_edges() * 0.8)
    max_tries = nswap * 20
    try:
        randomized_G = nx.double_edge_swap(G, nswap=nswap, max_tries=max_tries)
        rand_net = (
            nx.to_scipy_sparse_array(randomized_G)
            if hasattr(nx, "to_scipy_sparse_array")
            else nx.to_scipy_sparse_matrix(randomized_G)
        )  # type: ignore
    except nx.NetworkXAlgorithmError:
        rand_net = A.copy()

    G_rand = (
        nx.from_scipy_sparse_array(rand_net)
        if hasattr(nx, "from_scipy_sparse_array")
        else nx.from_scipy_sparse_matrix(rand_net)
    )  # type: ignore
    rand_ppi_list = [(proteins[i], proteins[j]) for i, j in G_rand.edges() if i < j]
    _, _, x_vec = compute_lp_score(
        rand_ppi_list,
        protein_domains,
        domain_pair_to_idx,
        domain_pairs,
        ddi_score,
        1,
        1,
        True,
    )
    return x_vec


def compute_random_x_matrix_parallel(
    A,
    protein_domains,
    domain_pair_to_idx,
    domain_pairs,
    proteins,
    ddi_score,
    num_iterations=1000,
    max_workers=1,
):
    random_x_matrix = np.zeros((num_iterations, len(domain_pairs)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one_random,
                run_idx,
                A,
                protein_domains,
                domain_pair_to_idx,
                domain_pairs,
                proteins,
                ddi_score,
            )
            for run_idx in range(num_iterations)
        ]
        for run_idx, future in enumerate(as_completed(futures)):
            x_vec = future.result()
            if x_vec is not None:
                random_x_matrix[run_idx, :] = x_vec
            else:
                random_x_matrix[run_idx, :] = 0
    return random_x_matrix
