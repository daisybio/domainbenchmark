#!/usr/bin/env python3
import multiprocessing as mp

import networkx as nx
import numpy as np
import scipy.optimize as opt
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import coo_matrix
from sklearn import metrics

from determinism import derive_seed


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


def randomization(network, num_iterations=1000, *, seed):
    """Unused today; kept seeded so it cannot reintroduce an unseeded RNG.

    `seed` is keyword-only and required for the same reason as in
    `compute_lp_score`: `double_edge_swap` without one randomises differently
    on every run.
    """
    randomized_networks = []
    for i in range(num_iterations):
        G = nx.from_numpy_array(network)
        nswap = int(G.number_of_edges() * 0.8)
        max_tries = nswap * 20
        try:
            randomized_G = nx.double_edge_swap(
                G, nswap=nswap, max_tries=max_tries, seed=derive_seed(seed, i)
            )
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
    *,
    seed,
    rng_token,
):
    """LP-relaxation parsimony score over `num_runs` reliability resamples.

    `seed`/`rng_token` are keyword-only and have no defaults on purpose. This
    function draws the reliability mask, and it used to draw it from the
    *global* numpy RNG -- inside ProcessPoolExecutor workers, which inherit no
    RNG state under spawn and inherit a shared one under fork. The result was
    reproducible only by accident (fork, plus every worker happening to run at
    most one task). A missing seed now raises TypeError at the call site rather
    than silently randomising a benchmark.

    `rng_token` names *what* is being scored -- the reliability rate, the test
    variant, the randomisation index. Two calls meant to reproduce each other
    must pass the same token; two calls meant to be independent must not.
    """
    # derive_seed keys on the token, not on call order, so this run's mask does
    # not depend on which worker picked the task up or how many ran before it.
    rng = np.random.default_rng(derive_seed(seed, "lp_score", rng_token))
    lp_scores = []
    x_vecs = []
    n_pairs = len(domain_pairs)
    # Hoisted out of the loop: the objective is identical every run, and this
    # used to allocate a fresh length-n_pairs vector per iteration.
    c = np.ones(n_pairs)

    for i in range(num_runs):
        if (i + 1) % 100 == 0:
            log_msg(f"Run {i + 1}/{num_runs}...", log_file=log_file)
        # Build the LP relaxation's constraint matrix as sparse COO triplets.
        # It used to be a dense `np.zeros(len(domain_pairs))` row per kept PPI,
        # stacked with np.array into an (n_constraints x n_pairs) float64
        # matrix -- while each row holds at most |domains(p1)| * |domains(p2)|
        # nonzeros. On a whole-proteome interactome that dense matrix is tens
        # of GB, and `compute_random_x_matrix_parallel` has every pool worker
        # build one concurrently: that is what got the workers OOM-killed and
        # surfaced as `BrokenProcessPool`. HiGHS converts its input to sparse
        # CSC internally either way, so the LP posed here is unchanged.
        rows = []
        cols = []
        constraint_count = 0
        for p1, p2 in ppi_list:
            if rng.random() < reliability:
                # A set, because the dense version *assigned* `row[idx] = 1`
                # rather than accumulating -- two domain combinations mapping
                # to the same pair contributed a single entry, not two.
                idxs = set()
                for d1 in protein_domains.get(p1, []):
                    for d2 in protein_domains.get(p2, []):
                        idx = domain_pair_to_idx.get((d1, d2), None)
                        if idx is not None:
                            idxs.add(idx)
                # Ensure each PPI must be explained by at least one DDI
                if idxs:
                    # Sorted so the triplet order is a function of the data and
                    # not of set iteration order.
                    for idx in sorted(idxs):
                        rows.append(constraint_count)
                        cols.append(idx)
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
        # -1 coefficients against b_ub = -1 is the same "at least one DDI
        # explains this PPI" inequality the dense `-row` / `-1` pair encoded.
        A_ub = coo_matrix(
            (np.full(len(rows), -1.0), (rows, cols)),
            shape=(constraint_count, n_pairs),
        ).tocsr()
        b_ub = np.full(constraint_count, -1.0)
        # bounds=(0, 1) broadcasts to every decision variable; the old form
        # materialised n_pairs Python tuples on every run.
        res = opt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs")
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
    run_idx,
    A,
    protein_domains,
    domain_pair_to_idx,
    domain_pairs,
    proteins,
    ddi_score,
    seed=42,
):
    G = (
        nx.from_scipy_sparse_array(A)
        if hasattr(nx, "from_scipy_sparse_array")
        else nx.from_scipy_sparse_matrix(A)
    )  # type: ignore
    nswap = int(G.number_of_edges() * 0.8)
    max_tries = nswap * 20
    try:
        # seed: this runs in a ProcessPoolExecutor worker, which inherits no RNG
        # state, so an unseeded double_edge_swap randomised the network
        # differently on every run. derive_seed keys on run_idx, so iteration
        # `i` gets the same rewiring whatever worker picks it up.
        randomized_G = nx.double_edge_swap(
            G, nswap=nswap, max_tries=max_tries, seed=derive_seed(seed, run_idx)
        )
        rand_net = (
            nx.to_scipy_sparse_array(randomized_G)
            if hasattr(nx, "to_scipy_sparse_array")
            else nx.to_scipy_sparse_matrix(randomized_G)
        )  # type: ignore
    except nx.NetworkXException:
        # Was NetworkXAlgorithmError only, which does not cover the
        # NetworkXError double_edge_swap raises for a graph with fewer than
        # four nodes -- a degenerate input killed the whole task instead of
        # taking the intended "rewiring impossible, use the original network"
        # fallback. Both derive from NetworkXException.
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
        reliability=1,
        num_runs=1,
        return_x=True,
        # reliability=1 means the mask keeps every PPI, so nothing actually
        # varies with the stream here -- the token is still keyed on run_idx so
        # that stays true if the reliability ever becomes configurable.
        seed=seed,
        rng_token=f"random_x:{run_idx}",
    )
    return x_vec


# Shared read-only state for the random-X pool workers, populated once per
# worker by `_init_random_x_worker`. See `compute_random_x_matrix_parallel`.
_RANDOM_X_ARGS: dict = {}


def _init_random_x_worker(
    A, protein_domains, domain_pair_to_idx, domain_pairs, proteins, ddi_score, seed
):
    _RANDOM_X_ARGS.update(
        A=A,
        protein_domains=protein_domains,
        domain_pair_to_idx=domain_pair_to_idx,
        domain_pairs=domain_pairs,
        proteins=proteins,
        ddi_score=ddi_score,
        seed=seed,
    )


def _random_x_worker(run_idx):
    return process_one_random(run_idx, **_RANDOM_X_ARGS)


def compute_random_x_matrix_parallel(
    A,
    protein_domains,
    domain_pair_to_idx,
    domain_pairs,
    proteins,
    ddi_score,
    num_iterations=1000,
    max_workers=1,
    seed=42,
):
    # float32, matching what the workers actually return: compute_lp_score
    # casts every x vector with `.astype(np.float32)`, so float64 storage
    # doubled a (num_iterations x n_pairs) array without holding one extra bit
    # of precision. Comparisons in compute_p_values_from_matrix are unchanged
    # -- the stored values were already float32-exact.
    random_x_matrix = np.zeros((num_iterations, len(domain_pairs)), dtype=np.float32)
    # spawn, for the reason kgiddi._MP_CTX documents: forked workers inherit
    # the parent's whole post-preprocessing image copy-on-write, and their RSS
    # climbs toward the parent's as soon as they touch those objects. Workers
    # here only read, so a clean start plus pickled args is strictly cheaper.
    # `process_one_random` is called with reliability=1, so it draws nothing
    # from the global numpy RNG (`np.random.rand() < 1` always holds) and the
    # start method carries no reproducibility consequence.
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_random_x_worker,
        initargs=(
            A,
            protein_domains,
            domain_pair_to_idx,
            domain_pairs,
            proteins,
            ddi_score,
            seed,
        ),
    ) as executor:
        # Only run_idx is submitted. Every one of the `num_iterations`
        # submissions used to carry the full argument tuple -- adjacency
        # matrix, domain maps, pair list -- so the identical objects were
        # pickled 1000 times over. The initializer ships them once per worker.
        futures = [
            executor.submit(_random_x_worker, run_idx)
            for run_idx in range(num_iterations)
        ]
        # Submission order, not as_completed: the previous loop numbered rows by
        # the order results came *back*, so iteration i's vector landed in a row
        # that depended on worker scheduling. Every row of the matrix moved
        # between runs even though the set of rows was the same. All futures
        # still run concurrently; only the collection is ordered.
        for run_idx, future in enumerate(futures):
            x_vec = future.result()
            if x_vec is not None:
                random_x_matrix[run_idx, :] = x_vec
            else:
                random_x_matrix[run_idx, :] = 0
    return random_x_matrix
