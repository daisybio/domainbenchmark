#! /usr/bin/env python3

import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from goatools.obo_parser import GODag
import logging


def load_go_graph(go_obo_file):
    go_dag = GODag(go_obo_file, optional_attrs=set("relationship"))
    go_graph = {
        go_term.id: go_term
        for go_term in go_dag.values()
        if go_term.namespace == "molecular_function"
    }
    G = nx.DiGraph()
    for go_id, term in go_graph.items():
        for parent in term.parents:
            G.add_edge(parent.id, go_id, depth=term.depth, level=term.level)
    return G


def extract_go_guided_ddi_subgraphs(
    go_domain_mapping, ddi_network_edges
) -> dict[str, nx.Graph]:
    """
    For each GO term, extract the DDI subgraph induced by domains from p1 and p2,
    but only include edges between d1 and d2 (not within d1 or within d2).
    Returns a dict: {go_term: nx.Graph()}
    """
    go_subgraph = {}
    for go_term, domain_lists in go_domain_mapping.items():
        if not isinstance(domain_lists, list) or len(domain_lists) != 2:
            continue  # skip malformed entries
        d1, d2 = set(domain_lists[0]), set(domain_lists[1])
        subgraph = nx.Graph()
        for dom1 in d1:
            for dom2 in d2:
                if (dom1, dom2) in ddi_network_edges or (
                    dom2,
                    dom1,
                ) in ddi_network_edges:
                    subgraph.add_edge(dom1, dom2)
        if subgraph.number_of_edges() > 0:
            go_subgraph[go_term] = subgraph
    return go_subgraph


# --- Biclustering algorithm ---


def approx_bimax(A, b=0.5):
    m_rows, n_cols = A.shape
    found_biclusters = set()
    # State tracker for repeated (I_row, J) pairs
    state_counter = {}
    max_repeats = 50  # Limit for repeated states before aborting

    def conquer_with_state(A, I_row, J, b, found_biclusters, depth=0):
        # Track state as tuple of sorted I_row and J
        state = (tuple(sorted(I_row)), tuple(sorted(J)))
        state_counter[state] = state_counter.get(state, 0) + 1
        if state_counter[state] > max_repeats:
            logging.warning(
                f"Biclustering abort: repeated state (I_row, J) {state} seen > {max_repeats} times at recursion depth {depth}. Returning no biclusters for this submatrix."
            )
            return set()
        if len(I_row) < 3 or len(J) < 3:
            return set()
        submatrix = A[np.ix_(np.array(I_row, dtype=int), np.array(J, dtype=int))]
        row_means = submatrix.mean(axis=1)
        col_means = submatrix.mean(axis=0)
        if np.all(row_means > b) and np.all(col_means > b):
            bicluster = (tuple(sorted(I_row)), tuple(sorted(J)))
            if bicluster not in found_biclusters:
                found_biclusters.add(bicluster)
                return {bicluster}
            else:
                return set()
        Iu, Iv, Iw, Ju, Jv = divide(A, I_row, J)
        result = set()
        if len(Iu) != 0:
            result |= conquer_with_state(
                A, list(np.union1d(Iu, Iw)), Ju, b, found_biclusters, depth + 1
            )
        if len(Iv) != 0 and len(Iw) == 0:
            result |= conquer_with_state(A, Iv, Jv, b, found_biclusters, depth + 1)
        elif len(Iw) != 0:
            result |= conquer_with_state(
                A,
                list(np.union1d(Iw, Iv)),
                list(np.union1d(Ju, Jv)),
                b,
                found_biclusters,
                depth + 1,
            )
        return result

    return conquer_with_state(
        A, list(range(m_rows)), list(range(n_cols)), b, found_biclusters
    )


def conquer(A, I_row, J, b, found_biclusters):
    if len(I_row) < 3 or len(J) < 3:
        return set()
    submatrix = A[np.ix_(np.array(I_row, dtype=int), np.array(J, dtype=int))]
    row_means = submatrix.mean(axis=1)
    col_means = submatrix.mean(axis=0)
    if np.all(row_means > b) and np.all(col_means > b):
        bicluster = (tuple(sorted(I_row)), tuple(sorted(J)))
        if bicluster not in found_biclusters:
            found_biclusters.add(bicluster)
            return {bicluster}
        else:
            return set()
    Iu, Iv, Iw, Ju, Jv = divide(A, I_row, J)
    result = set()
    if len(Iu) != 0:
        result |= conquer(A, list(np.union1d(Iu, Iw)), Ju, b, found_biclusters)
    if len(Iv) != 0 and len(Iw) == 0:
        result |= conquer(A, Iv, Jv, b, found_biclusters)
    elif len(Iw) != 0:
        result |= conquer(
            A, list(np.union1d(Iw, Iv)), list(np.union1d(Ju, Jv)), b, found_biclusters
        )
    return result


def divide(A, I_row, J):
    I_prime = reduce_rows(A, I_row, J)
    if not I_prime:
        return [], [], [], J, []
    submatrix = A[np.ix_(np.array(I_prime, dtype=int), np.array(J, dtype=int))]
    row_sums = submatrix.sum(axis=1)
    # Find i_candidate: first row with 0 < sum < len(J)
    mask = (row_sums > 0) & (row_sums < len(J))
    i_candidate_idx = np.where(mask)[0]
    if i_candidate_idx.size > 0:
        i_candidate = I_prime[i_candidate_idx[0]]
        row = A[np.array(i_candidate, dtype=int), np.array(J, dtype=int)]
        Ju = [J[idx] for idx in np.where(row == 1)[0]]
        Jv = [J[idx] for idx in np.where(row == 0)[0]]
    else:
        Ju = J
        Jv = []
    # For each row in I_prime, get the set of columns where A[i, j] == 1
    bool_matrix = submatrix == 1
    Ju_set = (
        set(np.where(row == 1)[0]) if i_candidate_idx.size > 0 else set(range(len(J)))
    )
    Jv_set = set(np.where(row == 0)[0]) if i_candidate_idx.size > 0 else set()
    Iu, Iv, Iw = [], [], []
    for idx, i in enumerate(I_prime):
        ones = set(np.where(bool_matrix[idx])[0])
        if ones and ones.issubset(Ju_set):
            Iu.append(i)
        elif ones and ones.issubset(Jv_set):
            Iv.append(i)
        else:
            Iw.append(i)
    # Map Ju/Jv indices back to original J values
    Ju = [J[j] for j in Ju_set]
    Jv = [J[j] for j in Jv_set]
    return Iu, Iv, Iw, Ju, Jv


def reduce_rows(A, I_row, J):
    submatrix = A[np.ix_(np.array(I_row, dtype=int), np.array(J, dtype=int))]
    row_sums = submatrix.sum(axis=1)
    return [I_row[i] for i in np.where(row_sums > 0)[0]]


# ---- Performance Measurement ----


def fold_enrichment(predicted_ddis, known_ddis, all_domains):
    n = len(predicted_ddis)
    k = len([ddi for ddi in predicted_ddis if ddi in known_ddis])
    N = len(all_domains) ** 2
    K = len(known_ddis)
    if n == 0 or N == 0 or K == 0:
        return 0
    return (k / n) / (K / N)


def compute_group_ddi_chi2(
    connected_components: dict,
    ppi_interactions: set[tuple[str, str]],
    protein_domain_mapping: dict[str, set[str]],
):
    """Compute chi2 for DDI patterns in each group.

    The contingency table for (group g, DDI d) is

        A = PPIs in g whose domain cross-product contains d
        B = PPIs outside g whose cross-product contains d
        C = PPIs in g without d          = |g| - A
        D = PPIs outside g without d     = (N - |g|) - B

    `B` used to be counted by scanning the entire interactome once **per
    (group, DDI)** -- rebuilding a list of the scanned PPI's domain pairs at
    every step, and rebuilding `outside_ppis` from scratch for every DDI on top
    of that. At `external_test` scale (1.78 M PPIs, ~120 k DDIs) that is upwards
    of 1e11 Python iterations.

    It is one precomputation instead. `total_ddi_counts[d]` counts, over the
    whole deduplicated interactome, the PPIs whose cross-product contains `d`.
    The union-find clusters partition that same set, so every PPI carrying `d`
    is either in `g` or outside it, giving `B = total[d] - A`; and every group is
    a subset of it, giving `|outside| = N - |g|`. Both are exact, not
    approximations.

    The result is bitwise identical to the scan: A, B, C, D are counts, and for
    any realistic `N` every product and sum in `numerator`/`denominator` stays
    below 2**53, so the float64 arithmetic is exact and only the final division
    rounds -- on identical inputs.

    Two things here look incidental and are not:

    * `for ddi in group_ddi_set` iterates a *set*. `score_test_split` builds
      `chi2_scores` as a dict comprehension, so for a DDI appearing in two
      groups the last one written wins. Reordering this loop would silently
      change the scores; `PYTHONHASHSEED=0` makes the current order
      deterministic, so it is left exactly as it was.
    * `missing_key_counter` still counts only in-group PPIs with an empty
      domain set, as before -- it is a diagnostic about the groups, not about
      the interactome.
    """
    N = len(ppi_interactions)

    # One pass over the whole interactome. `.get`, not `[...]`: the caller hands
    # us a defaultdict(set), and subscripting it inserted an empty set for every
    # protein without domains.
    total_ddi_counts = Counter()
    for p1, p2 in ppi_interactions:
        d1s = protein_domain_mapping.get(p1)
        d2s = protein_domain_mapping.get(p2)
        if not d1s or not d2s:
            continue
        for d1 in d1s:
            for d2 in d2s:
                total_ddi_counts[(d1, d2)] += 1

    group_ddi_chi2 = []
    missing_key_counter = 0
    for group_name, group in connected_components.items():
        group = set(group)
        group_ddis = Counter()
        group_ddi_set = set()
        for p1, p2 in group:
            d1s = protein_domain_mapping.get(p1, set())
            d2s = protein_domain_mapping.get(p2, set())
            if not d1s or not d2s:
                missing_key_counter += 1
                continue
            for d1 in d1s:
                for d2 in d2s:
                    group_ddis[(d1, d2)] += 1
                    group_ddi_set.add((d1, d2))
        n_outside = N - len(group)
        # For each DDI pattern in the group, compute chi-squared
        for ddi in group_ddi_set:
            A = group_ddis[ddi]
            B = total_ddi_counts[ddi] - A
            C = len(group) - A
            D = n_outside - B
            # Chi-squared calculation
            numerator = N * (A * D - C * B) ** 2
            denominator = (A + C) * (B + D) * (A + B) * (C + D)
            chi2 = numerator / denominator if denominator != 0 else 0
            group_ddi_chi2.append({"group_name": group_name, "ddi": ddi, "chi2": chi2})
    print(f"Missing keys in protein_domain_mapping: {missing_key_counter}")
    return group_ddi_chi2


def select_best_ddis_per_group(
    connected_components: dict, group_ddi_chi2: list[dict], chi_square_cutoff: float
) -> dict:
    """
    Select best DDIs per group using a quantile cutoff (chi_square_cutoff between 0 and 1).
    If chi_square_cutoff=0.75, selects top 25% by chi2. If 1.0, selects only max chi2, if 0.0, selects all.
    """
    # Bucketed once instead of a full scan of `group_ddi_chi2` per group, which
    # was O(groups x entries). Appending preserves each group's original
    # within-group order, and the outer loop still walks
    # `connected_components` in its own order, so the output is unchanged.
    entries_by_group = defaultdict(list)
    for entry in group_ddi_chi2:
        entries_by_group[entry["group_name"]].append(entry)

    best_ddis_per_group = defaultdict(list)
    for group_name in connected_components.keys():
        group_ddis = entries_by_group.get(group_name, [])
        if group_ddis:
            chi2_values = np.array([x["chi2"] for x in group_ddis])
            if len(chi2_values) == 0:
                continue
            if chi_square_cutoff <= 0.0:
                best_ddis = group_ddis
            elif chi_square_cutoff >= 1.0:
                max_chi2 = np.max(chi2_values)
                best_ddis = [x for x in group_ddis if x["chi2"] == max_chi2]
            else:
                quantile_cut = np.quantile(chi2_values, chi_square_cutoff)
                best_ddis = [x for x in group_ddis if x["chi2"] >= quantile_cut]
            best_ddis_per_group[group_name] = best_ddis
    return best_ddis_per_group


def build_ddi_adjacency_matrix(
    best_ddis_per_group: dict,
) -> tuple[np.ndarray, list[str]]:
    """Build binary adjacency matrix for DDI network."""
    all_domains = set()
    for group_ddis in best_ddis_per_group.values():
        for entry in group_ddis:
            d1, d2 = entry["ddi"]
            all_domains.add(d1)
            all_domains.add(d2)
    all_domains = sorted(all_domains)
    domain_to_idx = {d: i for i, d in enumerate(all_domains)}
    A = np.zeros((len(all_domains), len(all_domains)), dtype=int)
    for group_ddis in best_ddis_per_group.values():
        for entry in group_ddis:
            d1, d2 = entry["ddi"]
            i, j = domain_to_idx[d1], domain_to_idx[d2]
            A[i, j] = 1
    return A, all_domains


def compute_fp_rate(predicted_ddis, known_ddis):
    false_positives = len(predicted_ddis - known_ddis)
    return false_positives / len(predicted_ddis) if predicted_ddis else 0
