#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import networkx as nx
import json
from collections import defaultdict, Counter, deque
import numpy as np
from pathlib import Path
import gc
import psutil
import logging
import sys


import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

from determinism import derive_seed

# Force spawn-based workers. The default 'fork' start method has each worker
# inherit the parent's entire memory image via copy-on-write; once workers
# start mutating Python objects the shared pages get duplicated and the per-
# worker RSS balloons toward the parent's footprint. With ~16 GB resident
# after preprocessing and several workers, that previously tripped a cgroup
# OOM kill mid-pool and surfaced as `BrokenProcessPool` in network_expansion.
# Spawn workers start clean and only receive the pickled task args, keeping
# each worker's RSS bounded to the size of the subgraph it processes.
_MP_CTX = mp.get_context("spawn")


def _reset_resource_tracker():
    """Force-recreate the multiprocessing resource_tracker daemon.

    Under spawn, multiprocessing maintains a singleton resource_tracker
    subprocess to clean up shared semaphores. On long-running parents
    (kgiddi runs 4-7h under SLURM), that daemon can be reaped by the
    container/cgroup or have its pipe closed silently. The next
    ``ProcessPoolExecutor`` then crashes inside ``Queue.__init__`` with
    ``BrokenPipeError`` from ``resource_tracker._send`` because the
    cached fd points at a dead reader. Clearing ``_fd``/``_pid`` lets
    the next ``ensure_running()`` call respawn a fresh daemon.
    """
    try:
        from multiprocessing import resource_tracker as _rt
        rt = _rt._resource_tracker
        with rt._lock:
            if rt._fd is not None:
                try:
                    os.close(rt._fd)
                except OSError:
                    pass
            rt._fd = None
            rt._pid = None
    except Exception:
        pass


def _new_pool(threads):
    """Construct ``ProcessPoolExecutor`` with resource_tracker recovery.

    Wraps construction so that a stale resource_tracker daemon (see
    :func:`_reset_resource_tracker`) is detected by catching
    ``BrokenPipeError``/``OSError`` from ``Queue``/``Lock`` init, then
    we clear the tracker state and retry once. A second failure is
    surfaced as before.
    """
    try:
        return ProcessPoolExecutor(max_workers=threads, mp_context=_MP_CTX)
    except (BrokenPipeError, OSError) as err:
        logging.warning(
            "ProcessPoolExecutor init failed (%s); resetting "
            "resource_tracker daemon and retrying once.",
            err,
        )
        _reset_resource_tracker()
        return ProcessPoolExecutor(max_workers=threads, mp_context=_MP_CTX)

from kgiddi_functions import (
    approx_bimax,
    select_best_ddis_per_group,
    extract_go_guided_ddi_subgraphs,
    fold_enrichment,
    compute_fp_rate,
    compute_group_ddi_chi2,
    load_go_graph,
)
from load_data_gm import (
    DEFAULT_PPI_SCORE_CUTOFF,
    canonical_pair,
    load_ddi,
    load_pd_mapping,
    load_ppi,
    load_pgo,
    check_file_existence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        # Union by rank
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


def log_resource_usage(note=""):
    # Wrapped: in container teardown (SLURM SIGTERM, cgroup cleanup) /proc
    # entries may disappear before Python exits, making psutil raise
    # NoSuchProcess on our own pid. Swallow so the real cause (e.g. SLURM
    # timeout exit 140) surfaces instead of a misleading psutil traceback.
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu = process.cpu_percent(interval=0.1)
        logging.info(f"{note} | Memory usage: {mem_mb:.2f} MB | CPU: {cpu:.1f}%")
    except (psutil.Error, OSError) as e:
        logging.warning(f"{note} | resource sampling failed: {e}")


def summary_stats(data):
    return {
        "min": int(np.min(data)),
        "max": int(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
    }


def get_go_info(go_graph):
    # Get levels and depths of go terms
    go_levels = {}
    go_depths = {}
    for u, v, data in go_graph.edges(data=True):
        go_levels[v] = data["level"]
        go_depths[v] = data["depth"]
    level_stats = summary_stats(list(go_levels.values()))
    depth_stats = summary_stats(list(go_depths.values()))

    logging.info(level_stats)
    logging.info(depth_stats)

    # The undirected view, not an all-pairs distance dict.
    #
    # `dict(nx.all_pairs_shortest_path_length(go_undirected))` used to be built
    # here: ~1.4e8 entries over the 10131-node molecular_function graph, tens of
    # GB, for every kgiddi task. Its only consumer thresholded the distances, and
    # only ever between the few thousand GO terms actually annotated on proteins
    # in the split -- which `preprocessing` discovers, and this function cannot.
    # So the graph is handed on and the bounded BFS happens there, per used term.
    go_undirected = go_graph.to_undirected()

    return go_levels, go_undirected


def build_close_protein_index(protein_go_terms, go_undirected, threshold):
    """`{protein: set of proteins within GO distance <= threshold}`.

    Replaces `protein_distances` + `functionally_similar`, which computed and
    stored every finite pairwise GO distance in order to compare each one
    against `threshold` exactly once. See the note in `preprocessing`.

    `dist(p, q)` is `min over t1 in T(p), t2 in T(q)` of the GO-graph distance,
    so `dist(p, q) <= threshold` iff some `t2 in T(q)` lies in the
    radius-`threshold` ball around some `t1 in T(p)`. The balls are found by a
    depth-limited BFS from each *used* term over the full undirected GO graph --
    the full graph, because a shortest path between two annotated terms may pass
    through unannotated ones, exactly as the all-pairs matrix allowed.

    A protein is **not** its own neighbour, matching the old behaviour:
    `protein_distances` only ever held `i < j` pairs, so `(p, p)` was absent and
    `close[p]` never contained `p`. That is load-bearing in
    `build_ddi_network` -- it is what stops two PPIs that share a protein from
    being related through that shared protein -- so it is preserved rather than
    quietly "fixed".
    """
    used_terms = sorted({t for terms in protein_go_terms.values() for t in terms})
    used = set(used_terms)

    ball = {}
    for term in used_terms:
        seen = {term}
        frontier = deque([(term, 0)])
        reachable = []
        while frontier:
            node, dist = frontier.popleft()
            if node in used:
                reachable.append(node)
            if dist == threshold:
                continue
            for neighbour in go_undirected[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append((neighbour, dist + 1))
        ball[term] = reachable

    proteins_by_term = defaultdict(list)
    for protein, terms in protein_go_terms.items():
        for term in terms:
            proteins_by_term[term].append(protein)

    close = defaultdict(set)
    for protein, terms in protein_go_terms.items():
        reach = set()
        for term in terms:
            reach.update(ball.get(term, ()))
        neighbours = set()
        for term in reach:
            neighbours.update(proteins_by_term.get(term, ()))
        neighbours.discard(protein)
        if neighbours:
            close[protein] = neighbours
    return close


_MAX_SUBGRAPH_NODES = 800
_MAX_SUBGRAPH_DENSE_NODES = 100
_MAX_SUBGRAPH_DENSITY = 50  # edges per node


def process_go_term(go_term, subgraph, bicluster_cutoff):
    # Log the GO term and subgraph size
    # logging.info(f"Processing GO term: {go_term} | Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()}")
    local_predicted = set()
    n = subgraph.number_of_nodes()
    if n == 1:
        local_predicted.update(subgraph.edges)
    else:
        # Pathological cases: very large or dense subgraphs make approx_bimax
        # blow up combinatorially in bicluster search / conquer() recursion.
        # Skip them — benchmark-only output stability means dropping a handful
        # of mega-subgraphs is acceptable, and a multi-hour single subgraph is
        # what was causing the graph_model tasks to wedge.
        e = subgraph.number_of_edges()
        if n > _MAX_SUBGRAPH_NODES or (
            n > _MAX_SUBGRAPH_DENSE_NODES and e / max(n, 1) > _MAX_SUBGRAPH_DENSITY
        ):
            logging.warning(
                f"skip pathological subgraph go={go_term} nodes={n} edges={e}"
            )
            return go_term, local_predicted
        nodes = list(subgraph.nodes)
        A = nx.to_numpy_array(subgraph, nodelist=nodes)
        biclusters = approx_bimax(A, b=bicluster_cutoff)
        for bicluster in biclusters:
            row_indices, col_indices = bicluster
            if row_indices and col_indices:
                idx_pairs = np.array(np.meshgrid(row_indices, col_indices)).T.reshape(
                    -1, 2
                )
                local_predicted.update(
                    (nodes[int(i)], nodes[int(j)]) for i, j in idx_pairs
                )
                local_predicted.update(
                    (nodes[int(i)], nodes[int(j)]) for i, j in idx_pairs
                )
    return go_term, local_predicted


def evaluate_params(
    args,
    threads: int = 1,
):  # -> tuple[float | Literal[0], Any, Any, float | Literal[0]]:
    (
        chi_square_cutoff,
        bicluster_cutoff,
        connected_components,
        group_ddi_chi2,
        shared_go_domains,
        known_ddis,
        all_domains,
    ) = args
    # Only select best DDIs per group and build DDI network for this cutoff
    logging.info(
        f"Evaluating parameters: chi_square_cutoff={chi_square_cutoff}, bicluster_cutoff={bicluster_cutoff}"
    )
    best_ddis_per_group = select_best_ddis_per_group(
        connected_components, group_ddi_chi2, chi_square_cutoff
    )
    ddi_network_edges = {
        entry["ddi"]
        for group_ddis in best_ddis_per_group.values()
        for entry in group_ddis
    }
    predicted_ddis, fold, fp_rate = network_expansion(
        shared_go_domains, ddi_network_edges, known_ddis, all_domains, bicluster_cutoff, threads
    )
    # Save predicted DDIs for this training step
    with open(
        f"predicted_ddis_train_chi{chi_square_cutoff}_bic{bicluster_cutoff}.txt", "w"
    ) as f:
        for d1, d2 in predicted_ddis:
            f.write(f"{d1}\t{d2}\n")
    del ddi_network_edges, predicted_ddis
    gc.collect()
    return fold, chi_square_cutoff, bicluster_cutoff, fp_rate


def network_expansion(
    shared_go_domains: dict[str, set[str]],
    ddi_network_edges: set[tuple[str, str]],
    known_ddis: set[tuple[str, str]],
    all_domains: set[str],
    bicluster_cutoff: float,
    threads: int = 1,
):

    go_ddi_subgraphs = extract_go_guided_ddi_subgraphs(
        shared_go_domains, ddi_network_edges
    )
    logging.info(f"Number of GO-guided DDI subgraphs: {len(go_ddi_subgraphs)}")
    log_resource_usage("After extracting GO-guided DDI subgraphs")

    predicted_ddis = set()
    debug_lines = []
    # Single pool with bounded in-flight submission. Previously a fresh
    # ProcessPoolExecutor was instantiated per 100-item chunk; with the spawn
    # start method, ~18 workers paid ~1-2s spawn cost on every chunk boundary,
    # which on the test database (24+ chunks) burned 10-15 min before doing
    # any useful work. One pool + an in-flight cap of 2x threads keeps the
    # memory profile the chunking was originally there for.
    go_items = list(go_ddi_subgraphs.items())
    total = len(go_items)
    in_flight_cap = max(2 * threads, 4)
    log_every = max(in_flight_cap, 100)
    completed = 0
    with _new_pool(threads) as executor:
        in_flight = {}

        def _drain_one():
            nonlocal completed
            done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                go_term, local_predicted = fut.result()
                predicted_ddis.update(local_predicted)
                debug_lines.append(f"{go_term}\t{len(local_predicted)}\n")
                in_flight.pop(fut)
                completed += 1
                if completed % log_every == 0:
                    logging.info(f"Processed {completed}/{total} GO terms")

        for go_term, subgraph in go_items:
            while len(in_flight) >= in_flight_cap:
                _drain_one()
            fut = executor.submit(
                process_go_term, go_term, subgraph, bicluster_cutoff
            )
            in_flight[fut] = go_term
        while in_flight:
            _drain_one()
    log_resource_usage(f"After processing all {total} GO terms")

    with open("debug_go_predicted_ddis.txt", "a") as f:
        f.writelines(
            "After completing all chunks with bicluster cutoff "
            + str(bicluster_cutoff)
            + "\n"
        )
        # Sorted: the lines were appended in worker-completion order, so this
        # debug dump differed between runs even though the predictions did not.
        f.writelines(sorted(debug_lines))

    # Compute fold enrichment
    fold = fold_enrichment(predicted_ddis, known_ddis, all_domains)
    fp_rate = compute_fp_rate(predicted_ddis, known_ddis)
    logging.info(f"Number of predicted DDIs: {len(predicted_ddis)}")
    logging.info(f"Fold Enrichment: {fold}, FP-Rate: {fp_rate}\n")
    return predicted_ddis, fold, fp_rate


def preprocessing(
    db_path,
    go_graph,
    go_levels,
    go_undirected,
    out_dir,
    threshold,
    permutation=False,
    seed=42,
    ppi_score_cutoff=DEFAULT_PPI_SCORE_CUTOFF,
):

    logging.info(f"Starting preprocessing, loading data from {db_path}")

    ddi_df = load_ddi(db_path)
    pd_df = load_pd_mapping(db_path)
    ppi_df = load_ppi(db_path)
    pgo_df = load_pgo(db_path)
    pgo_df_raw = pgo_df

    logging.info(
        f"Filtering PPI data by STRING confidence (score >= {ppi_score_cutoff})..."
    )
    n_ppi_raw = len(ppi_df)
    # Inclusive because STRING's confidence bands are closed at their lower
    # edge (>= 900 highest, >= 700 high, >= 400 medium), so a strict > drops
    # every row sitting exactly on the boundary.
    #
    # The cutoff is a real knob, not a formality: a split database carries only
    # the interactome of its own proteins, so the highest-confidence band can be
    # a few percent of it. minimal_leakage/test_balanced holds 254 PPI rows, of
    # which 8 clear 900 and 2 survive the GO filter below -- too few for
    # build_ddi_network to find more than one union-find cluster, which makes
    # chi2 undefined. Hence ppi_score_cutoff = 400 (STRING "medium confidence",
    # its own default) in assets/kgiddi*.json. Re-probe cluster counts before
    # raising it again.
    ppi_df = ppi_df[ppi_df["score"] >= ppi_score_cutoff].reset_index(drop=True)

    # Reduce ddi_df and pd_df based on proteins in ppi_df
    logging.info("Filtering DDI and PD mapping data based on PPI proteins...")
    proteins_in_ppi = set(ppi_df["protein_1"]).union(set(ppi_df["protein_2"]))
    pgo_filtered_df = pgo_df[
        pgo_df["go_accession"].isin(go_graph.nodes)
        & pgo_df["uniprot_id"].isin(proteins_in_ppi)
    ].reset_index(drop=True)
    valid_proteins_in_pgo = set(pgo_filtered_df["uniprot_id"].unique())
    # Reduce ppi to proteins present in pgo_filtered_df
    ppi_filtered_df = ppi_df[
        (ppi_df["protein_1"].isin(valid_proteins_in_pgo))
        & (ppi_df["protein_2"].isin(valid_proteins_in_pgo))
    ].reset_index(drop=True)
    # Get all proteins present in filtered ppi
    proteins_in_ppi = set(ppi_filtered_df["protein_1"]).union(
        set(ppi_filtered_df["protein_2"])
    )
    # Now filter pd_df to proteins present in filtered ppi
    pd_filtered_df = pd_df[pd_df["uniprot_id"].isin(proteins_in_ppi)].reset_index(
        drop=True
    )
    involved_domains = set(pd_filtered_df["pfam_id"].unique())
    ddi_filtered_df = ddi_df[
        ddi_df["domain_a"].isin(involved_domains)
        & ddi_df["domain_b"].isin(involved_domains)
    ].reset_index(drop=True)

    pd_df = pd_filtered_df
    pgo_df = pgo_filtered_df
    ddi_df = ddi_filtered_df.drop_duplicates().reset_index(drop=True)
    ppi_df = ppi_filtered_df
    logging.info(
        f"After limiting: {len(ppi_df)} PPIs, {len(pd_df)} PD mappings, {len(pgo_df)} PGO mappings, {len(ddi_df)} DDIs"
    )
    # An empty interactome is never a legitimate input: every downstream stage
    # (union-find clustering, chi2, GO-guided expansion) degenerates silently
    # and the crash only surfaces further down as `max() iterable argument is
    # empty`. Name the split and the counts that produced it instead.
    if len(ppi_df) == 0:
        raise ValueError(
            f"{db_path}: no PPIs left after the confidence filter "
            f"(ppi_score_cutoff={ppi_score_cutoff}, {n_ppi_raw} PPI rows in the "
            f"database, {len(proteins_in_ppi)} proteins survived, "
            f"{len(pgo_filtered_df)} of {len(pgo_df_raw)} PGO rows matched). "
            "Either this split database ships no protein_protein_interaction "
            "rows, or its score column is on a different scale than the cutoff "
            "assumes -- check `SELECT count(*), min(score), max(score) FROM "
            "protein_protein_interaction`."
        )

    pd_mapping = pd_df.groupby("uniprot_id")["pfam_id"].apply(set).to_dict()

    # Build PPI partners mapping (vectorized)
    ppi_partners = defaultdict(set)
    p1s = ppi_df["protein_1"].values
    p2s = ppi_df["protein_2"].values
    for p1, p2 in zip(p1s, p2s):
        ppi_partners[p1].add(p2)
        ppi_partners[p2].add(p1)

    # (A loop over the whole all-pairs distance dict used to sit here, building
    # a `filtered_distances` list that nothing ever read. Removed with the dict.)

    # Now get for each protein the go_term(s) with the highest level (most specific)
    protein_go_terms = {}
    for protein, group in pgo_df.groupby("uniprot_id"):
        go_terms = group["go_accession"].tolist()
        max_level = -1
        specific_terms = []
        for go_term in go_terms:
            level = go_levels.get(go_term, float("inf"))
            if level > max_level:
                max_level = level
                specific_terms = [go_term]
            elif level == max_level:
                specific_terms.append(go_term)
        protein_go_terms[protein] = specific_terms

    if permutation:
        logging.info(
            "Permuting GO term assignments to proteins for control experiment..."
        )
        all_proteins = list(protein_go_terms.keys())
        all_go_terms = list(protein_go_terms.values())
        # Seeded per split (`train`, `test_balanced`, ...) rather than from the
        # global numpy RNG: the permutation control has to be reproducible, and
        # keying on the split stem keeps train and test permutations distinct
        # without depending on the order the splits happen to be processed in.
        random.Random(derive_seed(seed, "permutation", Path(db_path).stem)).shuffle(
            all_go_terms
        )
        permuted_protein_go_terms = {
            protein: go_terms for protein, go_terms in zip(all_proteins, all_go_terms)
        }
        protein_go_terms = permuted_protein_go_terms

    # Thresholded GO-similarity index, built from GO-term balls.
    #
    # What used to be here: an (n_go x n_go) all-pairs distance matrix, then a
    # `for i: for j > i` loop over every protein pair taking a numpy min per
    # pair, storing every finite distance in `protein_distances`.
    #
    # Two reasons that had to go, both measured on external_test/test
    # (47408 proteins with GO terms, 3564 used GO terms):
    #
    #   * the double loop is 1.14e9 Python iterations -- 2 h 38 m in the
    #     2026-09-03 run, between "Permuting GO term assignments" and the next
    #     log line;
    #   * the molecular_function GO graph is essentially one connected
    #     component, so almost every pair has a *finite* distance and got
    #     stored: ~1.1e9 dict entries, which is the 150 GB resident that run
    #     reported against a 160 GB cap.
    #
    # And none of those distances was ever read as a distance. The only
    # consumers are `build_ddi_network`, which immediately thresholds them into
    # "is q within `threshold` of p", and `functionally_similar`, which did the
    # same comparison. So the thresholded relation is built directly.
    #
    # It factors through GO terms exactly: `dist(p, q) <= threshold` iff some
    # term of q lies within `threshold` GO-graph hops of some term of p, i.e.
    # iff `T(q)` meets the union of the radius-`threshold` balls around `T(p)`.
    # With only ~3564 terms in play that is a few thousand bounded BFS runs
    # instead of a billion-entry matrix.
    close = build_close_protein_index(protein_go_terms, go_undirected, threshold)
    n_close_pairs = sum(len(v) for v in close.values())
    logging.info(
        f"GO-similarity index at threshold {threshold}: "
        f"{len(close)} proteins with at least one neighbour, "
        f"{n_close_pairs} ordered close pairs, "
        f"max neighbourhood {max((len(v) for v in close.values()), default=0)}"
    )

    # Creation of ppi_list
    ppi1s = ppi_df["protein_1"].values
    ppi2s = ppi_df["protein_2"].values
    ppi_list = [tuple(sorted((p1, p2))) for p1, p2 in zip(ppi1s, ppi2s)]

    # Build GO to protein list mapping
    shared_go_domains = {}
    # Create go_protein_list from the created protein_go_terms
    go_protein_list = defaultdict(set)
    for protein, go_terms in protein_go_terms.items():
        for go_term in go_terms:
            go_protein_list[go_term].add(protein)

    logging.info(f"Number of GO terms in go_proteinlist: {len(go_protein_list)}")
    logging.info(f"pd_mapping has {len(pd_mapping)} entries")
    logging.info(f"ppi_partners has {len(ppi_partners)} entries")

    for go_term, proteins_with_go in go_protein_list.items():
        interacting_proteins = set()
        for protein in proteins_with_go:
            interacting_proteins.update(ppi_partners.get(protein, set()))
        d1 = set()
        d2 = set()
        for protein in proteins_with_go:
            d1.update(pd_mapping.get(protein, set()))
        for protein in interacting_proteins:
            d2.update(pd_mapping.get(protein, set()))
        shared_go_domains[go_term] = [list(d1), list(d2)]

    # Delete ppi_partners to free memory
    del ppi_partners, go_protein_list
    gc.collect()

    with open(os.path.join(out_dir, "kgiddi_go_domains.json"), "w") as f:
        json.dump(shared_go_domains, f, indent=2)

    return ddi_df, close, ppi_list, pd_df, shared_go_domains


def build_ddi_network(close, ppi_list, pd_df, threshold, context=""):
    """Cluster functionally similar PPIs with union-find, then chi2 the DDIs.

    `close` is the thresholded GO-similarity index from
    `build_close_protein_index`: `close[p]` is every protein within `threshold`
    GO-graph hops of `p`, excluding `p` itself. It replaces the
    `protein_distances` dict this used to threshold on entry -- see the note in
    `preprocessing` for why that dict is gone.

    The `KGIDDI_LEGACY_BUILD` O(N^2) all-pairs escape hatch is gone with it: it
    called `functionally_similar(protein_distances, ...)`, and there are no
    stored distances to call it on. The inverted-index path below is the only
    one, and it is the one the 2026-09-03 run used.
    """
    # Prepare list of all PPIs as sorted tuples, ordered by degree (number of interactions) descending
    # Ordering allows to cluster high-degree PPIs first, improving efficiency
    ppi_degree = Counter()
    for p1, p2 in ppi_list:
        ppi_degree[p1] += 1
        ppi_degree[p2] += 1

    def ppi_node_degree(ppi):
        # Degree is the sum of degrees of both proteins in the PPI
        return ppi_degree[ppi[0]] + ppi_degree[ppi[1]]

    ppi_nodes = [tuple(sorted(ppi)) for ppi in ppi_list]
    ppi_nodes.sort(key=ppi_node_degree, reverse=True)
    uf = UnionFind(ppi_nodes)

    clusters_dict = defaultdict(set)
    # Inverted-index path: enumerate only candidate similar PPIs via a
    # protein-keyed index.
    #
    # Similarity: ppi(a,b) ~ ppi(c,d) iff (c in close[a] AND d in close[b])
    #                                  OR (c in close[b] AND d in close[a])
    ppi_by_protein = defaultdict(list)
    for idx, (a, b) in enumerate(ppi_nodes):
        ppi_by_protein[a].append(idx)
        ppi_by_protein[b].append(idx)

    for i, (a, b) in enumerate(ppi_nodes):
        close_a = close.get(a, ())
        close_b = close.get(b, ())
        candidates = set()
        # Rule 1: c ~ a AND d ~ b
        for c in close_a:
            for j in ppi_by_protein.get(c, ()):
                if j <= i:
                    continue
                x, y = ppi_nodes[j]
                other = y if x == c else x
                if other in close_b:
                    candidates.add(j)
        # Rule 2: c ~ b AND d ~ a
        for c in close_b:
            for j in ppi_by_protein.get(c, ()):
                if j <= i:
                    continue
                x, y = ppi_nodes[j]
                other = y if x == c else x
                if other in close_a:
                    candidates.add(j)
        for j in candidates:
            if not uf.connected(ppi_nodes[i], ppi_nodes[j]):
                uf.union(ppi_nodes[i], ppi_nodes[j])

    for ppi in ppi_nodes:
        clusters_dict[uf.find(ppi)].add(ppi)
    clusters = []
    for idx, (root, members) in enumerate(clusters_dict.items(), 1):
        clusters.append({"group_name": f"Cluster {idx}", "members": list(members)})
    logging.info(
        f"Number of functionally similar PPI clusters (Union-Find) for threshold {threshold}: {len(clusters)}"
    )
    if not clusters:
        raise ValueError(
            f"build_ddi_network: the PPI network is empty (0 clusters) at "
            f"threshold {threshold}, so there is nothing to score. This means "
            "preprocessing returned no usable interactome for this split."
        )
    logging.info(
        f"Biggest cluster has size: {max(len(c['members']) for c in clusters)}"
    )
    # chi2 contrasts each cluster against everything outside it. With a single
    # cluster the union-find partition swallows the whole interactome, so
    # B (outside-with-DDI) and D (outside-without-DDI) are both 0, the
    # contingency denominator (A+C)*(B+D)*(A+B)*(C+D) is 0, and
    # compute_group_ddi_chi2 returns 0 for *every* DDI. Nothing is rankable and
    # the failure only surfaces later as a ZeroDivisionError while normalising
    # the scores. Name the degenerate partition here instead.
    if len(clusters) == 1:
        raise ValueError(
            f"build_ddi_network{f' ({context})' if context else ''}: union-find "
            f"collapsed all {len(ppi_list)} PPIs into a single cluster at "
            f"GO-similarity threshold {threshold}. chi2 has no outside stratum "
            "to contrast against, so every DDI would score 0 and KGIDDI can "
            "rank nothing. The interactome is too small or too GO-uniform for "
            "this method -- check the PPI count above and the number of "
            "distinct protein_go_terms in this split."
        )
    # PPI interactions as set of tuples
    ppi_interactions = set(ppi_list)
    # Protein to domain mapping as dict of sets
    # zip over the column arrays, not `iterrows()`: that builds a Series per
    # row, and pd_df has ~63 k rows for external_test.
    pd_filtered_dict = defaultdict(set)
    for uniprot_id, pfam_id in zip(
        pd_df["uniprot_id"].values, pd_df["pfam_id"].values
    ):
        pd_filtered_dict[uniprot_id].add(pfam_id)

    connected_components = {
        cluster["group_name"]: cluster["members"] for cluster in clusters
    }

    group_ddi_chi2 = compute_group_ddi_chi2(
        connected_components, ppi_interactions, pd_filtered_dict
    )

    return connected_components, group_ddi_chi2


def run_kgiddi(
    database_path, params_file, out_dir, test_splits,
    threads=1, seed=42, ppi_score_cutoff=None,
):
    """Train once, score every test split.

    `test_splits` maps variant -> output predictions path, e.g.
    {"balanced": ".../predictions_balanced.parquet"}. A database shipping both
    `test_balanced` and `test_realistic` shares one training phase -- the
    expensive part -- and only the scoring phase runs per variant.

    `ppi_score_cutoff` is the pipeline-level `params.ppi_score_cutoff`
    (`--ppi_score_cutoff` on the command line). When it is None the model JSON's
    own `parameter_list.ppi_score_cutoff` is used, and failing that the STRING
    "medium confidence" default of 400 -- so a hand-written JSON still works
    standalone, but a pipeline run always drives every graph model from one
    value.
    """

    db_train = Path(os.path.join(database_path, "train.sqlite3"))
    check_file_existence(db_train)

    test_dbs = {
        variant: Path(os.path.join(database_path, f"{split}.sqlite3"))
        for variant, (split, _) in test_splits.items()
    }
    for db_test in test_dbs.values():
        check_file_existence(db_test)

    # Load json parameters
    with open(params_file) as f:
        params_json = json.load(f)
        data_to_load = params_json.get("data", ["DDI", "PD", "PPI", "PGO"])
        go_graph_path = params_json.get("parameter_list", {}).get("go_graph", "")

    chi_square_cutoffs = params_json["parameter_list"]["chi_square_cutoff"]
    bicluster_cutoffs = params_json["parameter_list"]["bicluster_cutoff"]
    threshold = params_json["parameter_list"]["threshold"]
    if ppi_score_cutoff is None:
        ppi_score_cutoff = params_json["parameter_list"].get(
            "ppi_score_cutoff", DEFAULT_PPI_SCORE_CUTOFF
        )
    permutation = params_json.get("permutation", False)

    logging.info(f"Data to load: {data_to_load}")

    # Check if optimized parameters exist, if so training is skipped
    optimized_params = params_json.get("optimized", None)
    # optimized_params = {"chi_square_cutoff" : 0.7, "bicluster_cutoff": 0.5}

    training = False
    if optimized_params is None:
        training = True
        logging.info("No optimized parameters found, starting training phase.")

    logging.info(f"Training mode: {training}")

    go_graph_nx = load_go_graph(go_graph_path)
    logging.info(
        f"GO graph loaded with {go_graph_nx.number_of_nodes()} nodes and {go_graph_nx.number_of_edges()} edges."
    )

    # Get necessary information from GO graph
    go_levels, go_undirected = get_go_info(go_graph_nx)

    log_resource_usage("Start run_kgiddi")

    if training:
        best_fold = -1
        best_params = {}

        # Get clusters
        ddi_df, close, ppi_list, pd_df, shared_go_domains = preprocessing(
            db_train,
            go_graph_nx,
            go_levels,
            go_undirected,
            out_dir,
            threshold,
            permutation=permutation,
            seed=seed,
            ppi_score_cutoff=ppi_score_cutoff,
        )

        # Precompute parameter-independent structures
        connected_components, group_ddi_chi2 = build_ddi_network(
            close, ppi_list, pd_df, threshold, context=f"{database_path} train"
        )

        known_ddis = set(
            tuple(x)
            for x in ddi_df.loc[
                ddi_df["interaction"] == 1, ["domain_a", "domain_b"]
            ].values
        )
        # Get all domains from pd_df
        all_domains = set(pd_df["pfam_id"])

        param_grid = [
            (
                chi_square_cutoff,
                bicluster_cutoff,
                connected_components,
                group_ddi_chi2,
                shared_go_domains,
                known_ddis,
                all_domains,
            )
            for chi_square_cutoff in chi_square_cutoffs
            for bicluster_cutoff in bicluster_cutoffs
        ]

        results = []
        for args in tqdm(
            param_grid, total=len(param_grid), desc="Parameter grid search"
        ):
            results.append(evaluate_params(args, threads))

        for fold, chi_square_cutoff, bicluster_cutoff, fp_rate in results:
            if fold > best_fold:
                best_fold = fold
                best_params = {
                    "chi_square_cutoff": chi_square_cutoff,
                    "bicluster_cutoff": bicluster_cutoff,
                    "fp_rate": fp_rate,
                }
            logging.info(
                f"Evaluated: chi_square_cutoff={chi_square_cutoff}, bicluster_cutoff={bicluster_cutoff}, fold={fold}, fp_rate={fp_rate}"
            )
        logging.info(
            f"Current Best Fold Enrichment: {best_fold} with parameters: {best_params}"
        )
        # Delete large training objects
        del ddi_df, close, ppi_list, pd_df
        gc.collect()

    if not training:
        best_params = optimized_params

    for variant, (split, out_predictions) in test_splits.items():
        logging.info(f"----- Scoring test split {split} (variant {variant}) -----")
        score_test_split(
            test_dbs[variant],
            out_predictions,
            variant,
            best_params,
            params_json,
            go_graph_nx,
            go_levels,
            go_undirected,
            out_dir,
            threshold,
            permutation,
            threads,
            seed,
            ppi_score_cutoff,
        )


def score_test_split(
    db_test,
    out_predictions,
    variant,
    best_params,
    params_json,
    go_graph_nx,
    go_levels,
    go_undirected,
    out_dir,
    threshold,
    permutation,
    threads,
    seed=42,
    ppi_score_cutoff=DEFAULT_PPI_SCORE_CUTOFF,
):
    """Score one test split with parameters already chosen on the train split."""
    # Run preprocessing for test data
    (
        ddi_df_test,
        close_test,
        ppi_list_test,
        pd_df_test,
        shared_go_domains_test,
    ) = preprocessing(
        db_test,
        go_graph_nx,
        go_levels,
        go_undirected,
        out_dir,
        threshold,
        permutation=permutation,
        seed=seed,
        ppi_score_cutoff=ppi_score_cutoff,
    )
    gc.collect()
    log_resource_usage("After preprocessing test data")
    known_ddis_test = set(
        tuple(x)
        for x in ddi_df_test.loc[
            ddi_df_test["interaction"] == 1, ["domain_a", "domain_b"]
        ].values
    )

    all_domains_test = set(pd_df_test["pfam_id"])
    logging.info(f"Number of known DDIs in test data: {len(known_ddis_test)}")
    logging.info(f"Number of all domains in test data: {len(all_domains_test)}")
    # Filter PPI network by optimized parameters

    logging.info("----- Test Data Evaluation -----")
    logging.info(f"Using optimized parameters: {best_params}")
    # ------- I: Build DDI network ----- #
    logging.info("----- I: Build DDI network -----")
    # Precompute parameter-independent structures for test data
    connected_components_test, group_ddi_chi2_test = build_ddi_network(
        close_test, ppi_list_test, pd_df_test, threshold,
        context=f"{db_test} test_{variant}",
    )

    # For the selected chi_square_cutoff, get DDI edges
    best_ddis_per_group_test = select_best_ddis_per_group(
        connected_components_test, group_ddi_chi2_test, best_params["chi_square_cutoff"]
    )

    # ------- II: DDI network expansion ----- #
    logging.info("----- II: DDI network expansion -----")
    ddi_network_edges_test = {
        entry["ddi"]
        for group_ddis in best_ddis_per_group_test.values()
        for entry in group_ddis
    }
    predicted_ddis_test, fold_test, fp_rate_test = network_expansion(
        shared_go_domains_test,
        ddi_network_edges_test,
        known_ddis_test,
        all_domains_test,
        best_params["bicluster_cutoff"],
        threads,
    )

    logging.info(
        f"Test Data - Number of statistically significant DDIs: {len(ddi_network_edges_test)}"
    )
    # For debugging write DDI edges to file
    ddi_network_df = pd.DataFrame(
        list(ddi_network_edges_test), columns=["domain_a", "domain_b"]
    )
    ddi_network_df.to_csv(
        os.path.join(out_dir, f"kgiddi_ddi_network_{variant}.csv"), index=False
    )

    with open(os.path.join(out_dir, f"predicted_ddis_{variant}.txt"), "w") as f:
        for d1, d2 in predicted_ddis_test:
            f.write(f"{d1}\t{d2}\n")
    # Delete large test objects
    del (
        pd_df_test,
        shared_go_domains_test,
        known_ddis_test,
        all_domains_test,
        ddi_network_edges_test,
    )
    gc.collect()

    ###
    # Save results
    ###

    # Save optimized parameters
    optimized_params_output = {
        "chi_square_cutoff": best_params["chi_square_cutoff"],
        "bicluster_cutoff": best_params["bicluster_cutoff"],
        "fold_enrichment": fold_test,
        "fp_rate": fp_rate_test,
    }
    # Add to original json
    with open(os.path.join(out_dir, f"kgiddi_{variant}.json"), "w") as f:
        json.dump(dict(params_json, optimized=optimized_params_output), f, indent=2)

    chi2_scores = {
        (entry["ddi"][0], entry["ddi"][1]): entry["chi2"]
        for group_ddis in best_ddis_per_group_test.values()
        for entry in group_ddis
    }
    # Prepare output: Domain id1, domain id2, true interaction (0/1), predicted interaction (0/1), predicted probability (chi2 score normalized)
    # Every DDI row in a split database belongs to that split by construction
    # (domainsplit's SUBSET_SPLIT_DB), so there is nothing to filter out.
    # zip over the column arrays rather than `iterrows()` -- 138642 rows for
    # external_test, each of which was materialised as a Series. Insertion order
    # is unchanged, which matters: `random_jitter` below is indexed positionally
    # against `ddi_actual.items()`. Duplicate keys still keep the last
    # occurrence, exactly as the dict comprehension did.
    ddi_actual = dict(
        zip(
            zip(
                ddi_df_test["domain_a"].values,
                ddi_df_test["domain_b"].values,
            ),
            ddi_df_test["interaction"].values,
        )
    )
    output_rows = []
    # Normalize chi2 scores for later roc curve plotting, using max chi2 score in test data
    # A single DDI can legitimately score 0 (its contingency row or column is
    # empty), but an all-zero set means the chi2 stratification degenerated and
    # there is no scale to normalise against -- dividing by it raised
    # ZeroDivisionError here. The single-cluster case is caught upstream in
    # build_ddi_network; this guard names anything else that gets here.
    max_chi2 = max(chi2_scores.values(), default=0.0)
    if chi2_scores and max_chi2 <= 0:
        raise ValueError(
            f"{db_test} test_{variant}: all {len(chi2_scores)} selected DDIs "
            "have chi2 == 0, so the scores cannot be normalised and every "
            "predicted_probability would be identical. The chi2 contingency "
            "tables degenerated -- check the cluster count logged by "
            "build_ddi_network for this split."
        )
    if not chi2_scores:
        max_chi2 = 1.0
    chi2_scores = {k: v / max_chi2 for k, v in chi2_scores.items()}

    multiplier = sys.float_info.epsilon * 1000
    # Generate for each key, randomly positive/negative or no jitter
    # Seeded per variant: the jitter only breaks score ties for ROC plotting,
    # but drawn from the global RNG it made every predicted_probability differ
    # in its last bits between runs.
    random_jitter = np.random.default_rng(
        derive_seed(seed, "jitter", variant)
    ).uniform(-multiplier, multiplier, size=len(ddi_actual))
    # If a value is < 0 or > 1 after adding jitter, set it to 0 or 1 respectively, to avoid issues with log scale in roc curve plotting
    for i, ((d1, d2), actual) in enumerate(ddi_actual.items()):
        predicted = (d1, d2) in predicted_ddis_test or (d2, d1) in predicted_ddis_test
        score = chi2_scores.get((d1, d2), chi2_scores.get((d2, d1), 0))
        score += random_jitter[i]
        score = max(0, min(1, score))
        # Both lookups above are order-independent, so the emitted orientation is
        # free -- pin it so every predictions file in the run agrees.
        out_a, out_b = canonical_pair(d1, d2)
        output_rows.append(
            {
                "domain_a": out_a,
                "domain_b": out_b,
                "true_interaction": int(actual),
                "predicted_interaction": int(predicted),
                "predicted_probability": float(score),
            }
        )

    # for (d1, d2), actual in ddi_actual.items():
    #     predicted = (d1, d2) in predicted_ddis_test or (d2, d1) in predicted_ddis_test
    #     score = chi2_scores.get((d1, d2), chi2_scores.get((d2, d1), 0))
    #     output_rows.append({
    #         "domain_a": d1,
    #         "domain_b": d2,
    #         "true_interaction": int(actual),
    #         "predicted_interaction": int(predicted),
    #         "predicted_probability": float(score)
    #     })

    output_df = pd.DataFrame(output_rows)
    if "true_interaction" in output_df.columns:
        output_df["true_interaction"] = output_df["true_interaction"].astype("int8")
    if "predicted_interaction" in output_df.columns:
        output_df["predicted_interaction"] = output_df["predicted_interaction"].astype("int8")
    if "predicted_probability" in output_df.columns:
        output_df["predicted_probability"] = output_df["predicted_probability"].astype("float32")
    if str(out_predictions).endswith(".csv"):
        output_df.to_csv(out_predictions, index=False)
    else:
        output_df.to_parquet(out_predictions, index=False, compression="zstd")
    del output_df, output_rows, chi2_scores, ddi_actual
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    print("Starting KGIDDI...")
    variant = (
        args.test_split[len("test_"):]
        if args.test_split.startswith("test_")
        else args.test_split
    )
    run_kgiddi(
        args.database,
        args.params,
        args.out_dir,
        {variant: (args.test_split, args.out_predictions)},
    )
