#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import networkx as nx
import json
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import gc
import psutil
import logging
import math
import sys


from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    cpu = process.cpu_percent(interval=0.1)
    logging.info(f"{note} | Memory usage: {mem_mb:.2f} MB | CPU: {cpu:.1f}%")


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

    # For each go term, compute the shortest path to all other go terms, in the induced indirected graph, as the distances will be symmetric, calculate efficiently the all pairs shortest path lengths in the undirected version of the graph
    go_undirected = go_graph.to_undirected()
    all_pairs_shortest_path_length = dict(
        nx.all_pairs_shortest_path_length(go_undirected)
    )

    return go_levels, all_pairs_shortest_path_length


# Functionally similar check (replace with your actual logic)
def functionally_similar(protein_distances, ppi1, ppi2, threshold):
    a, b = ppi1
    c, d = ppi2
    dist_ac = protein_distances.get(tuple(sorted((a, c))), float("inf"))
    dist_ad = protein_distances.get(tuple(sorted((a, d))), float("inf"))
    dist_bc = protein_distances.get(tuple(sorted((b, c))), float("inf"))
    dist_bd = protein_distances.get(tuple(sorted((b, d))), float("inf"))
    # Use your threshold and logic here (example: at least one pair below threshold)
    return (
        dist_ac is not None
        and dist_bd is not None
        and dist_ac <= threshold
        and dist_bd <= threshold
    ) or (
        dist_ad is not None
        and dist_bc is not None
        and dist_ad <= threshold
        and dist_bc <= threshold
    )


def process_go_term(go_term, subgraph, bicluster_cutoff):
    # Log the GO term and subgraph size
    # logging.info(f"Processing GO term: {go_term} | Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()}")
    local_predicted = set()
    if len(subgraph.nodes) == 1:
        local_predicted.update(subgraph.edges)
    else:
        nodes = list(subgraph.nodes)
        A = nx.to_numpy_array(subgraph, nodelist=nodes)
        # Pathological cases: very large or dense subgraphs, or subgraphs with highly connected nodes,
        # can cause approx_bimax to run extremely slowly or use excessive memory due to
        # combinatorial explosion in bicluster search or deep recursion in conquer().
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
        shared_go_domains, ddi_network_edges, known_ddis, all_domains, bicluster_cutoff
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
):

    go_ddi_subgraphs = extract_go_guided_ddi_subgraphs(
        shared_go_domains, ddi_network_edges
    )
    logging.info(f"Number of GO-guided DDI subgraphs: {len(go_ddi_subgraphs)}")
    log_resource_usage("After extracting GO-guided DDI subgraphs")

    predicted_ddis = set()
    debug_lines = []
    # Chunked processing for memory efficiency
    go_items = list(go_ddi_subgraphs.items())
    chunk_size = 100
    total = len(go_ddi_subgraphs)
    num_chunks = math.ceil(total / chunk_size)
    processed = 0
    for chunk_idx in range(num_chunks):
        chunk = go_items[processed : processed + chunk_size]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(
                    process_go_term, go_term, subgraph, bicluster_cutoff
                ): go_term
                for go_term, subgraph in chunk
            }
            for future in as_completed(futures):
                go_term, local_predicted = future.result()
                predicted_ddis.update(local_predicted)
                debug_lines.append(f"{go_term}\t{len(local_predicted)}\n")
        processed += len(chunk)
        logging.info(
            f"Processed chunk {chunk_idx + 1}/{num_chunks} ({processed}/{total} GO terms)"
        )
        log_resource_usage(f"After chunk {chunk_idx + 1}")

    with open("debug_go_predicted_ddis.txt", "a") as f:
        f.writelines(
            "After completing all chunks with bicluster cutoff "
            + str(bicluster_cutoff)
            + "\n"
        )
        f.writelines(debug_lines)

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
    all_pairs_shortest_path_length,
    out_dir,
    permutation=False,
):

    logging.info(f"Starting preprocessing, loading data from {db_path}")

    ddi_df = load_ddi(db_path)
    pd_df = load_pd_mapping(db_path)
    ppi_df = load_ppi(db_path)
    pgo_df = load_pgo(db_path)

    logging.info("Filtering PPI data for high-confidence interactions...")
    ppi_df = ppi_df[ppi_df["score"] > 900].reset_index(drop=True)

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

    pd_mapping = pd_df.groupby("uniprot_id")["pfam_id"].apply(set).to_dict()

    # Build PPI partners mapping (vectorized)
    ppi_partners = defaultdict(set)
    p1s = ppi_df["protein_1"].values
    p2s = ppi_df["protein_2"].values
    for p1, p2 in zip(p1s, p2s):
        ppi_partners[p1].add(p2)
        ppi_partners[p2].add(p1)

    # Filter distances to only those go terms present in pgo_df
    valid_go_terms = set(pgo_df["go_accession"].unique())
    filtered_distances = []
    for source, targets in all_pairs_shortest_path_length.items():
        if source in valid_go_terms:
            for target, dist in targets.items():
                if target in valid_go_terms and source != target:
                    filtered_distances.append(dist)

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
        np.random.shuffle(all_go_terms)
        permuted_protein_go_terms = {
            protein: go_terms for protein, go_terms in zip(all_proteins, all_go_terms)
        }
        protein_go_terms = permuted_protein_go_terms

    # Now I have the mapping of proteins to their most specific go terms in protein_go_terms
    proteins = np.array(list(protein_go_terms.keys()))
    n_proteins = len(proteins)
    protein_distances = {}
    # Precompute all pairs indices
    idx_i, idx_j = np.triu_indices(n_proteins, k=1)
    for i, j in zip(idx_i, idx_j):
        p1 = proteins[i]
        p2 = proteins[j]
        terms1 = protein_go_terms[p1]
        terms2 = protein_go_terms[p2]
        t1_arr = np.array(terms1)
        t2_arr = np.array(terms2)
        # Build distance matrix for all pairs (t1, t2)
        dists = np.full((len(t1_arr), len(t2_arr)), np.inf)
        for m, t1 in enumerate(t1_arr):
            for n, t2 in enumerate(t2_arr):
                d = all_pairs_shortest_path_length.get(t1, {}).get(t2, np.inf)
                dists[m, n] = d
        min_dist = np.min(dists) if dists.size > 0 else np.inf
        protein_distances[tuple(sorted((p1, p2)))] = (
            min_dist if min_dist != np.inf else None
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

    return ddi_df, protein_distances, ppi_list, pd_df, shared_go_domains


def build_ddi_network(protein_distances, ppi_list, pd_df, threshold):

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
    # Cluster PPIs with Union-Find
    for i, ppi1 in enumerate(ppi_nodes):
        for j in range(i + 1, len(ppi_nodes)):
            ppi2 = ppi_nodes[j]
            if not uf.connected(ppi1, ppi2):
                if functionally_similar(protein_distances, ppi1, ppi2, threshold):
                    uf.union(ppi1, ppi2)
    for ppi in ppi_nodes:
        clusters_dict[uf.find(ppi)].add(ppi)
    clusters = []
    for idx, (root, members) in enumerate(clusters_dict.items(), 1):
        clusters.append({"group_name": f"Cluster {idx}", "members": list(members)})
    logging.info(
        f"Number of functionally similar PPI clusters (Union-Find) for threshold {threshold}: {len(clusters)}"
    )
    logging.info(
        f"Biggest cluster has size: {max(len(c['members']) for c in clusters)}"
    )
    # PPI interactions as set of tuples
    ppi_interactions = set(ppi_list)
    # Protein to domain mapping as dict of sets
    pd_filtered_dict = defaultdict(set)
    for _, row in pd_df.iterrows():
        pd_filtered_dict[row["uniprot_id"]].add(row["pfam_id"])

    connected_components = {
        cluster["group_name"]: cluster["members"] for cluster in clusters
    }

    group_ddi_chi2 = compute_group_ddi_chi2(
        connected_components, ppi_interactions, pd_filtered_dict
    )

    return connected_components, group_ddi_chi2


def run_kgiddi(database_path, params_file, out_dir, out_predictions):

    db_train = Path(os.path.join(database_path, "train.sqlite3"))
    db_test = Path(os.path.join(database_path, "test.sqlite3"))
    check_file_existence(db_train)
    check_file_existence(db_test)

    # Load json parameters
    with open(params_file) as f:
        params_json = json.load(f)
        data_to_load = params_json.get("data", ["DDI", "PD", "PPI", "PGO"])
        go_graph_path = params_json.get("parameter_list", {}).get("go_graph", "")

    chi_square_cutoffs = params_json["parameter_list"]["chi_square_cutoff"]
    bicluster_cutoffs = params_json["parameter_list"]["bicluster_cutoff"]
    threshold = params_json["parameter_list"]["threshold"]
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
    go_levels, aps_paths = get_go_info(go_graph_nx)

    log_resource_usage("Start run_kgiddi")

    if training:
        best_fold = -1
        best_params = {}

        # Get clusters
        ddi_df, protein_distances, ppi_list, pd_df, shared_go_domains = preprocessing(
            db_train,
            go_graph_nx,
            go_levels,
            aps_paths,
            out_dir,
            permutation=permutation,
        )

        # Precompute parameter-independent structures
        connected_components, group_ddi_chi2 = build_ddi_network(
            protein_distances, ppi_list, pd_df, threshold
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
            results.append(evaluate_params(args))

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
        del ddi_df, protein_distances, ppi_list, pd_df
        gc.collect()

    if not training:
        best_params = optimized_params

    # Run preprocessing for test data
    (
        ddi_df_test,
        protein_distances_test,
        ppi_list_test,
        pd_df_test,
        shared_go_domains_test,
    ) = preprocessing(
        db_test, go_graph_nx, go_levels, aps_paths, out_dir, permutation=permutation
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
        protein_distances_test, ppi_list_test, pd_df_test, threshold
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
    )

    logging.info(
        f"Test Data - Number of statistically significant DDIs: {len(ddi_network_edges_test)}"
    )
    # For debugging write DDI edges to file
    ddi_network_df = pd.DataFrame(
        list(ddi_network_edges_test), columns=["domain_a", "domain_b"]
    )
    ddi_network_df.to_csv(
        os.path.join(out_dir, "kgiddi_ddi_network_test.csv"), index=False
    )

    with open(os.path.join(out_dir, "predicted_ddis_test.txt"), "w") as f:
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
    params_json["optimized"] = optimized_params_output
    with open(os.path.join(out_dir, "kgiddi.json"), "w") as f:
        json.dump(params_json, f, indent=2)

    chi2_scores = {
        (entry["ddi"][0], entry["ddi"][1]): entry["chi2"]
        for group_ddis in best_ddis_per_group_test.values()
        for entry in group_ddis
    }
    # Prepare output: Domain id1, domain id2, true interaction (0/1), predicted interaction (0/1), predicted probability (chi2 score normalized)
    # Filter for eval_relevant DDIs only
    ddi_df_test = ddi_df_test[ddi_df_test["eval_relevant"] == 1].reset_index(drop=True)
    ddi_actual = {
        (row["domain_a"], row["domain_b"]): row["interaction"]
        for _, row in ddi_df_test.iterrows()
    }
    output_rows = []
    # Normalize chi2 scores for later roc curve plotting, using max chi2 score in test data
    max_chi2 = max(chi2_scores.values()) if chi2_scores else 1
    chi2_scores = {k: v / max_chi2 for k, v in chi2_scores.items()}

    multiplier = sys.float_info.epsilon * 1000
    # Generate for each key, randomly positive/negative or no jitter
    random_jitter = np.random.uniform(-multiplier, multiplier, size=len(ddi_actual))
    # If a value is < 0 or > 1 after adding jitter, set it to 0 or 1 respectively, to avoid issues with log scale in roc curve plotting
    for i, ((d1, d2), actual) in enumerate(ddi_actual.items()):
        predicted = (d1, d2) in predicted_ddis_test or (d2, d1) in predicted_ddis_test
        score = chi2_scores.get((d1, d2), chi2_scores.get((d2, d1), 0))
        score += random_jitter[i]
        score = max(0, min(1, score))
        output_rows.append(
            {
                "domain_a": d1,
                "domain_b": d2,
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
    args = parser.parse_args()
    print("Starting KGIDDI...")
    run_kgiddi(args.database, args.params, args.out_dir, args.out_predictions)
