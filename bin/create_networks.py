#! /usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools
import os
import numpy as np
from pathlib import Path

from load_data_gm import load_ddi, load_ppi


def add_random_edges(G, percent_increase=0.1, seed=None):
    """
    Add random edges to a NetworkX graph, increasing the number of edges by percent_increase.
    The relative degree distribution is preserved.
    """
    num_edges = G.number_of_edges()
    num_new_edges = int(percent_increase * num_edges)
    nodes = list(G.nodes())
    existing_edges = set(G.edges())
    all_possible_edges = set(itertools.combinations(nodes, 2))
    possible_edges = list(all_possible_edges - existing_edges)
    if num_new_edges > len(possible_edges):
        num_new_edges = len(possible_edges)
    random.seed(seed)
    new_edges = random.sample(possible_edges, num_new_edges)
    G.add_edges_from(new_edges)
    return G


def remove_random_edges(G, percent_decrease=0.1, seed=None):
    """
    Remove random edges from a NetworkX graph, decreasing the number of edges by percent_decrease.
    """
    num_edges = G.number_of_edges()
    num_remove = int(percent_decrease * num_edges)
    edges = list(G.edges())
    if num_remove > len(edges):
        num_remove = len(edges)
    random.seed(seed)
    edges_to_remove = random.sample(edges, num_remove)
    G.remove_edges_from(edges_to_remove)
    return G


def plot_degree_distribution(G, plot_dir, name_suffix=""):
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(
        degrees,
        bins=range(min(degrees), max(degrees) + 2),
        align="left",
        color="steelblue",
        edgecolor="black",
    )
    plt.xlabel("Node Degree")
    plt.ylabel("Frequency")
    plt.title("Node Degree Distribution" + (f" ({name_suffix})" if name_suffix else ""))
    plt.tight_layout()
    file_path = os.path.join(plot_dir, f"degree_distribution{name_suffix}.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Degree distribution plot saved to {file_path}")


def plot_degree_change(G_before, G_after, plot_dir, name_suffix=""):
    nodes = list(G_before.nodes())
    degree_before = dict(G_before.degree())
    degree_after = dict(G_after.degree())
    # Compute relative change, handle division by zero
    degree_rel_change = []
    for n in nodes:
        before = degree_before[n]
        after = degree_after[n]
        if before == 0:
            # If node had degree 0, use np.nan or skip
            degree_rel_change.append(float("nan"))
        else:
            degree_rel_change.append((after - before) / before)
    # Remove NaNs for plotting
    degree_rel_change = [x for x in degree_rel_change if not np.isnan(x)]
    # Add mean/median and std lines
    mean_dc = float(np.mean(degree_rel_change))
    median_dc = float(np.median(degree_rel_change))
    std_dc = float(np.std(degree_rel_change))
    plt.figure(figsize=(8, 6))
    plt.hist(degree_rel_change, bins=50, color="lightblue", edgecolor="black")
    plt.axvline(
        mean_dc,
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Mean: {mean_dc:.4f}",
    )
    plt.axvline(
        median_dc,
        color="green",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Median: {median_dc:.4f}",
    )
    plt.axvline(
        mean_dc + std_dc,
        color="orange",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Mean + 1 SD: {mean_dc + std_dc:.4f}",
    )
    plt.axvline(
        mean_dc - std_dc,
        color="orange",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Mean - 1 SD: {mean_dc - std_dc:.4f}",
    )
    plt.legend()
    plt.xlabel("Relative Degree Change per Node")
    plt.ylabel("Frequency")
    plt.title(
        "Node Relative Degree Change Distribution"
        + (f" ({name_suffix})" if name_suffix else "")
    )
    plt.tight_layout()
    file_path = os.path.join(
        plot_dir, f"degree_rel_change_distribution{name_suffix}.png"
    )
    plt.savefig(file_path)
    plt.close()
    print(f"Relative degree change distribution plot saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    plot_dir = "tmp/plots"
    os.makedirs(plot_dir, exist_ok=True)
    # Load data from data_test/

    train_db_path = Path("data_test/train.sqlite3")
    test_db_path = Path("data_test/test.sqlite3")
    ddi_dt_train = load_ddi(train_db_path)
    ppi_dt_train = load_ppi(train_db_path)
    ddi_dt_test = load_ddi(test_db_path)
    ppi_dt_test = load_ppi(test_db_path)

    # Convert dataframes to NetworkX graphs
    G_ddi_train = nx.from_pandas_edgelist(ddi_dt_train, "domain_a", "domain_b")
    G_ppi_train = nx.from_pandas_edgelist(ppi_dt_train, "protein_1", "protein_2")
    G_ddi_test = nx.from_pandas_edgelist(ddi_dt_test, "domain_a", "domain_b")
    G_ppi_test = nx.from_pandas_edgelist(ppi_dt_test, "protein_1", "protein_2")

    name_list = ["G_ddi_train", "G_ppi_train", "G_ddi_test", "G_ppi_test"]
    graph_list = [G_ddi_train, G_ppi_train, G_ddi_test, G_ppi_test]
    for i, G in enumerate(graph_list):
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        G_augmented = add_random_edges(G.copy(), percent_increase=0.2, seed=42)
        G_reduced = remove_random_edges(G.copy(), percent_decrease=0.1, seed=42)

        plot_degree_distribution(G, plot_dir=plot_dir, name_suffix=f"_{name_list[i]}")
        plot_degree_distribution(
            G_augmented, plot_dir=plot_dir, name_suffix=f"_{name_list[i]}_augmented"
        )
        plot_degree_distribution(
            G_reduced, plot_dir=plot_dir, name_suffix=f"_{name_list[i]}_reduced"
        )

        plot_degree_change(
            G, G_augmented, plot_dir=plot_dir, name_suffix=f"_{name_list[i]}_augmented"
        )
        plot_degree_change(
            G, G_reduced, plot_dir=plot_dir, name_suffix=f"_{name_list[i]}_reduced"
        )

    # G = nx.erdos_renyi_graph(5000, 0.05)
    # G_augmented = add_random_edges(G.copy(), percent_increase=0.2, seed=42)
    # G_reduced = remove_random_edges(G.copy(), percent_decrease=0.1, seed=42)

    # plot_degree_distribution(G, plot_dir=plot_dir, name_suffix="_example")
    # plot_degree_distribution(G_augmented, plot_dir=plot_dir, name_suffix="_augmented")
    # plot_degree_distribution(G_reduced, plot_dir=plot_dir, name_suffix="_reduced")

    # plot_degree_change(G, G_augmented, plot_dir=plot_dir, name_suffix="_augmented")
    # plot_degree_change(G, G_reduced, plot_dir=plot_dir, name_suffix="_reduced")
