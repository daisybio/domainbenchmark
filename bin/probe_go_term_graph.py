#!/usr/bin/env python3
"""Probe the GO-term similarity graph kgiddi's union-find actually runs on.

Diagnostic only -- reads nothing the pipeline writes and writes nothing the
pipeline reads. It answers the one question that decides whether optimising
`build_ddi_network` is worth doing at all:

    At `threshold`, does the distance-<=threshold graph over the GO terms this
    split actually uses collapse into a single connected component?

If it does, every PPI in the split lands in one union-find cluster, chi2 has no
outside stratum, and `build_ddi_network` raises `union-find collapsed all N
PPIs into a single cluster` no matter how fast we make it. The threshold (or
the method's applicability to this split) is then the thing to change, not the
implementation.

Why component *pairs* and not just components: kgiddi's PPI-similarity relation
is `ppi(a,b) ~ ppi(c,d)` iff `dist(a,c) <= T and dist(b,d) <= T` (or crossed).
Factored through GO terms that is a relation on *unordered pairs* of GO terms,
i.e. the tensor square of the graph this script builds. So the reachable number
of PPI clusters is bounded by the number of realised component pairs, which is
what the last section reports.

Reuses `kgiddi_functions.load_go_graph` and `load_data_gm.load_pgo` on purpose:
the probe is only meaningful if it builds the exact same graph and reads the
exact same GO annotations that `bin/kgiddi.py` does. Run it from a checkout so
those siblings import.

Usage (inside the general container, from the repo root):

    apptainer exec /nfs/scratch/k.pelz/sandbox/domainbenchmark-general \\
        python bin/probe_go_term_graph.py \\
            --db /path/to/databases/external_test/test.sqlite3 \\
            --go_graph /nfs/data/CoBiNet_Masterpraktikum/go-basic.obo \\
            --ppi_score_cutoff 400 \\
            --threshold 3

Add `--threshold 1 --threshold 2 --threshold 3 --threshold 4` to sweep, and
repeat `--db` to probe several splits in one load of the OBO file (parsing
go-basic.obo is the slow part).
"""

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import networkx as nx

# Same graph construction and same annotation loader kgiddi.py uses. If these
# imports fail, run from the repo root so bin/ is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kgiddi_functions import load_go_graph  # noqa: E402
from load_data_gm import load_pd_mapping, load_pgo, load_ppi  # noqa: E402


def used_go_terms(db_path: Path, go_graph, ppi_score_cutoff: int):
    """The GO terms kgiddi's `preprocessing` would end up keying `D` on.

    Mirrors `kgiddi.preprocessing` exactly: filter PPIs by confidence, keep the
    GO annotations of proteins that survive, then reduce each protein to its
    *most specific* terms (highest `level` in the GO graph). Those most-specific
    terms are the only ones `protein_distances` is ever computed between.
    """
    ppi_df = load_ppi(db_path)
    ppi_df = ppi_df[ppi_df["score"] >= ppi_score_cutoff].reset_index(drop=True)
    proteins_in_ppi = set(ppi_df["protein_1"]).union(set(ppi_df["protein_2"]))

    pgo_df = load_pgo(db_path)
    pgo_df = pgo_df[
        pgo_df["go_accession"].isin(go_graph.nodes)
        & pgo_df["uniprot_id"].isin(proteins_in_ppi)
    ].reset_index(drop=True)
    valid_proteins = set(pgo_df["uniprot_id"].unique())

    ppi_df = ppi_df[
        ppi_df["protein_1"].isin(valid_proteins)
        & ppi_df["protein_2"].isin(valid_proteins)
    ].reset_index(drop=True)
    proteins_in_ppi = set(ppi_df["protein_1"]).union(set(ppi_df["protein_2"]))

    # NOT re-filtered on this second `proteins_in_ppi`. `kgiddi.preprocessing`
    # assigns `pgo_df = pgo_filtered_df`, i.e. the frame filtered on the *first*
    # round of PPI proteins, and builds `protein_go_terms` from that. So its GO
    # term universe can include proteins that the second filter dropped from the
    # interactome. The probe has to reproduce that universe, not a tidier one,
    # or its component counts describe a graph kgiddi never builds.

    # `level` lives on the edges of the DiGraph load_go_graph builds, keyed by
    # the child (target) node -- same lookup kgiddi.get_go_info does.
    go_levels = {v: data["level"] for _, v, data in go_graph.edges(data=True)}

    protein_terms = {}
    for protein, group in pgo_df.groupby("uniprot_id"):
        best, chosen = -1, []
        for term in group["go_accession"].tolist():
            level = go_levels.get(term, float("inf"))
            if level > best:
                best, chosen = level, [term]
            elif level == best:
                chosen.append(term)
        protein_terms[protein] = chosen

    pd_df = load_pd_mapping(db_path)
    n_mapped = pd_df["uniprot_id"].isin(proteins_in_ppi).sum()

    return protein_terms, ppi_df, len(proteins_in_ppi), int(n_mapped)


def ball(graph, source, radius):
    """Nodes within `radius` undirected hops of `source`, source included."""
    seen = {source}
    frontier = deque([(source, 0)])
    while frontier:
        node, dist = frontier.popleft()
        if dist == radius:
            continue
        for nbr in graph[node]:
            if nbr not in seen:
                seen.add(nbr)
                frontier.append((nbr, dist + 1))
    return seen


def probe(protein_terms, ppi_df, terms, go_undirected, threshold):
    """Report components of the distance-<=threshold graph over `terms`."""
    term_index = {t: i for i, t in enumerate(terms)}

    # Edge (t1, t2) iff their GO-graph distance is <= threshold. Built by a
    # radius-limited BFS per used term over the *full* undirected GO graph --
    # paths may leave the used set, exactly as the all-pairs version allowed.
    g3 = nx.Graph()
    g3.add_nodes_from(range(len(terms)))
    for t in terms:
        for other in ball(go_undirected, t, threshold):
            j = term_index.get(other)
            if j is not None and j != term_index[t]:
                g3.add_edge(term_index[t], j)

    components = list(nx.connected_components(g3))
    comp_of = {n: i for i, c in enumerate(components) for n in c}
    sizes = sorted((len(c) for c in components), reverse=True)

    print(f"\n=== threshold = {threshold} ===")
    print(f"  used GO terms                 : {len(terms)}")
    print(f"  distance-<={threshold} edges among them : {g3.number_of_edges()}")
    print(f"  mean degree                   : "
          f"{2 * g3.number_of_edges() / max(len(terms), 1):.1f}")
    print(f"  connected components          : {len(components)}")
    print(f"  component sizes (top 15)      : {sizes[:15]}")
    print(f"  singletons                    : {sum(1 for s in sizes if s == 1)}")

    # Every PPI's realised (component, component) pairs, over the actual PPI
    # list. Counted **per distinct PPI**, not per term-pair occurrence: a PPI
    # with |T(a)| x |T(b)| term-pairs hits the same component pair repeatedly,
    # and counting occurrences made this figure exceed the PPI total.
    realised = Counter()
    ppis_without_terms = 0
    # blocks[i] = union-find parent over component pairs, keyed by the pair.
    parent: dict = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for p1, p2 in zip(ppi_df["protein_1"].values, ppi_df["protein_2"].values):
        t1s = protein_terms.get(p1)
        t2s = protein_terms.get(p2)
        if not t1s or not t2s:
            ppis_without_terms += 1
            continue
        touched = set()
        for a in t1s:
            for b in t2s:
                ca, cb = comp_of[term_index[a]], comp_of[term_index[b]]
                touched.add((min(ca, cb), max(ca, cb)))
        for block in touched:
            realised[block] += 1          # once per PPI, not once per term-pair
            parent.setdefault(block, block)
        # A PPI belongs to *all* the component pairs its term-pairs realise, so
        # it welds them into one union-find class. This is the step the first
        # version of this probe missed, and it is the one that matters: without
        # it the "upper bound" below reads as a cluster count when it is only a
        # ceiling.
        it = iter(touched)
        first = next(it)
        for block in it:
            union(first, block)

    bipartite = [nx.is_bipartite(g3.subgraph(c)) for c in components]
    both_bip = sum(1 for (ca, cb) in realised if bipartite[ca] and bipartite[cb])
    ceiling = len(realised) + both_bip
    floor = len({find(b) for b in parent})

    print(f"  PPIs (after filtering)        : {len(ppi_df)}")
    print(f"  PPIs with no usable GO terms  : {ppis_without_terms}")
    print(f"  realised component pairs      : {len(realised)}")
    print(f"  ... of which both bipartite   : {both_bip}")
    # Two bounds, because neither is tight on its own:
    #  * ceiling: PPIs in different component pairs can never be unioned, and
    #    one component pair splits in two only when both its components are
    #    bipartite (Weichsel). Loose, because it ignores PPIs that bridge pairs.
    #  * floor: after welding every component pair a shared PPI touches, and
    #    assuming each block is internally connected. The real count can only be
    #    higher (a block's *realised* term-pairs may not be connected to each
    #    other, and parity may split it), never lower.
    print(f"  => PPI clusters, lower bound  : {floor}")
    print(f"  => PPI clusters, upper bound  : {ceiling}")
    if floor <= 1:
        print("  !! DEGENERATE: the whole interactome welds into one cluster.")
        print("     chi2 gets no outside stratum, so build_ddi_network raises")
        print("     here however fast the implementation is. This threshold is")
        print("     unusable on this split.")
    ordered = realised.most_common(3)
    for (ca, cb), n in ordered:
        print(f"  component pair ({ca},{cb}) covers  : {n} / {len(ppi_df)} PPIs "
              f"({100.0 * n / max(len(ppi_df), 1):.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", action="append", required=True, type=Path,
                    help="split sqlite3 file; repeat to probe several splits")
    ap.add_argument("--go_graph", required=True, type=Path,
                    help="go-basic.obo (same file assets/kgiddi.json points at)")
    ap.add_argument("--ppi_score_cutoff", type=int, default=400,
                    help="same value as params.ppi_score_cutoff (default 400)")
    ap.add_argument("--threshold", action="append", type=int, default=None,
                    help="GO-similarity threshold; repeat to sweep (default 3)")
    args = ap.parse_args()

    thresholds = args.threshold or [3]

    print(f"Loading GO graph from {args.go_graph} ...")
    go_graph = load_go_graph(str(args.go_graph))
    go_undirected = go_graph.to_undirected()
    print(f"GO graph (molecular_function): {go_graph.number_of_nodes()} nodes, "
          f"{go_graph.number_of_edges()} edges")

    for db in args.db:
        print(f"\n################ {db} ################")
        protein_terms, ppi_df, n_proteins, n_pd = used_go_terms(
            db, go_graph, args.ppi_score_cutoff
        )
        terms = sorted({t for ts in protein_terms.values() for t in ts})
        print(f"  proteins in filtered interactome : {n_proteins}")
        print(f"  proteins with GO terms           : {len(protein_terms)}")
        print(f"  domain_protein_map rows kept     : {n_pd}")
        # What the current code would have materialised at this size.
        pairs = len(protein_terms) * (len(protein_terms) - 1) // 2
        print(f"  protein pairs the current loop visits: {pairs:,} "
              f"(~{pairs * 100 / 1e9:.0f} GB if all distances are finite)")
        for threshold in thresholds:
            probe(protein_terms, ppi_df, terms, go_undirected, threshold)


if __name__ == "__main__":
    main()
