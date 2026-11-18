# kgiddi rewrite plan

Status as of 2026-09-04, `threshold = 1` chosen:

| step | state | verification |
|---|---|---|
| Step 0 probe | done | see the two result sections below |
| `threshold` 3 -> 1 | **done** (`assets/kgiddi.json`, `assets/kgiddi_random.json`) | probe |
| R3 chi2 one-pass + `select_best_ddis_per_group` bucketing | **done** | V2: 300 randomised interactomes, bitwise-identical `group_ddi_chi2` (order included) and identical `best_ddis_per_group` at cutoffs 0.0/0.5/0.9/1.0 |
| R1 GO-term balls replacing `protein_distances` | **done** | V3: 200 random GO graphs x 4 thresholds, `build_close_protein_index` == the old all-pairs -> `close` pipeline exactly |
| R4 `iterrows` sites | **done** (2 of 3) | order-preserving by construction |
| R4 `extract_go_guided_ddi_subgraphs` inversion | **deferred** — see below | — |
| R2 union-find on GO-term pairs | **done**, on the *stated* relation (self-pair bug fixed, by decision 2026-09-04) | V4: 1200 cases (400 GO graphs x 3 thresholds), partition-identical to brute force on the stated relation |
| V1 end-to-end gate on `minimal_leakage` | not run | needs R2 |
| V5 `test_graph` profile | not started | — |

Two things removed outright along the way: the `filtered_distances` loop in
`preprocessing` (walked the entire all-pairs dict, result never read) and the
`KGIDDI_LEGACY_BUILD` branch with `functionally_similar` (it took
`protein_distances`, which no longer exists).

`extract_go_guided_ddi_subgraphs` is deferred on purpose. Inverting it is easy,
but `process_go_term` does `nodes = list(subgraph.nodes)` and feeds that order
to `nx.to_numpy_array` and then `approx_bimax` — so **node insertion order
changes the biclusters and therefore the predictions.** Today that order comes
from iterating `set(domain_lists[0])`, i.e. salted hash order pinned only by
`PYTHONHASHSEED=0`. Any inversion has to sort the nodelist at the same time
(which CLAUDE.md's rule on set iteration already asks for), and that is a
second output change on top of the threshold. Worth doing, not worth bundling.

## Why

On the 2026-09-03 cluster run, six `GRAPH_MODEL` tasks (kgiddi and
kgiddi_random on `external_test`, `external_test_hcni`, `random`) ran for over
20 h and produced nothing before SLURM killed them. ddiparsimony on the same
databases finished in 1.8 h. Evidence, from a stuck task's `.command.log`
(`external_test_hcni`, kgiddi_random):

```
13:43:42  After limiting: 1785508 PPIs, 63004 PD, 151371 PGO, 120317 DDIs
13:43:51  Permuting GO term assignments...
16:21:58  Number of GO terms in go_proteinlist: 3564          <- 2 h 38 m gap
          pd_mapping has 47408 entries
16:24:54  After preprocessing test data | Memory usage: 150258.81 MB
16:24:55  ----- I: Build DDI network -----
          <17 h of silence, then SIGKILL>
```

Three separate problems, in the order they bite:

| # | Site | Cost observed | Cause |
|---|------|---------------|-------|
| P1 | `kgiddi.py:529` double loop | 2 h 38 m | `for i in range(n_proteins): for j in range(i+1, n_proteins)` at n_proteins = 47408 = 1.12e9 Python iterations, each a numpy fancy-index min |
| P2 | `protein_distances` dict | 150 GB of a 160 GB cap | the MF GO graph is essentially one component, so `np.isfinite(min_dist)` filters almost nothing and ~1.1e9 pairs get stored at ~100 B each |
| P3 | `build_ddi_network` candidate loop | >17 h, never finished | 1.78M PPIs x mean \|close[p]\| (thousands) x mean \|ppi_by_protein[c]\| (~75) = ~1e11 iterations. The inverted index that replaced the O(N^2) scan has no selectivity at this protein count |
| P4 | `kgiddi_functions.py:188` `compute_group_ddi_chi2` | not reached in 2026-09-03 | per-DDI full scan over all PPIs, plus an `outside_ppis` list rebuilt per DDI |

Regression origin: commit `5c64c40` moved `params.ppi_score_cutoff` from 900 to
400. Split interactomes went from ~2-8k proteins to 36-66k (see the
`benchmark-db-scale` memory). `minimal_leakage` (8358 proteins) still completes,
which is why those graph tasks were `-resume` cache hits and only the large
databases wedged.

## Step 0 — probe first, do not optimise blind

`bin/probe_go_term_graph.py` (already committed) answers the one question that
decides whether any of this is worth building:

> At `threshold`, does the distance-<=threshold graph over the GO terms a split
> actually uses collapse into a single connected component?

If it does, every PPI in the split lands in one union-find cluster,
`build_ddi_network` raises its existing `union-find collapsed all N PPIs into a
single cluster` error, and kgiddi cannot produce a rankable result on that
database at that threshold no matter how fast the code is. The thing to change
would then be `threshold` (or kgiddi's applicability to these splits), not the
implementation.

Run:

```bash
apptainer exec /nfs/scratch/k.pelz/sandbox/domainbenchmark-general \
    python bin/probe_go_term_graph.py \
        --db .../databases/external_test/test.sqlite3 \
        --db .../databases/minimal_leakage/test_balanced.sqlite3 \
        --go_graph /nfs/data/CoBiNet_Masterpraktikum/go-basic.obo \
        --ppi_score_cutoff 400 \
        --threshold 1 --threshold 2 --threshold 3 --threshold 4
```

Decision table on the reported `upper bound on PPI clusters` for
`external_test/test` at threshold 3:

| Probe says | Action |
|---|---|
| `<= 1` | Stop. Rewriting `build_ddi_network` cannot help. Re-calibrate `threshold` (the probe's sweep shows which value gives a non-degenerate partition), or record that kgiddi does not apply at `ppi_score_cutoff = 400`. |
| a handful (2-20) | Proceed, but the chi2 strata will be very coarse. Worth checking against `minimal_leakage`, which currently completes, to see what cluster count kgiddi normally works at. |
| hundreds+ | Proceed with the full plan below. |

`minimal_leakage/test_balanced` is in the command as a control: it is the split
that already completes, so its numbers are the reference for "healthy".

### Step 0 result (2026-09-04) — `threshold = 3` is the primary finding

| threshold | `external_test/test` components (of 3564 terms) | realised component pairs | largest pair | `minimal_leakage` components (of 1090) |
|---|---|---|---|---|
| 1 | **1156** (top: 662, 596, 220, 120, …), 1030 singletons | 35586 | 35.9% | 530, 448 singletons |
| 2 | 146, but **3171 in one** | 1267 | dominant | 125, but 826 in one |
| 3 | **9**, sizes `[3551, 5, 2, 1x6]`, mean degree 39.9 | **12** | ~everything | 30, sizes `[1038, 12, 6, …]` |
| 4 | **2**, sizes `[3563, 1]` | 2 | ~everything | 6, sizes `[1085, 1x5]` |

At the configured `threshold = 3`, 3551 of 3564 used GO terms sit in a single
component with mean degree 39.9. Two consequences, and they explain both
observed failures at once:

* `close[p]` is essentially every protein, so nearly all 1.14e9 protein pairs
  have a finite distance <= 3. That *is* P2's 150 GB and P1's 2 h 38 m: they are
  not incidental inefficiencies, they are what this threshold asks for.
* Only 12 component pairs are realised and one covers effectively the whole
  interactome, so the union-find welds nearly all 1.78M PPIs into one cluster.
  chi2 then has no outside stratum for any DDI in it.

`threshold = 1` is the only column with genuine stratification at this scale:
1156 components, 35586 realised component pairs, largest covering 35.9% of the
interactome. `threshold = 2` is already dominated (3171 of 3564 in one
component).

**`minimal_leakage` has the same shape at `threshold = 3`** (1038 of 1090 terms
in one component, 55 realised pairs). That split *completes* today, so the
kgiddi output currently in the report is very likely already near-degenerate --
one dominant cluster. This is a finding about the published numbers, not only
about runtime, and it should be checked against the cluster count that split's
`.command.log` reports before the existing kgiddi results are used.

So step 0's gate has fired, with a different answer than either branch of the
table above: the implementation is worth fixing (`threshold = 1` needs R1 and R2
just as much -- `protein_distances` is materialised before any thresholding), but
**`threshold` has to be re-calibrated first, or the fast version will faithfully
reproduce a degenerate partition.** That is a science decision.

### Step 0, second pass (2026-09-04) — cluster bounds, and the threshold verdict

The first pass counted term-pair occurrences instead of distinct PPIs (hence
"196%" in its output) and reported only a loose ceiling. Re-run with both bounds:

| threshold | `external_test/test` clusters (low–high) | largest cluster share | `minimal_leakage/test_balanced` (low–high) | largest share |
|---|---|---|---|---|
| 1 | **6140 – 66999** | 21.5% | **1453 – 10379** | 7.6% |
| 2 | 257 – 1502 | 96.0% | 129 – 587 | 84.0% |
| 3 | **5 – 14** | **100.0%** | 15 – 66 | 97.1% |
| 4 | **1 – 2** → probe flags DEGENERATE | 100.0% | 6 – 10 | 99.9% |

Verdict, and it is unambiguous:

* **`threshold = 3` (the configured value) is unusable on `external_test`.** One
  cluster holds 1784836 of 1785508 PPIs — 100.0%. It does not trip the
  `len(clusters) == 1` guard (there are ~5), so it would not raise; it would
  produce a report in which chi2 has a 672-PPI outside stratum to contrast the
  entire interactome against. Every DDI in the dominant cluster gets
  essentially the same chi2, so nothing is rankable.
* **`threshold = 4` is provably degenerate** — lower bound 1.
* **`minimal_leakage` at `threshold = 3` is 97.1% one cluster.** That split
  completes today, so **the kgiddi numbers currently in the report are already
  near-degenerate.** Confirm against the `Number of functionally similar PPI
  clusters (Union-Find) for threshold 3:` line in that task's `.command.log`
  before those results are used for anything.
* **`threshold = 1` is the only setting with real stratification at this
  scale**: 6140+ clusters on `external_test` with the largest at 21.5%, 1453+ on
  `minimal_leakage` with the largest at 7.6%. `threshold = 2` is already
  dominated (96% / 84%).

`threshold = 1` means the two proteins' most-specific MF terms are identical or
directly parent/child. That is a strict reading of "functionally similar", and
plausibly the intended one — `3` only became permissive because
`ppi_score_cutoff = 400` grew the protein set ~10x, which grew the used-term set
to 3564 with mean degree 39.9 at radius 3.

Note that R1 is required regardless of the threshold: `protein_distances` is
materialised in full *before* any thresholding, so it is 1.14e9 entries and
~114 GB at `threshold = 1` too. R2 gets much cheaper at `threshold = 1` (mean
degree 1.5) but is still needed — the current candidate loop is
`1.78M x |close[p]| x deg`.

## The rewrite

Everything below rests on one observation: **`protein_distances` is never read
as a distance.** Its only two consumers are

* `build_ddi_network`, which immediately thresholds it into
  `close[p] = {q : d(p,q) <= threshold}`; and
* `functionally_similar`, reached only under `KGIDDI_LEGACY_BUILD=1`.

So the distances need not be computed, let alone stored. And both the
thresholded relation and the PPI-similarity relation factor through GO terms,
of which there are only ~3564 in play.

### R1 — replace `protein_distances` with GO-term balls (fixes P1 + P2)

Definitions, all exact restatements of the current semantics:

* `T(p)` = protein `p`'s most-specific GO terms (what `protein_go_terms`
  already holds).
* `ball(t)` = `{t' in used_terms : goDist(t, t') <= threshold}`, computed by a
  radius-limited BFS from each used term over the undirected MF graph. Paths may
  leave the used set, exactly as the all-pairs matrix allowed. 3564 bounded BFS
  runs; milliseconds.
* `E(p)` = union of `ball(t)` over `t in T(p)`.

Then `dist(p, q) <= threshold` iff `T(q) ∩ E(p) != {}`. Proof: `dist(p,q)` is
defined as `min` over `t1 in T(p)`, `t2 in T(q)` of `goDist(t1,t2)`, so the min
is `<= threshold` iff some `t2 in T(q)` sits within `threshold` of some
`t1 in T(p)`, i.e. iff `t2 in ball(t1) ⊆ E(p)`.

Changes:

* `get_go_info` stops returning `dict(nx.all_pairs_shortest_path_length(...))`.
  It returns `go_levels` and the undirected graph; the ball computation moves
  into `preprocessing`, which is the first point that knows the used term set.
* `preprocessing` returns the thresholded neighbour structure instead of
  `protein_distances`. It therefore has to take `threshold` as an argument
  (currently only `run_kgiddi` and `build_ddi_network` see it).
* Delete `kgiddi.py:453` `filtered_distances` — it walks the entire all-pairs
  dict and the result is never read. Dead on arrival.
* `functionally_similar` and the `KGIDDI_LEGACY_BUILD` branch lose their input.
  Either drop both (the branch exists as a parity escape hatch for an
  optimisation we are about to replace anyway) or keep them behind a flag that
  reconstructs distances on demand. **Recommend dropping**, and pinning parity
  with the equivalence test in step V1 instead.

Expected: P1's 2 h 38 m -> seconds. P2's 150 GB -> a few hundred MB.

### R2 — union-find on GO-term pairs, not PPI pairs (fixes P3)

The relation `build_ddi_network` computes the transitive closure of is

```
ppi(a,b) ~ ppi(c,d)  iff  (dist(a,c) <= T and dist(b,d) <= T)
                       or (dist(a,d) <= T and dist(b,c) <= T)
```

Substituting R1's characterisation, this is a statement about **unordered pairs
of GO terms**: it holds iff there exist `t1 in T(a)`, `t2 in T(b)`,
`t3 in T(c)`, `t4 in T(d)` with `t1~t3` and `t2~t4`, where `t~t'` means
`goDist <= T`. The crossed clause is the same statement against the reversed
pair, so canonicalising term-pairs as unordered absorbs it. (PPIs are already
stored as sorted tuples, so this is consistent.)

Construction:

1. For each PPI `(a,b)`, its **realised term-pairs** are
   `{ {t1,t2} : t1 in T(a), t2 in T(b) }`.
2. Union each PPI with each of its realised term-pairs. Sound because two PPIs
   realising the *same* term-pair are always similar: `goDist(t,t) = 0 <= T`.
3. Union realised term-pairs `{t1,t2}` and `{t3,t4}` when
   `(t1~t3 and t2~t4)` or `(t1~t4 and t2~t3)`.

Components over `PPIs ∪ realised term-pairs`, restricted to the PPI elements,
equal the components of the original relation. Both directions are immediate:
a direct similarity is witnessed by adjacent term-pairs (step 3), and any path
through a shared or adjacent term-pair is a chain of genuine similarities.

Step 3 is the tensor square of the term-adjacency graph, so it does **not** need
enumerating: within one (component-of-t1, component-of-t2) block everything
merges into a single class, except when *both* components are bipartite, where
it splits in two by the parity of the two sides (Weichsel). The probe already
reports exactly these numbers, which is why its output doubles as a prediction
of the cluster count.

I would still **not** ship the parity shortcut on its own. Recommended
implementation: group realised term-pairs by component pair, then run an
explicit BFS *inside each block* over the adjacency of step 3. Blocks are small,
the cost is bounded by `sum over blocks of (pairs_in_block x |ball|^2)`, and
correctness needs no theorem. Use the parity formula only as a cross-check in
the test, where a mismatch is a signal rather than a silent wrong answer.

Cost: `sum over PPIs of |T(a)|*|T(b)|` — about 7M operations for
`external_test` — versus ~1e11.

### The self-pair question (blocks R2)

R2 cannot be made byte-identical to the current code, and the reason is a
latent bug rather than anything about the rewrite.

`protein_distances` only ever held `i < j` pairs, so `(p, p)` was never in it.
`close[p]` therefore never contained `p`, in **both** old paths — the
inverted-index one and the `KGIDDI_LEGACY_BUILD` one, where
`functionally_similar` looked up `tuple(sorted((a, a)))`, missed, and got
`inf`. So what the code actually computes is

```
ppi(a,b) ~ ppi(c,d)  iff  (a != c and b != d and dist(a,c) <= T and dist(b,d) <= T)
                      or  (a != d and b != c and dist(a,d) <= T and dist(b,c) <= T)
```

not the relation `functionally_similar`'s docstring describes. `dist(p, p) = 0`,
so under the stated relation two PPIs that share a protein are always similar;
under the implemented one they are related only through their *other* endpoints.

That extra `a != c` clause is a condition on protein *identity*, so it does not
factor through GO terms — which is exactly what R2 needs in order to move the
union-find onto term pairs. **R2 can implement the stated relation or the
current one, but the fast algorithm can only express the stated one.**

Consequences either way:

* **Stated relation** (R2 as designed): more unions, so fewer and larger
  clusters than the current code would give at the same threshold. The probe's
  cluster bounds were computed this way, so the `threshold = 1` numbers
  (6140–66999 on `external_test`) already reflect it — non-degenerate, so this
  is viable.
* **Current relation**: R1 alone must carry it, keeping the existing
  protein-keyed candidate loop. Whether that finishes at `threshold = 1` depends
  on `|close[p]|`, which R1 now logs (`GO-similarity index at threshold 1: ...
  max neighbourhood ...`). If the mean neighbourhood is small the existing loop
  may be fast enough and R2 is unnecessary; if it is thousands, the loop is
  `1.78M x |close| x deg` and still infeasible.

**Decision (2026-09-04): implement the stated relation, i.e. fix the bug.**
R2 is built that way. Consequences to expect on the next run, on top of the
threshold change:

* Two PPIs sharing a protein are now related through it, so there are more
  unions and therefore **fewer, larger clusters** than the old code produced at
  the same threshold. Measured on 1200 randomised cases, the two relations give
  a different partition in **22%** of them -- this is not a corner case.
* The `threshold = 1` cluster bounds from the probe (6140-66999 on
  `external_test`) were computed under the stated relation, so they are the ones
  that apply.
* `build_close_protein_index` is gone with the protein-keyed loop. Nothing
  materialises a protein-pair structure any more, so the log line it added
  (`GO-similarity index at threshold ...`) is replaced by
  `Similarity index: N realised GO-term pairs over M annotated terms, mean ball
  size ...` from `cluster_similar_ppis`.

V1's original form -- "byte-identical `predictions_test_balanced.parquet`
against the old code" -- is therefore **not** the right gate any more and has
been dropped. V4 (partition-identity against brute force on the stated
relation) is what establishes correctness; the old code is a different
specification, not a reference.

### R3 — `compute_group_ddi_chi2` in one pass (fixes P4)

Clusters partition the deduped PPI set, so with `total[ddi]` = the number of
PPIs *anywhere* whose domain cross-product contains `ddi` (one pass over all
PPIs), and `A` the current in-group count:

```
B = total[ddi] - A                  # PPIs outside the group with this DDI
C = len(group) - A                  # unchanged
D = (N - len(group)) - B            # PPIs outside the group without this DDI
```

`len(outside_ppis) == N - len(group)` because every group is a subset of
`ppi_interactions` and the groups are disjoint. All four are counts, and every
product and sum in `numerator`/`denominator` stays below 2^53 for realistic `N`,
so the float64 arithmetic is exact and `chi2` comes out **bit-identical**.

Two things to keep exactly as they are:

* `A` counting: a PPI contributes each `(d1,d2)` at most once because `d1s` and
  `d2s` are sets. Do not switch to multiplicities.
* Iteration order of `for ddi in group_ddi_set`. It looks incidental but is not:
  `score_test_split` builds `chi2_scores` as a dict comprehension, so for a DDI
  appearing in two groups the *last* one wins. Reordering would change the
  scores. `PYTHONHASHSEED=0` makes the current set order deterministic; leave
  the iteration untouched and only replace the `B`/`D` computation.

Also fix `select_best_ddis_per_group`: it does
`[x for x in group_ddi_chi2 if x["group_name"] == group_name]` inside the group
loop, i.e. O(groups x entries). Bucket once into a dict, preserving
within-group insertion order.

### R4 — small, uncontroversial

* `build_ddi_network`: `for _, row in pd_df.iterrows()` -> `zip` over `.values`.
* `score_test_split`: `for _, row in ddi_df_test.iterrows()` (138642 rows for
  `external_test`) -> `zip` over `.values`. Order preserved, which matters
  because `random_jitter` is indexed positionally against `ddi_actual`.
* `extract_go_guided_ddi_subgraphs`: nested `for dom1 in d1: for dom2 in d2`
  with a set-membership test per pair. Invert — iterate `ddi_network_edges` and
  bucket by GO term.

## Verification (V)

`params.graph_models = ''` in the test profile, so **nf-test carries no signal
for any of this**. Verification has to be built as part of the work, not
inherited.

* **V1 — equivalence on a split that completes.** `minimal_leakage/test_balanced`
  (62228 PPIs, 8358 proteins) finishes under the current code. Run old and new
  end to end and require: identical cluster count, identical
  `connected_components` as a set of frozensets, identical `group_ddi_chi2`
  (list order included), and byte-identical `predictions_test_balanced.parquet`.
  This is the gate. Do not proceed on `external_test` until it passes.
* **V2 — R3 in isolation.** Unit test: random small interactomes, assert the
  new `B`/`D` equal the old loop's for every `(group, ddi)`.
* **V3 — R1 in isolation.** Unit test on a small GO subgraph: assert
  `T(q) ∩ E(p) != {}` agrees with `protein_distances[(p,q)] <= threshold` for
  every pair, across thresholds 1-4.
* **V4 — R2 in isolation.** On `minimal_leakage`, compare against the
  `KGIDDI_LEGACY_BUILD=1` O(N^2) path (if kept for this purpose) or against a
  brute-force enumeration on a subsampled interactome. Cross-check the block
  BFS against the parity formula and against `probe_go_term_graph.py`'s
  reported ceiling.
* **V5 — add graph models to a test profile.** The reason none of the last three
  graph-model fixes (sparse LP, seeded LP, spawn pool) has a regression test is
  that `graph_models = ''`. A `test_graph` profile with a tiny interactome and
  `chi_square_cutoff`/`bicluster_cutoff` pinned would give the next change a
  snapshot to break. Worth doing while the equivalence harness from V1 is fresh.

## Order of work

1. Step 0 probe. **Gate.**
2. R3 + `select_best_ddis_per_group` — smallest, bit-identical, independently
   testable (V2), and it is the next wall even if R1/R2 land first.
3. R1 (+V3). Removes the 2 h 38 m and the 150 GB.
4. R2 (+V4). The 17 h.
5. R4.
6. V1 as the gate on the whole thing, then V5.

## Open questions

* `threshold = 3` is hardcoded in `assets/kgiddi.json` and was calibrated on a
  ~10x smaller protein set. The probe's sweep will say whether it still gives a
  usable partition. If it does not, that is a science decision, not a code one.
* `assets/kgiddi.json` carries a 6x4 `chi_square_cutoff` x `bicluster_cutoff`
  grid that never executes, because `optimized` is set and `run_kgiddi` takes
  `training = False`. Either delete the unused keys or gate the search behind an
  explicit flag; right now the file reads as though a search happens. Same for
  `assets/ddiparsimony.json`. Note the asymmetry this creates in the benchmark:
  NN gets 100 candidates and RF 200, both selected on train+validation, while
  the graph models get two hardcoded numbers.
