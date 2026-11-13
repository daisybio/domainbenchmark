#!/usr/bin/env python3
"""
Metadata table for one (database, test variant), plus its `observed` fit.

Builds one row per canonical test DDI (domain_a <= domain_b, Pfam) from the
split's `<split>_instances.csv` and the run-wide metadata directory:

- domain_metadata.csv (instance_id, pfam_id, uniprot_id, numeric columns):
  averaged per Pfam domain over the split's instances, then emitted per DDI as
  `<col>_min` / `<col>_max` over its two domains -- symmetric, because which
  domain is `a` is arbitrary.
- ddi_metadata.csv (pfam_id_a/b, uniprot_id_a/b, instance_id_a/b, source,
  numeric af_* columns): the rows whose instance pair belongs to this split,
  averaged per DDI. Every numeric column is kept.

Missing values stay missing (no imputation); eval_enrichment counts the rows
they cost. The `observed` target (true label) is model-independent, so it is
fit here once rather than once per model.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from eval_enrichment import analyse, complete_cases, feature_columns
from load_data_gm import canonical_pair

DOMAIN_ID_COLS = {"instance_id", "pfam_id", "uniprot_id"}
DDI_ID_COLS = {
    "pfam_id_a", "pfam_id_b", "uniprot_id_a", "uniprot_id_b",
    "instance_id_a", "instance_id_b", "source",
}


def _canonical(df, a, b):
    pairs = [canonical_pair(x, y) for x, y in zip(df[a], df[b])]
    df = df.copy()
    df["domain_a"] = [x for x, _ in pairs]
    df["domain_b"] = [y for _, y in pairs]
    return df


def _numeric_columns(df, id_cols, name):
    missing = {c for c in id_cols if c.startswith(("pfam_id", "instance_id"))} - set(df.columns)
    if missing:
        raise SystemExit(f"[metadata] {name} lacks column(s) {sorted(missing)}")
    cols = sorted(c for c in df.columns if c not in id_cols)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < len(cols):
        print(f"[metadata] {name}: ignoring non-numeric column(s) {sorted(set(cols) - set(numeric))}")
    return numeric


def build_table(domain_md, ddi_md, instances):
    """One row per canonical DDI: domain_a, domain_b, label, sorted features."""
    if instances.empty:
        raise SystemExit("[metadata] The split has no instance pairs.")
    inst = _canonical(instances, "domain_1", "domain_2")
    inst["label"] = inst["interaction"].astype(int)
    labels = inst.groupby(["domain_a", "domain_b"])["label"].agg(["min", "max"])
    if (labels["min"] != labels["max"]).any():
        raise SystemExit("[metadata] A DDI carries both labels across its instance pairs.")
    table = labels["min"].rename("label").reset_index()

    # Domain level: mean per Pfam domain over this split's instances, then
    # min / max over the DDI's two domains (NaN if either is unknown).
    split_instances = set(inst["instance_1"]) | set(inst["instance_2"])
    dom = domain_md[domain_md["instance_id"].isin(split_instances)]
    dom_cols = _numeric_columns(domain_md, DOMAIN_ID_COLS, "domain_metadata.csv")
    per_domain = dom.groupby("pfam_id")[dom_cols].mean()
    a = per_domain.reindex(table["domain_a"]).to_numpy(dtype=float)
    b = per_domain.reindex(table["domain_b"]).to_numpy(dtype=float)
    for i, c in enumerate(dom_cols):
        table[f"{c}_min"] = np.minimum(a[:, i], b[:, i])
        table[f"{c}_max"] = np.maximum(a[:, i], b[:, i])

    # DDI level: the metadata rows of this split's instance pairs, either orientation.
    if len(ddi_md):
        ddi_cols = _numeric_columns(ddi_md, DDI_ID_COLS, "ddi_metadata.csv")
        clash = set(ddi_cols) & set(table.columns)
        if clash:
            raise SystemExit(f"[metadata] DDI-level column(s) {sorted(clash)} clash with domain-level ones.")
        members = {tuple(sorted(p)) for p in zip(inst["instance_1"], inst["instance_2"])}
        keep = [tuple(sorted(p)) in members for p in zip(ddi_md["instance_id_a"], ddi_md["instance_id_b"])]
        ddi = _canonical(ddi_md[keep], "pfam_id_a", "pfam_id_b")
        per_ddi = ddi.groupby(["domain_a", "domain_b"])[ddi_cols].mean().reset_index()
        table = table.merge(per_ddi, on=["domain_a", "domain_b"], how="left")
        # The mean runs over the pairs that have a value (e.g. only the
        # structure-covered ones). Count the DDIs that holds back pairs for, so
        # the report shows it instead of calling them complete.
        expected = inst.assign(pair=[tuple(sorted(p)) for p in zip(inst["instance_1"], inst["instance_2"])])
        expected = expected.groupby(["domain_a", "domain_b"])["pair"].nunique()
        ddi = ddi.assign(pair=[tuple(sorted(p)) for p in zip(ddi["instance_id_a"], ddi["instance_id_b"])])
        # Each column's mean is taken separately, so coverage is per column: a
        # DDI is partial if any column's value comes from some but not all pairs.
        partial = pd.Series(False, index=expected.index)
        for c in ddi_cols:
            n = ddi[ddi[c].notna()].groupby(["domain_a", "domain_b"])["pair"].nunique()
            n = n.reindex(expected.index).fillna(0)
            partial |= (n > 0) & (n < expected)
        n_partial = int(partial.sum())
    else:
        n_partial = 0

    table = table[["domain_a", "domain_b", "label"] + sorted(table.columns[3:])]
    return table, n_partial


def observed_result(table, threads=1, n_partial_pair_metadata=0):
    complete, used, dropped, counts = complete_cases(table, feature_columns(table))
    return {
        "counts": {"n_ddis": int(len(table)), **counts,
                   "n_ddis_partial_pair_metadata": n_partial_pair_metadata},
        "dropped_columns": dropped,
        "targets": {"observed": analyse("observed", complete[used], complete["label"].to_numpy(), threads)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--metadata", required=True, help="Directory with domain_metadata.csv and ddi_metadata.csv")
    p.add_argument("--instances", required=True, help="<split>_instances.csv from DDI_EXTRACTION")
    p.add_argument("--out_table", required=True)
    p.add_argument("--out_observed", required=True)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    read = lambda name: pd.read_csv(os.path.join(args.metadata, name))  # noqa: E731
    instances = pd.read_csv(args.instances, dtype={c: str for c in ("domain_1", "domain_2", "instance_1", "instance_2")})
    table, n_partial = build_table(read("domain_metadata.csv"), read("ddi_metadata.csv"), instances)
    table.to_csv(args.out_table, index=False)
    print(f"[metadata] {len(table)} DDIs, {len(table.columns) - 3} feature columns -> {args.out_table}")

    with open(args.out_observed, "w", encoding="utf-8") as fh:
        json.dump(observed_result(table, args.threads, n_partial), fh, indent=1)


if __name__ == "__main__":
    main()
