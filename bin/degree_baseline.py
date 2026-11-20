#!/usr/bin/env python3
"""Train-positive degree baseline (W-15): score(A, B) = deg(A) x deg(B).

deg(F) = number of distinct positive DDIs of family F in `train.csv`
(undirected; a same-family positive (F, F) adds 1 to deg(F)); a family absent
from train's positives has degree 0. Model-free and threshold-free: it is a
ranking reference ("hub x hub") next to every model, reported by AUC and AP
only. Scores are never inverted.

Writes one `predictions_<variant>.parquet` per test split, one row per
canonical DDI of `<split>.csv` (`domain_a <= domain_b`), with
`domain_a, domain_b, true_interaction (int8), predicted_probability (float32)`
and no `predicted_interaction*` columns: there is no validation threshold.

The score is an integer carried in float32, which is exact only up to 2^24,
so a larger product fails instead of silently creating ties (EVAL_ONE casts
every score to float32).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FLOAT32_EXACT = 2 ** 24


def canonical_frame(csv_path: Path) -> pd.DataFrame:
    """One row per undirected DDI of the CSV; fails on a pair labelled both ways."""
    df = pd.read_csv(csv_path, dtype={"domain_1": str, "domain_2": str})
    if not df["interaction"].isin([0, 1]).all():
        raise ValueError(f"{csv_path}: interaction must be 0 or 1")
    a, b = df["domain_1"], df["domain_2"]
    lo = a <= b
    out = pd.DataFrame({
        "domain_a": a.where(lo, b),
        "domain_b": b.where(lo, a),
        "true_interaction": df["interaction"].astype(np.int8),
    })
    n_labels = out.groupby(["domain_a", "domain_b"])["true_interaction"].nunique()
    if (n_labels > 1).any():
        first = n_labels[n_labels > 1].index[0]
        raise ValueError(f"{csv_path}: DDI {first[0]}/{first[1]} is labelled both 0 and 1")
    return out.drop_duplicates(["domain_a", "domain_b"]).sort_values(["domain_a", "domain_b"]).reset_index(drop=True)


def degrees(train: pd.DataFrame) -> dict:
    """deg(F): distinct positive train DDIs touching F (a self-pair counts once)."""
    deg = {}
    for a, b in train.loc[train["true_interaction"] == 1, ["domain_a", "domain_b"]].itertuples(index=False):
        for f in {a, b}:
            deg[f] = deg.get(f, 0) + 1
    return deg


def score(frame: pd.DataFrame, deg: dict) -> pd.DataFrame:
    product = np.array(
        [deg.get(a, 0) * deg.get(b, 0) for a, b in zip(frame["domain_a"], frame["domain_b"])],
        dtype=np.int64,
    )
    if len(product) and product.max() > FLOAT32_EXACT:
        raise ValueError(
            f"degree product {product.max()} exceeds 2^24: float32 would merge distinct scores into ties."
        )
    out = frame.copy()
    out["predicted_probability"] = product.astype(np.float32)
    return out[["domain_a", "domain_b", "true_interaction", "predicted_probability"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ddi_path", type=Path, required=True, help="DDI_EXTRACTION's DDI/ directory")
    p.add_argument("--test", action="append", required=True, metavar="VARIANT=SPLIT",
                   help="One per test split, e.g. uniform_1_1=test_uniform_1_1")
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args(argv)

    deg = degrees(canonical_frame(args.ddi_path / "train.csv"))
    print(f"degree_baseline: {len(deg)} families with a positive train DDI, max degree {max(deg.values(), default=0)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sorted(args.test):
        variant, split = spec.split("=", 1)
        pred = score(canonical_frame(args.ddi_path / f"{split}.csv"), deg)
        out = args.out_dir / f"predictions_{variant}.parquet"
        pred.to_parquet(out, index=False, compression="zstd")
        print(f"degree_baseline: {split} -> {out}: {len(pred)} DDIs, "
              f"{int((pred['predicted_probability'] > 0).sum())} with a nonzero score, "
              f"max {pred['predicted_probability'].max() if len(pred) else 0:.0f}")


if __name__ == "__main__":
    main()
