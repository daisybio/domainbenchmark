#!/usr/bin/env python3
"""Probe whether cuML's RandomForest can be fed device arrays without changing results.

Diagnostic only. Answers the three questions that gate the optimisation
`random_forest.py` does *not* currently do: keeping X on the GPU across the
~200 candidate fits instead of handing cuML a host array each time, which
re-transfers the whole matrix per fit.

    Q1 (correctness gate) Is a fit on a cupy device array bitwise-identical to
       a fit on the equivalent host numpy array?
    Q2 (control)          Is the host path even reproducible run to run? If two
       identical host fits disagree, Q1 means nothing.
    Q3 (payoff)           How much wall time does the transfer actually cost
       over a realistic candidate count, and does the matrix even fit in VRAM?

Nothing here is optimised. It deliberately mirrors what
`RandomForestTrainer._search` does: same estimator class, `n_streams=1`,
`random_state` from `--seed`, candidates drawn by `ParameterSampler` from
`assets/RandomForest.json`, and `predict_proba(...)[:, 1]` as the observable
(that is what `average_precision_score` ranks candidates on, so it is the thing
that must not move).

Run on a GPU node, inside the GPU container:

    srun --partition=<gpu queue> --gres=gpu:1 --mem=64G --time=1:00:00 --pty \\
      apptainer exec --nv /nfs/scratch/k.pelz/sandbox/domainbenchmark-gpu \\
        python bin/probe_cuml_device.py --rows 60000 --cols 9174 --candidates 5

`--cols 9174` is the real all-feature width (from the
`external_test_neural_network_all` log). Start with `--rows 60000` -- that is
~2.2 GB, enough to be representative and small enough to fail fast. Then repeat
with `--rows 283000` (the real train height: 141545 instances x 2 orientations)
to get the true VRAM answer and the true timing.

Interpreting the result:

  * Q1 mismatch  -> do NOT make the change. The device path is a different
                    estimator as far as the benchmark is concerned, and every
                    number in the report would shift for a speedup.
  * Q2 mismatch  -> stop. `n_streams=1` is not delivering a reproducible forest
                    on this cuML build, which is a bigger problem than transfer
                    cost and invalidates the pipeline's reproducibility claim.
  * Q1 clean, transfer share small -> not worth the risk; leave it alone.
  * Q1 clean, transfer share large -> worth doing, and this script is the
                    regression test for it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rows", type=int, default=60000,
                    help="training rows (real all-feature train height is ~283000)")
    ap.add_argument("--cols", type=int, default=9174,
                    help="feature width (9174 = the real all-feature combo)")
    ap.add_argument("--opt_rows", type=int, default=13903,
                    help="validation rows (13903 = the real external_test value)")
    ap.add_argument("--candidates", type=int, default=5,
                    help="how many grid candidates to time; the real search runs 67 per "
                         "balance method")
    ap.add_argument("--seed", type=int, default=42,
                    help="same as params.seed, so the estimator is configured as in production")
    ap.add_argument("--grid", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets" / "RandomForest.json",
                    help="the real hyperparameter grid")
    args = ap.parse_args()

    try:
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: cupy/cuML unavailable ({type(exc).__name__}: {exc}).")
        print("Run this inside the GPU container on a GPU node, with --nv.")
        return 2

    from sklearn.model_selection import ParameterSampler

    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU: {cp.cuda.Device().id}  VRAM free {_fmt_bytes(free)} of {_fmt_bytes(total)}")

    grid = json.loads(args.grid.read_text())["model_parameters"]
    grid = {k: v for k, v in grid.items() if k != "balance_method"}
    candidates = list(ParameterSampler(grid, args.candidates, random_state=args.seed))

    # Synthetic, but shaped like the real thing and with a learnable signal so
    # the forest actually splits (a forest on pure noise can return constant
    # probabilities, which would make Q1 pass vacuously).
    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.rows, args.cols), dtype=np.float32)
    y = (x[:, :8].sum(1) + 0.5 * rng.standard_normal(args.rows) > 0).astype(np.int32)
    x_opt = rng.standard_normal((args.opt_rows, args.cols), dtype=np.float32)

    nbytes = x.nbytes + x_opt.nbytes
    print(f"X train {x.shape} + X opt {x_opt.shape} = {_fmt_bytes(nbytes)} host")
    if nbytes > free * 0.6:
        print(f"WARNING: {_fmt_bytes(nbytes)} against {_fmt_bytes(free)} free VRAM. The "
              "device path needs both resident plus the forest; expect an OOM.")

    def fit_predict(params, on_device):
        """One candidate, exactly as _search configures it. Returns proba[:, 1]."""
        if on_device:
            xt, yt, xo = cp.asarray(x), cp.asarray(y), cp.asarray(x_opt)
        else:
            xt, yt, xo = x, y, x_opt
        clf = RandomForestClassifier(**params, random_state=args.seed, n_streams=1)
        clf.fit(xt, yt)
        proba = clf.predict_proba(xo)[:, 1]
        out = cp.asnumpy(proba) if on_device else np.asarray(proba)
        del clf, xt, yt, xo
        cp.get_default_memory_pool().free_all_blocks()
        return out

    # ---- Q2: is the host path reproducible at all? --------------------------
    print("\n--- Q2: host reproducibility (control) ---")
    p = candidates[0]
    a = fit_predict(p, on_device=False)
    b = fit_predict(p, on_device=False)
    q2 = np.array_equal(a, b)
    print(f"two identical host fits bitwise equal : {q2}")
    if not q2:
        d = np.abs(a - b)
        print(f"  max |diff| = {d.max():.3e}, differing elements = {int((d > 0).sum())}")
        print("  STOP. n_streams=1 is not producing a reproducible forest on this")
        print("  build. That breaks the pipeline's reproducibility claim on its own,")
        print("  independently of the transfer question. Report this before anything")
        print("  else; the device comparison below is meaningless without it.")
        return 1

    # ---- Q1: host vs device ------------------------------------------------
    print("\n--- Q1: host vs device equivalence (the gate) ---")
    all_equal = True
    for i, params in enumerate(candidates, 1):
        host = fit_predict(params, on_device=False)
        dev = fit_predict(params, on_device=True)
        same = np.array_equal(host, dev)
        all_equal &= same
        label = ", ".join(f"{k}={params[k]}" for k in sorted(params))
        if same:
            print(f"  [{i}/{len(candidates)}] identical    | {label}")
        else:
            d = np.abs(host.astype(np.float64) - dev.astype(np.float64))
            print(f"  [{i}/{len(candidates)}] DIFFERS      | {label}")
            print(f"      max |diff| = {d.max():.3e}, "
                  f"differing = {int((d > 0).sum())} / {d.size}")
    print(f"\nall {len(candidates)} candidates bitwise identical : {all_equal}")

    # ---- Q3: what the transfer costs --------------------------------------
    print("\n--- Q3: transfer cost ---")
    t0 = time.monotonic()
    for params in candidates:
        fit_predict(params, on_device=False)
    host_s = time.monotonic() - t0

    # Device path as it would actually be written: transfer ONCE, reuse.
    xt, yt, xo = cp.asarray(x), cp.asarray(y), cp.asarray(x_opt)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.monotonic()
    for params in candidates:
        clf = RandomForestClassifier(**params, random_state=args.seed, n_streams=1)
        clf.fit(xt, yt)
        _ = cp.asnumpy(clf.predict_proba(xo)[:, 1])
        del clf
    cp.cuda.runtime.deviceSynchronize()
    dev_s = time.monotonic() - t0
    del xt, yt, xo
    cp.get_default_memory_pool().free_all_blocks()

    per_cand = (host_s - dev_s) / max(len(candidates), 1)
    print(f"  host array, re-transferred per fit : {host_s:8.2f} s "
          f"({host_s / len(candidates):.2f} s/candidate)")
    print(f"  device array, transferred once     : {dev_s:8.2f} s "
          f"({dev_s / len(candidates):.2f} s/candidate)")
    if host_s > 0:
        print(f"  transfer share of the search       : "
              f"{100.0 * (host_s - dev_s) / host_s:5.1f}%")
    print(f"  extrapolated saving over 201 fits  : {per_cand * 201 / 60:.1f} min "
          "(the real search is 67 candidates x 3 balance methods)")

    print("\n--- verdict ---")
    if not all_equal:
        print("Q1 FAILED. Do not move X to the device: the device path scores")
        print("candidates differently, so it can pick a different model and every")
        print("metric in the report shifts. The transfer cost is not worth that.")
        return 1
    print("Q1 passed. Whether to do it is now purely the Q3 number above -- and")
    print("check the VRAM line at the top against --rows 283000 before committing:")
    print("the saving is irrelevant if the real matrix does not fit resident.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
