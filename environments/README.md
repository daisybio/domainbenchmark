# daisybio/domainbenchmark conda environments

Two environments are the single source of truth for runtime dependencies.

| File          | Purpose                          | Container tag                       |
|---------------|----------------------------------|-------------------------------------|
| `general.yml`    | CPU env (eval, graph models, IO) | `konstantinpelz/domainbenchmark-general:1.0.0` |
| `ml.yml`      | GPU env (ML, RF, cuML, torch)    | `konstantinpelz/domainbenchmark-gpu:1.0.0`   |

The `docker/Dockerfile.base` and `docker/Dockerfile.ml` images mirror these
files. When pinning a package version here, update the corresponding line in
the Dockerfile and re-tag the image (bump the patch component when the env
changes).

Module-local `environment.yml` duplicates were removed in favour of these
canonical files. Modules now reference them via:

```groovy
conda "${projectDir}/environments/general.yml"   // or ml.yml
```

## Reconciliation choices

When the same package is in both files, versions match unless the GPU env
needs a CUDA-built wheel. Notable picks:

- `python=3.13.3` (base) / `python=3.12.2` (ml — pinned to match RAPIDS
  wheel availability for cuda 12.8).
- `pandas=2.3.0` (base) vs `pandas=2.2.3` (ml — pinned to RAPIDS-compatible
  series).
- `scikit-learn=1.7.2` (base) vs `1.6.1` (ml — pinned to cuML interop).
- `pytorch=2.7.0` is installed via pip in the GPU env using the upstream
  `https://download.pytorch.org/whl/cu128` index, since the conda channel
  lags the cuda-12.8 wheels.
