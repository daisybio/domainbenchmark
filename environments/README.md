# daisybio/domainbenchmark conda environments

These two files are the single source of truth for runtime dependencies. The
Dockerfiles install them directly, so the `conda` profile and the containers
get the same versions.

| File          | Purpose                          | Dockerfile                  | Container tag                                  |
|---------------|----------------------------------|-----------------------------|------------------------------------------------|
| `general.yml` | CPU env (eval, graph models, IO) | `docker/Dockerfile.general` | `konstantinpelz/domainbenchmark-general:1.0.1` |
| `ml.yml`      | GPU env (NN, RF, cuML, torch)    | `docker/Dockerfile.gpu`     | `konstantinpelz/domainbenchmark-gpu:1.0.2`     |

After changing a yml, bump the image's patch version in the Dockerfile label,
in the module `container` directives and in the table above, then build from
the repo root (`.dockerignore` limits the context to `environments/`):

```bash
docker build -f docker/Dockerfile.general -t konstantinpelz/domainbenchmark-general:<ver> .
docker build -f docker/Dockerfile.gpu     -t konstantinpelz/domainbenchmark-gpu:<ver> .
```

Creating `ml.yml` with conda on a machine without a GPU needs
`CONDA_OVERRIDE_CUDA=13.4`.

`ml.yml` is pip-free: `pytorch` comes from conda-forge, not the PyTorch wheel
index, because the wheels bundle their own CUDA libraries and their
`libnvJitLink` shadowed the one cuML needs.
