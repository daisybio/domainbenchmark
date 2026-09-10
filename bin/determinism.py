#!/usr/bin/env python3
"""One place to make a run reproducible.

Every entrypoint in ``bin/`` calls :func:`seed_everything` before it touches an
RNG. That covers the obvious global generators (``random``, ``numpy``,
``torch``) and, more importantly, the ones that silently reintroduce
run-to-run drift:

* ``PYTHONHASHSEED``. String hashing is salted per interpreter, so iterating a
  ``set`` of domain IDs yields a different order on every run -- which reorders
  training rows, the sample drawn by a class-balancing step, and the rows of a
  predictions parquet. The salt is fixed before the interpreter starts, so all
  this module can do is check it and complain; the Nextflow modules export
  ``PYTHONHASHSEED=0``.
* cuDNN autotuning (``benchmark``) picks a kernel by timing it, so the kernel
  -- and therefore the last bits of the result -- depends on machine load.
* A handful of ``torch`` ops have no deterministic implementation unless asked
  for one, and some cuBLAS matmuls reuse a workspace in an order-dependent way.

:func:`derive_seed` exists for the parallel loops: a worker must not be seeded
from its completion order or its position in an ``as_completed`` stream, so it
gets a seed derived from the run's master seed and its own stable identity.
"""

import hashlib
import os
import random
import sys

import numpy as np

HASH_SEED_ENV = "PYTHONHASHSEED"


def hash_seed_is_fixed() -> bool:
    """True when ``PYTHONHASHSEED`` pins the string-hash salt for this process."""
    value = os.environ.get(HASH_SEED_ENV)
    return value not in (None, "", "random")


def derive_seed(seed: int, *tokens) -> int:
    """A child seed for ``tokens``, stable and independent of iteration order.

    Used to seed parallel workers: ``derive_seed(42, run_idx)`` gives worker
    ``run_idx`` the same seed no matter which worker picks the task up, how many
    workers there are, or in what order their results come back.
    """
    payload = ":".join([str(seed)] + [str(token) for token in tokens]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


_warned_about_hash_seed = False


def seed_everything(seed: int, *, torch_deterministic: bool = True) -> None:
    """Seed every global RNG and switch off nondeterministic kernels.

    Safe to call repeatedly -- the per-fit reseed hook in neural_network.py does,
    once per candidate fit, so the warning below is printed at most once per
    process.
    """
    global _warned_about_hash_seed
    if not hash_seed_is_fixed() and not _warned_about_hash_seed:
        _warned_about_hash_seed = True
        print(
            f"[determinism] {HASH_SEED_ENV} is unset, so iteration over sets and "
            "dicts keyed by strings will differ between runs and results will "
            f"not be reproducible. Export {HASH_SEED_ENV}=0 before starting "
            "python (the Nextflow modules do).",
            file=sys.stderr,
        )

    # Read when cuBLAS initialises, which has usually not happened yet at import
    # time; the Nextflow modules also export it, for the case where it has.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        # general.yml has no torch -- the graph models and feature extraction
        # are seeded by the two calls above.
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch_deterministic:
        # warn_only: an op with no deterministic implementation should degrade
        # to a warning, not abort a multi-hour fit.
        torch.use_deterministic_algorithms(True, warn_only=True)
