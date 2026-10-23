"""Shared run-reproducibility helper for every training script in this folder.

WHY THIS EXISTS
---------------
Until this module was added, none of the three training scripts (train.py,
train_baseline.py, train_paysim.py) seeded any random number generator. That
meant two runs of the *same* configuration produced different weights, and
therefore different reported metrics -- the numbers were measured and
inspectable, but not bit-reproducible. For a dissertation whose central
methodological commitment is that every reported number is a reproducible
artefact of a specific run, that gap mattered.

`set_seed()` fixes every RNG the training path actually draws from:

  * Python's `random`      -- used indirectly by some library code
  * NumPy's global RNG     -- used by the data loaders and sampling helpers
  * PyTorch CPU RNG        -- weight initialisation, dropout
  * PyTorch CUDA RNG       -- the same, on GPU
  * cuDNN                  -- forced into deterministic, non-autotuning mode

Opacus draws its calibrated DP-SGD noise, and its Poisson subsampling indices,
from PyTorch's global RNG, so seeding that covers the privacy mechanism too.

WHAT THIS DOES *NOT* GUARANTEE
------------------------------
This gives reproducibility for a fixed seed on the *same hardware with the same
library versions*. It deliberately does NOT call
`torch.use_deterministic_algorithms(True)`: cuDNN's GRU backward kernel has no
deterministic implementation, so forcing that flag would make these scripts
raise at the first backward pass rather than train. Cross-GPU-model or
cross-CUDA-version bit-identity is therefore not claimed, and should not be
claimed in the write-up -- pin the environment (see requirements-hpc.txt) if
exact reproduction elsewhere is required.
"""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """Seed every RNG the training path draws from. Call once, early, before
    any model construction or data shuffling. Returns the seed so callers can
    log it straight into their metrics JSON."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True picks reproducible cuDNN kernels; benchmark=False
        # stops cuDNN autotuning, which otherwise selects different algorithms
        # run-to-run depending on transient GPU load.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[reproducibility] All RNGs seeded with seed={seed}")
    return seed
