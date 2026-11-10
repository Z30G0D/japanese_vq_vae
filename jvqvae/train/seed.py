"""One place to make a run reproducible."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch together.

    Data loading and splitting use explicit generators, so this covers what is
    left: weight initialization and the dead-code restart picks.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
