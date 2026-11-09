"""Device selection, so the same code runs on a Mac, a CUDA box, or CPU."""

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Return the device to train on.

    ``preference`` may be an explicit device string ("cpu", "mps", "cuda") or
    "auto" to pick the best available.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
