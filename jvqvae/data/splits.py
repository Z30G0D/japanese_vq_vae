"""Dataset splitting.

ETL4 splits by writer, never by image: a random image split puts the same
person's handwriting in both train and test, which inflates held-out numbers.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_by_writer(
    writer_ids: np.ndarray | None,
    *,
    seed: int,
    n_val_writers: int,
    n_test_writers: int,
) -> Split:
    """Partition sample indices so that no writer spans two splits."""
    if writer_ids is None:
        raise ValueError(
            "cannot split by writer: this source has no writer identities. "
            "Use carve_validation for a random split instead."
        )

    unique_writers = np.unique(writer_ids)
    held_out = n_val_writers + n_test_writers
    if len(unique_writers) <= held_out:
        raise ValueError(
            f"need more than {held_out} writers to hold out "
            f"{n_val_writers} val + {n_test_writers} test, got {len(unique_writers)}"
        )

    shuffled = np.random.default_rng(seed).permutation(unique_writers)
    test_writers = shuffled[:n_test_writers]
    val_writers = shuffled[n_test_writers:held_out]
    train_writers = shuffled[held_out:]

    def indices_for(writers: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(writer_ids, writers)).astype(np.int64)

    return Split(
        train=indices_for(train_writers),
        val=indices_for(val_writers),
        test=indices_for(test_writers),
    )


def carve_validation(
    n: int, *, seed: int, n_val: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``n`` indices into (train, val) at random.

    Used for sources such as K49 that ship a train/test split but no
    validation set and no writer identities.
    """
    if n_val >= n:
        raise ValueError(f"cannot carve {n_val} validation samples out of {n}")
    shuffled = np.random.default_rng(seed).permutation(n).astype(np.int64)
    return shuffled[n_val:], shuffled[:n_val]
