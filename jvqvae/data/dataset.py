"""Source-agnostic container for a labeled character image set.

Every loader produces one of these, and everything downstream consumes one, so
adding a dataset never touches the models.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class CharacterData:
    images: np.ndarray  # float32 (N, H, W) in [0, 1]
    class_ids: np.ndarray  # int64 (N,)
    writer_ids: np.ndarray | None  # int64 (N,) or None when the source has no writers
    classes: list[str]  # class index -> unicode character

    def __post_init__(self) -> None:
        if self.images.ndim != 3:
            raise ValueError(f"images must be (N, H, W), got {self.images.shape}")
        if self.images.dtype != np.float32:
            raise ValueError(f"images must be float32, got {self.images.dtype}")
        if len(self.class_ids) != len(self.images):
            raise ValueError(
                f"length mismatch: {len(self.images)} images, "
                f"{len(self.class_ids)} class ids"
            )
        if self.writer_ids is not None and len(self.writer_ids) != len(self.images):
            raise ValueError(
                f"length mismatch: {len(self.images)} images, "
                f"{len(self.writer_ids)} writer ids"
            )
        if len(self.images) and not (
            self.images.min() >= 0.0 and self.images.max() <= 1.0
        ):
            raise ValueError(
                f"images must lie in [0, 1], got "
                f"[{self.images.min():.3f}, {self.images.max():.3f}]"
            )
        if len(self.class_ids) and int(self.class_ids.max()) >= len(self.classes):
            raise ValueError(
                f"class id {int(self.class_ids.max())} exceeds class table of "
                f"size {len(self.classes)}"
            )

    def __len__(self) -> int:
        return len(self.images)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.images.shape[1], self.images.shape[2]

    def subset(self, indices: np.ndarray) -> "CharacterData":
        return replace(
            self,
            images=self.images[indices],
            class_ids=self.class_ids[indices],
            writer_ids=None if self.writer_ids is None else self.writer_ids[indices],
        )
