"""Figures for the blog post.

Agg is selected before pyplot is imported so these render on a headless box and
never try to open a window during a test run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jvqvae.data.dataset import CharacterData  # noqa: E402
from jvqvae.models.vqvae import VQVAE  # noqa: E402


@torch.no_grad()
def save_reconstruction_grid(
    model: VQVAE,
    data: CharacterData,
    path: str | Path,
    *,
    device: torch.device,
    num_images: int = 8,
    seed: int = 0,
) -> Path:
    """Write originals over their reconstructions, for a visual sanity check."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval().to(device)

    count = min(num_images, len(data))
    if count < 1:
        raise ValueError("cannot draw a reconstruction grid from an empty dataset")
    picks = np.random.default_rng(seed).choice(len(data), count, replace=False)
    picks.sort()

    originals = data.images[picks]
    tensor = torch.from_numpy(originals).unsqueeze(1).to(device)
    reconstructions = np.clip(
        model(tensor).reconstruction.squeeze(1).cpu().numpy(), 0.0, 1.0
    )

    if was_training:
        model.train()

    figure, axes = plt.subplots(2, count, figsize=(count * 1.2, 2.8), squeeze=False)
    for column in range(count):
        for row, image in enumerate((originals[column], reconstructions[column])):
            axis = axes[row][column]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
        # Class index, not the unicode glyph: a missing CJK font would warn
        # on every test run, and the index is what a reader can look up.
        axes[0][column].set_title(str(int(data.class_ids[picks[column]])), fontsize=9)

    axes[0][0].set_ylabel("original", fontsize=8)
    axes[1][0].set_ylabel("decoded", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path
