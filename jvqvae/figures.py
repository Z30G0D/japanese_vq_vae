"""Figures for the blog post.

Agg is selected before pyplot is imported so these render on a headless box and
never try to open a window during a test run.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jvqvae.data.dataset import CharacterData  # noqa: E402
from jvqvae.models.prior import Prior  # noqa: E402
from jvqvae.models.vqvae import VQVAE  # noqa: E402
from jvqvae.sample import sample_images  # noqa: E402
from jvqvae.train.seed import set_seed  # noqa: E402


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


def _show(axis, image: np.ndarray) -> None:
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_xticks([])
    axis.set_yticks([])


@torch.no_grad()
def save_sample_grid(
    prior: Prior,
    vqvae: VQVAE,
    path: str | Path,
    *,
    class_ids: Sequence[int],
    samples_per_class: int = 4,
    device: torch.device,
    temperature: float | None = None,
    top_p: float | None = None,
    guidance_scale: float | None = None,
    seed: int = 0,
) -> Path:
    """Write one row of decoded samples per class.

    Titles are class indices, not glyphs. Pixel values are clipped to ``[0, 1]``.
    ``set_seed(seed)`` runs before the draw. ``sample_images`` restores each
    module's train/eval flag.
    """
    path = Path(path)
    if len(class_ids) < 1:
        raise ValueError("cannot draw a sample grid from an empty class list")

    path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    repeated = [
        int(class_id) for class_id in class_ids for _ in range(samples_per_class)
    ]
    images = sample_images(
        prior,
        vqvae,
        torch.tensor(repeated, dtype=torch.long),
        device=device,
        temperature=temperature,
        top_p=top_p,
        guidance_scale=guidance_scale,
    )
    pixels = np.clip(images.squeeze(1).detach().cpu().numpy(), 0.0, 1.0)

    rows = len(class_ids)
    cols = samples_per_class
    figure, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.2, rows * 1.4), squeeze=False
    )
    for row, class_id in enumerate(class_ids):
        for col in range(cols):
            _show(axes[row][col], pixels[row * cols + col])
        axes[row][0].set_title(str(int(class_id)), fontsize=9)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


@torch.no_grad()
def save_guidance_sweep(
    prior: Prior,
    vqvae: VQVAE,
    path: str | Path,
    *,
    class_id: int,
    scales: Sequence[float] = (0.0, 1.0, 1.5, 2.0, 3.0),
    num_samples: int = 4,
    device: torch.device,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int = 0,
) -> Path:
    """Write a guidance-strength sweep for one character.

    Rows are samples and columns are ``scales``. Each scale is drawn on its own;
    the seed is set once, so later columns continue the same generator.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    columns = []
    for scale in scales:
        class_ids = torch.full((num_samples,), int(class_id), dtype=torch.long)
        images = sample_images(
            prior,
            vqvae,
            class_ids,
            device=device,
            temperature=temperature,
            top_p=top_p,
            guidance_scale=float(scale),
        )
        columns.append(np.clip(images.squeeze(1).detach().cpu().numpy(), 0.0, 1.0))

    figure, axes = plt.subplots(
        num_samples,
        len(scales),
        figsize=(len(scales) * 1.2, num_samples * 1.4),
        squeeze=False,
    )
    for col, scale in enumerate(scales):
        axes[0][col].set_title(str(scale), fontsize=9)
        for row in range(num_samples):
            _show(axes[row][col], columns[col][row])

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
