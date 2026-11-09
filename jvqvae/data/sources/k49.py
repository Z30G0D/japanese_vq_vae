"""Kuzushiji-49: the development dataset.

Freely downloadable (CC BY-SA 4.0), which makes it the source we build against
while ETL4 access is pending. It is cursive historical script, so it is a
development convenience and not a stand-in for ETL4 results.

Attribution: "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji
Dataset" (created by NIJL and others), doi:10.20676/00000341.
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np

from jvqvae.data.dataset import CharacterData
from jvqvae.data.transforms import pad_to_square

_BASE = "https://codh.rois.ac.jp/kmnist/dataset/k49"

K49_FILES = {
    "k49-train-imgs.npz": f"{_BASE}/k49-train-imgs.npz",
    "k49-train-labels.npz": f"{_BASE}/k49-train-labels.npz",
    "k49-test-imgs.npz": f"{_BASE}/k49-test-imgs.npz",
    "k49-test-labels.npz": f"{_BASE}/k49-test-labels.npz",
    "k49_classmap.csv": f"{_BASE}/k49_classmap.csv",
}


def download_k49(dest_dir: str | Path) -> Path:
    """Download any K49 files not already present. Returns the directory.

    Each file lands in a `.part` sibling and is renamed only once the
    transfer completes, so an interrupted download cannot leave a truncated
    file that later runs mistake for a finished one.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in K49_FILES.items():
        target = dest / name
        if target.exists():
            print(f"already have {name}")
            continue

        print(f"downloading {name} from {url}")
        partial = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return dest


def _load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["index"]))

    indices = [int(row["index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(
            f"{path}: class indices must be contiguous from 0, got {indices}"
        )
    return [row["char"] for row in rows]


def _cap_per_class(
    class_ids: np.ndarray, max_per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for class_id in np.unique(class_ids):
        candidates = np.flatnonzero(class_ids == class_id)
        if len(candidates) > max_per_class:
            candidates = rng.choice(candidates, max_per_class, replace=False)
        keep.append(np.sort(candidates))
    return np.sort(np.concatenate(keep)).astype(np.int64)


def load_k49(
    dest_dir: str | Path,
    *,
    split: str,
    image_size: int = 32,
    max_per_class: int | None = None,
    seed: int = 0,
) -> CharacterData:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    directory = Path(dest_dir)
    needed = [f"k49-{split}-imgs.npz", f"k49-{split}-labels.npz", "k49_classmap.csv"]
    missing = [name for name in needed if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing K49 file(s) in {directory}: {', '.join(missing)}. "
            f"Run `uv run jvqvae fetch-k49 --dest {directory}` first."
        )

    raw_images = np.load(directory / f"k49-{split}-imgs.npz")["arr_0"]
    class_ids = np.load(directory / f"k49-{split}-labels.npz")["arr_0"].astype(np.int64)
    classes = _load_classes(directory / "k49_classmap.csv")

    if max_per_class is not None:
        keep = _cap_per_class(class_ids, max_per_class, seed)
        raw_images, class_ids = raw_images[keep], class_ids[keep]

    images = raw_images.astype(np.float32) / 255.0
    images = pad_to_square(images)
    if images.shape[1] != image_size:
        pad = image_size - images.shape[1]
        if pad < 0:
            raise ValueError(
                f"image_size {image_size} is smaller than K49's {images.shape[1]}"
            )
        before = pad // 2
        images = np.pad(
            images,
            ((0, 0), (before, pad - before), (before, pad - before)),
        )

    return CharacterData(
        images=images,
        class_ids=class_ids,
        writer_ids=None,
        classes=classes,
    )
