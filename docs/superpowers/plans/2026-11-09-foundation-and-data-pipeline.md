# Foundation and Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project skeleton and a tested, source-agnostic data pipeline that
delivers normalized, labeled, writer-aware character images to the models in plans 2 and 3.

**Architecture:** A single in-memory container (`CharacterData`) holds images, class IDs,
optional writer IDs and a class-index-to-character table. Two loaders populate it —
Kuzushiji-49 (downloadable today) and ETL4 (parsed from a user-supplied zip). Splitting,
transforms and the PyTorch adapter are written against the container, never against a
specific source, so swapping sources is a config change.

**Tech Stack:** Python 3.12, uv, numpy, Pillow, PyYAML, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-japanese-vq-vae-design.md`

## Global Constraints

- Python 3.12 (`uv` 0.11.15 with CPython 3.12.13 is already installed locally).
- Dependencies limited to: torch, numpy, pillow, pyyaml, matplotlib, pytest. No
  experiment-tracking service.
- The repository must never contain ETL4 data and must never download it. AIST's Terms
  of Use clause 5 prohibits redistribution and direct URL links.
- Required citation in the README: Electrotechnical Laboratory, Japanese Technical
  Committee for Optical Character Recognition, ETL Character Database, 1973-1984.
- Required K49 attribution: "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji
  Dataset" (created by NIJL and others), doi:10.20676/00000341.
- ETL4 images resize to 64x64; K49 images pad to 32x32. Both produce an 8x8 code grid
  later (three downsampling stages for ETL4, two for K49).
- All randomness goes through an explicit seed. No global `np.random` calls.
- The full test suite must pass with no ETL4 file present and no network access.
- Package name is `jvqvae`. Run everything through `uv run`.

### Known deviation from the spec

Spec section 4 says preprocessing should "apply the per-record gain field". The C-type
record layout we parse (`>9xB206x2736s`) does not expose a documented gain offset, and
guessing one is not verifiable while AIST is offline. This plan normalizes by the 4-bit
maximum (divide by 15.0) instead, which is deterministic and contrast-equivalent for
training. Revisit once the real file is available.

Spec section 6 lists `jvqvae/data/jis.py` and `etl_format.yml` as vendored lookup
tables. Neither is needed: the C-type layout is five integer constants, and the
JIS X 0201 halfwidth-katakana to hiragana mapping is derivable with stdlib
`unicodedata.normalize("NFKC", ...)`. Record parsing lives in
`jvqvae/data/sources/etl4_records.py` instead, split out from `etl4.py` so the byte-level
format is testable without any archive. Vendoring a JIS X 0208 table becomes necessary
only for the kanji sets, which are out of scope.

### Standing risk

AIST (`etlcdb.db.aist.go.jp`) has returned HTTP 503 continuously since at least
2026-11-04. Every ETL4 task here is therefore built and tested against synthetic records
encoded from the documented layout. Task 7 defines the validation that runs against the
real file when it arrives.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `.gitignore` | Keep data and caches out of git |
| `jvqvae/device.py` | Select MPS / CUDA / CPU |
| `jvqvae/config.py` | YAML-backed dataclass configs |
| `jvqvae/data/dataset.py` | `CharacterData` container and its invariants |
| `jvqvae/data/transforms.py` | Pad-to-square and resize on float image stacks |
| `jvqvae/data/splits.py` | Writer-aware and index-based splitting |
| `jvqvae/data/sources/k49.py` | Download and load Kuzushiji-49 |
| `jvqvae/data/sources/etl4_records.py` | C-type record parsing and character decoding |
| `jvqvae/data/sources/etl4.py` | Zip verification, ingestion, writer inference, caching |
| `jvqvae/data/torch_dataset.py` | PyTorch `Dataset` adapter |
| `jvqvae/cli.py` | `fetch-k49` and `ingest-etl4` subcommands |
| `configs/k49_dev.yaml`, `configs/etl4.yaml` | Concrete run configs |
| `tests/` | One test module per source module |

---

### Task 1: Project scaffolding and device selection

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `jvqvae/__init__.py`
- Create: `jvqvae/device.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `jvqvae.device.select_device(preference: str = "auto") -> torch.device`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
*.egg-info/

# Never commit datasets. ETL4 redistribution is prohibited by AIST's terms.
data/
results/
checkpoints/
.ipynb_checkpoints/
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "jvqvae"
version = "0.1.0"
description = "VQ-VAE with a class-conditional transformer prior for handwritten Japanese characters"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.4",
    "numpy>=1.26",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["jvqvae"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Create the package marker**

Create `jvqvae/__init__.py` containing exactly:

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create the environment and confirm torch imports**

```bash
uv sync --python 3.12
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expected: a torch version such as `2.x.y` followed by `True` on Apple Silicon. If it
prints `False`, that is not a failure — `select_device` falls back to CPU and everything
in this plan still works. Record which you saw; plan 2 needs to know.

- [ ] **Step 5: Write the failing test**

Create `tests/test_device.py`:

```python
import torch

from jvqvae.device import select_device


def test_explicit_preference_is_respected():
    assert select_device("cpu").type == "cpu"


def test_auto_returns_a_usable_device():
    device = select_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"mps", "cuda", "cpu"}
    torch.zeros(2, 2, device=device)
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.device'`

- [ ] **Step 7: Write the implementation**

Create `jvqvae/device.py`:

```python
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
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_device.py -v`
Expected: PASS, 2 passed

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore jvqvae/__init__.py jvqvae/device.py tests/test_device.py uv.lock
git commit -m "feat: scaffold jvqvae package with device selection"
```

---

### Task 2: Configuration

**Files:**
- Create: `jvqvae/config.py`
- Create: `configs/k49_dev.yaml`
- Create: `configs/etl4.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SplitConfig`, `DataConfig`, and
  `jvqvae.config.load_data_config(path: str | Path) -> DataConfig`.
  `DataConfig` fields: `source: str`, `image_size: int`, `cache_dir: str`,
  `etl4_zip: str | None`, `k49_dir: str`, `k49_max_per_class: int | None`,
  `split: SplitConfig`. `SplitConfig` fields: `seed: int`, `val_writers: int`,
  `test_writers: int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import dataclasses

import pytest

from jvqvae.config import DataConfig, SplitConfig, load_data_config


def test_defaults_apply_when_yaml_is_minimal(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source: k49\nimage_size: 32\n")

    config = load_data_config(path)

    assert config.source == "k49"
    assert config.image_size == 32
    assert config.split == SplitConfig()
    assert config.etl4_zip is None


def test_nested_split_is_parsed(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "source: etl4\n"
        "image_size: 64\n"
        "etl4_zip: data/raw/ETL4.zip\n"
        "split:\n"
        "  seed: 7\n"
        "  val_writers: 10\n"
    )

    config = load_data_config(path)

    assert config.split.seed == 7
    assert config.split.val_writers == 10
    assert config.split.test_writers == 12  # default preserved
    assert config.etl4_zip == "data/raw/ETL4.zip"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source: k49\nimage_size: 32\nlearnig_rate: 0.1\n")

    with pytest.raises(ValueError, match="learnig_rate"):
        load_data_config(path)


def test_config_is_immutable():
    config = DataConfig(source="k49", image_size=32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.source = "etl4"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.config'`

- [ ] **Step 3: Write the implementation**

Create `jvqvae/config.py`:

```python
"""YAML-backed configuration.

Typos in a config file are a common and expensive way to waste a training run,
so unknown keys are rejected rather than ignored.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    val_writers: int = 12
    test_writers: int = 12


@dataclass(frozen=True)
class DataConfig:
    source: str
    image_size: int
    cache_dir: str = "data/cache"
    etl4_zip: str | None = None
    k49_dir: str = "data/k49"
    k49_max_per_class: int | None = None
    split: SplitConfig = field(default_factory=SplitConfig)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}"
        )
    return cls(**raw)


def load_data_config(path: str | Path) -> DataConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["split"] = _build(SplitConfig, raw.get("split") or {})
    return _build(DataConfig, raw)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Write the config files**

Create `configs/k49_dev.yaml`:

```yaml
source: k49
image_size: 32
k49_dir: data/k49
k49_max_per_class: 500
split:
  seed: 0
```

Create `configs/etl4.yaml`:

```yaml
source: etl4
image_size: 64
etl4_zip: data/raw/ETL4.zip
split:
  seed: 0
  val_writers: 12
  test_writers: 12
```

- [ ] **Step 6: Commit**

```bash
git add jvqvae/config.py configs tests/test_config.py
git commit -m "feat: add YAML-backed data configuration"
```

---

### Task 3: CharacterData container and transforms

**Files:**
- Create: `jvqvae/data/__init__.py`
- Create: `jvqvae/data/dataset.py`
- Create: `jvqvae/data/transforms.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `jvqvae.data.dataset.CharacterData(images, class_ids, writer_ids, classes)` — a
    frozen dataclass. `images` is `np.ndarray` float32 `(N, H, W)` in `[0, 1]`;
    `class_ids` is int64 `(N,)`; `writer_ids` is int64 `(N,)` or `None`; `classes` is
    `list[str]`. Exposes `__len__`, `num_classes: int`, `image_shape: tuple[int, int]`,
    and `subset(indices: np.ndarray) -> CharacterData`.
  - `jvqvae.data.transforms.pad_to_square(images: np.ndarray) -> np.ndarray`
  - `jvqvae.data.transforms.resize(images: np.ndarray, size: int) -> np.ndarray`

- [ ] **Step 1: Write the failing test for the container**

Create `tests/test_dataset.py`:

```python
import numpy as np
import pytest

from jvqvae.data.dataset import CharacterData


def make(n=6, n_classes=3):
    return CharacterData(
        images=np.zeros((n, 4, 4), dtype=np.float32),
        class_ids=np.arange(n, dtype=np.int64) % n_classes,
        writer_ids=np.arange(n, dtype=np.int64) // n_classes,
        classes=[chr(0x3042 + i) for i in range(n_classes)],
    )


def test_basic_properties():
    data = make()
    assert len(data) == 6
    assert data.num_classes == 3
    assert data.image_shape == (4, 4)


def test_subset_keeps_alignment():
    data = make()
    subset = data.subset(np.array([0, 4]))
    assert len(subset) == 2
    assert subset.class_ids.tolist() == [0, 1]
    assert subset.writer_ids.tolist() == [0, 1]
    assert subset.classes == data.classes


def test_writer_ids_may_be_absent():
    data = CharacterData(
        images=np.zeros((2, 4, 4), dtype=np.float32),
        class_ids=np.zeros(2, dtype=np.int64),
        writer_ids=None,
        classes=["あ"],
    )
    assert data.writer_ids is None
    assert data.subset(np.array([1])).writer_ids is None


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length"):
        CharacterData(
            images=np.zeros((3, 4, 4), dtype=np.float32),
            class_ids=np.zeros(2, dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )


def test_out_of_range_pixels_are_rejected():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        CharacterData(
            images=np.full((2, 4, 4), 255.0, dtype=np.float32),
            class_ids=np.zeros(2, dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )


def test_class_id_beyond_class_table_is_rejected():
    with pytest.raises(ValueError, match="class id"):
        CharacterData(
            images=np.zeros((2, 4, 4), dtype=np.float32),
            class_ids=np.array([0, 9], dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data'`

- [ ] **Step 3: Implement the container**

Create `jvqvae/data/__init__.py` as an empty file. Then create
`jvqvae/data/dataset.py`:

```python
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
```

- [ ] **Step 4: Run the container test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Write the failing test for transforms**

Create `tests/test_transforms.py`:

```python
import numpy as np

from jvqvae.data.transforms import pad_to_square, resize


def test_pad_to_square_centers_content():
    images = np.ones((2, 2, 4), dtype=np.float32)
    padded = pad_to_square(images)
    assert padded.shape == (2, 4, 4)
    # One row of zeros added above and below the original two rows.
    assert padded[0, 0].sum() == 0.0
    assert padded[0, 3].sum() == 0.0
    assert padded[0, 1].sum() == 4.0


def test_pad_to_square_is_a_noop_when_already_square():
    images = np.ones((1, 3, 3), dtype=np.float32)
    assert pad_to_square(images).shape == (1, 3, 3)


def test_resize_changes_shape_and_preserves_range():
    rng = np.random.default_rng(0)
    images = rng.random((3, 16, 16)).astype(np.float32)
    out = resize(images, 8)
    assert out.shape == (3, 8, 8)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_preserves_a_constant_image():
    images = np.full((1, 16, 16), 0.5, dtype=np.float32)
    out = resize(images, 8)
    assert np.allclose(out, 0.5, atol=1e-6)
```

- [ ] **Step 6: Run the transforms test to verify it fails**

Run: `uv run pytest tests/test_transforms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data.transforms'`

- [ ] **Step 7: Implement transforms**

Create `jvqvae/data/transforms.py`:

```python
"""Geometry transforms on stacks of float images in [0, 1].

ETL4 records are 72x76, so they are padded to square rather than cropped; the
2018 project cropped the top 72 rows, which can clip strokes.
"""

import numpy as np
from PIL import Image


def pad_to_square(images: np.ndarray) -> np.ndarray:
    """Zero-pad each image to a square, centering the original content."""
    _, height, width = images.shape
    side = max(height, width)
    if height == width:
        return images

    top = (side - height) // 2
    left = (side - width) // 2
    out = np.zeros((len(images), side, side), dtype=images.dtype)
    out[:, top : top + height, left : left + width] = images
    return out


def resize(images: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resize each image to ``size`` x ``size``.

    Uses PIL's 32-bit float mode so stroke-pressure gradients survive the
    resize without being quantized to 8 bits.
    """
    out = np.empty((len(images), size, size), dtype=np.float32)
    for i, image in enumerate(images):
        pil = Image.fromarray(image.astype(np.float32), mode="F")
        out[i] = np.asarray(pil.resize((size, size), Image.BILINEAR), dtype=np.float32)
    return np.clip(out, 0.0, 1.0)
```

- [ ] **Step 8: Run the transforms test to verify it passes**

Run: `uv run pytest tests/test_transforms.py -v`
Expected: PASS, 4 passed

- [ ] **Step 9: Commit**

```bash
git add jvqvae/data tests/test_dataset.py tests/test_transforms.py
git commit -m "feat: add CharacterData container and image transforms"
```

---

### Task 4: Writer-aware splitting

**Files:**
- Create: `jvqvae/data/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `jvqvae.data.splits.Split` — frozen dataclass with `train`, `val`, `test`, each an
    int64 `np.ndarray` of indices.
  - `split_by_writer(writer_ids: np.ndarray, *, seed: int, n_val_writers: int,
    n_test_writers: int) -> Split`
  - `carve_validation(n: int, *, seed: int, n_val: int) -> tuple[np.ndarray, np.ndarray]`
    returning `(train_indices, val_indices)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_splits.py`:

```python
import numpy as np
import pytest

from jvqvae.data.splits import carve_validation, split_by_writer


def writer_ids(n_writers=10, n_classes=5):
    return np.repeat(np.arange(n_writers, dtype=np.int64), n_classes)


def test_splits_are_disjoint_and_cover_everything():
    ids = writer_ids()
    split = split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)

    all_indices = np.concatenate([split.train, split.val, split.test])
    assert sorted(all_indices.tolist()) == list(range(len(ids)))


def test_no_writer_appears_in_two_splits():
    ids = writer_ids()
    split = split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)

    train_w = set(ids[split.train].tolist())
    val_w = set(ids[split.val].tolist())
    test_w = set(ids[split.test].tolist())
    assert train_w & val_w == set()
    assert train_w & test_w == set()
    assert val_w & test_w == set()
    assert len(val_w) == 2 and len(test_w) == 2 and len(train_w) == 6


def test_split_is_deterministic_for_a_seed():
    ids = writer_ids()
    a = split_by_writer(ids, seed=3, n_val_writers=2, n_test_writers=2)
    b = split_by_writer(ids, seed=3, n_val_writers=2, n_test_writers=2)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)


def test_different_seeds_give_different_splits():
    ids = writer_ids()
    a = split_by_writer(ids, seed=1, n_val_writers=2, n_test_writers=2)
    b = split_by_writer(ids, seed=2, n_val_writers=2, n_test_writers=2)
    assert not np.array_equal(a.test, b.test)


def test_too_few_writers_is_rejected():
    ids = writer_ids(n_writers=3)
    with pytest.raises(ValueError, match="writers"):
        split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)


def test_every_split_contains_every_class():
    n_writers, n_classes = 10, 5
    ids = writer_ids(n_writers, n_classes)
    class_ids = np.tile(np.arange(n_classes, dtype=np.int64), n_writers)
    split = split_by_writer(ids, seed=0, n_val_writers=2, n_test_writers=2)

    for indices in (split.train, split.val, split.test):
        assert set(class_ids[indices].tolist()) == set(range(n_classes))


def test_carve_validation_is_disjoint_and_deterministic():
    train, val = carve_validation(100, seed=0, n_val=20)
    assert len(val) == 20 and len(train) == 80
    assert set(train.tolist()) & set(val.tolist()) == set()

    train2, val2 = carve_validation(100, seed=0, n_val=20)
    assert np.array_equal(val, val2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data.splits'`

- [ ] **Step 3: Write the implementation**

Create `jvqvae/data/splits.py`:

```python
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
    writer_ids: np.ndarray,
    *,
    seed: int,
    n_val_writers: int,
    n_test_writers: int,
) -> Split:
    """Partition sample indices so that no writer spans two splits."""
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_splits.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add jvqvae/data/splits.py tests/test_splits.py
git commit -m "feat: add writer-aware dataset splitting"
```

---

### Task 5: Kuzushiji-49 source

**Files:**
- Create: `jvqvae/data/sources/__init__.py`
- Create: `jvqvae/data/sources/k49.py`
- Test: `tests/test_k49.py`

**Interfaces:**
- Consumes: `CharacterData` (Task 3), `pad_to_square` (Task 3).
- Produces:
  - `jvqvae.data.sources.k49.K49_FILES: dict[str, str]` mapping local filename to URL.
  - `download_k49(dest_dir: str | Path) -> Path` — downloads any missing file, returns
    the directory.
  - `load_k49(dest_dir: str | Path, *, split: str, image_size: int = 32,
    max_per_class: int | None = None, seed: int = 0) -> CharacterData` where `split` is
    `"train"` or `"test"`.

K49 is 28x28 and has no writer identities, so `writer_ids` is `None`. Images are padded
to 32x32 rather than resized, which keeps the original pixel grid intact.

- [ ] **Step 1: Write the failing test**

Create `tests/test_k49.py`. These tests write tiny fake `.npz` files, so the suite never
touches the network.

```python
import numpy as np
import pytest

from jvqvae.data.sources.k49 import K49_FILES, load_k49


@pytest.fixture
def fake_k49(tmp_path):
    rng = np.random.default_rng(0)
    # 3 classes x 4 samples, deliberately imbalanced to exercise max_per_class.
    labels = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
    images = rng.integers(0, 256, size=(len(labels), 28, 28), dtype=np.uint8)
    np.savez_compressed(tmp_path / "k49-train-imgs.npz", images)
    np.savez_compressed(tmp_path / "k49-train-labels.npz", labels)
    (tmp_path / "k49_classmap.csv").write_text(
        "index,codepoint,char\n0,U+3042,あ\n1,U+3044,い\n2,U+3046,う\n",
        encoding="utf-8",
    )
    return tmp_path


def test_urls_cover_the_files_we_need():
    assert "k49-train-imgs.npz" in K49_FILES
    assert "k49_classmap.csv" in K49_FILES
    assert all(url.startswith("https://") for url in K49_FILES.values())


def test_load_produces_padded_normalized_data(fake_k49):
    data = load_k49(fake_k49, split="train")

    assert len(data) == 7
    assert data.image_shape == (32, 32)
    assert data.images.dtype == np.float32
    assert data.images.min() >= 0.0 and data.images.max() <= 1.0
    assert data.writer_ids is None


def test_class_table_is_read_from_the_classmap(fake_k49):
    data = load_k49(fake_k49, split="train")
    assert data.classes == ["あ", "い", "う"]
    assert data.num_classes == 3


def test_padding_puts_original_pixels_in_the_middle(fake_k49):
    data = load_k49(fake_k49, split="train")
    # 28 -> 32 means a 2px border of zeros on every side.
    assert data.images[:, :2, :].sum() == 0.0
    assert data.images[:, -2:, :].sum() == 0.0
    assert data.images[:, :, :2].sum() == 0.0


def test_max_per_class_caps_each_class(fake_k49):
    data = load_k49(fake_k49, split="train", max_per_class=2)
    counts = np.bincount(data.class_ids, minlength=3)
    assert counts.tolist() == [2, 2, 1]


def test_subsampling_is_deterministic(fake_k49):
    a = load_k49(fake_k49, split="train", max_per_class=2, seed=5)
    b = load_k49(fake_k49, split="train", max_per_class=2, seed=5)
    assert np.array_equal(a.images, b.images)


def test_missing_files_give_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-k49"):
        load_k49(tmp_path, split="train")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_k49.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data.sources'`

- [ ] **Step 3: Write the implementation**

Create `jvqvae/data/sources/__init__.py` as an empty file. Then create
`jvqvae/data/sources/k49.py`:

```python
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
    """Download any K49 files not already present. Returns the directory."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in K49_FILES.items():
        target = dest / name
        if target.exists():
            print(f"already have {name}")
            continue
        print(f"downloading {name} from {url}")
        urllib.request.urlretrieve(url, target)
    return dest


def _load_classes(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["index"]))
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_k49.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add jvqvae/data/sources tests/test_k49.py
git commit -m "feat: add Kuzushiji-49 download and loader"
```

---

### Task 6: ETL4 record parsing and character decoding

**Files:**
- Create: `jvqvae/data/sources/etl4_records.py`
- Test: `tests/test_etl4_records.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Constants `RECORD_SIZE = 2952`, `IMAGE_BYTES = 2736`, `WIDTH = 72`, `HEIGHT = 76`,
    `CHAR_CODE_OFFSET = 9`, `MAX_PIXEL_VALUE = 15`.
  - `parse_record(raw: bytes) -> tuple[int, np.ndarray]` returning the character-code
    byte and a uint8 `(76, 72)` image with values 0-15.
  - `encode_record(char_code: int, image: np.ndarray) -> bytes` — the inverse, used by
    tests and only by tests.
  - `decode_hiragana(char_code: int) -> str | None` — JIS X 0201 halfwidth katakana code
    to the matching hiragana, or `None` when the code is not a kana.

**Format background.** ETL4 uses the C-type layout: fixed 2,952-byte records holding a
72x76 image at 4 bits per pixel. 72 x 76 / 2 = 2,736 image bytes, and 9 + 1 + 206 + 2,736
= 2,952, so the layout is arithmetically self-consistent. Character codes are JIS X 0201,
which has no hiragana — it has halfwidth katakana — which is why the reference extractor
carries an `is_hiragana` flag for ETL4. Decoding therefore goes halfwidth katakana to
fullwidth katakana via NFKC, then subtracts 0x60 to reach hiragana. Example: 0xB1 maps to
U+FF71 (ｱ), normalizes to U+30A2 (ア), and becomes U+3042 (あ).

- [ ] **Step 1: Write the failing test**

Create `tests/test_etl4_records.py`:

```python
import numpy as np
import pytest

from jvqvae.data.sources.etl4_records import (
    HEIGHT,
    IMAGE_BYTES,
    RECORD_SIZE,
    WIDTH,
    decode_hiragana,
    encode_record,
    parse_record,
)


def test_layout_arithmetic_is_consistent():
    assert WIDTH * HEIGHT == IMAGE_BYTES * 2
    assert 9 + 1 + 206 + IMAGE_BYTES == RECORD_SIZE


def test_round_trip_recovers_code_and_pixels():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 16, size=(HEIGHT, WIDTH), dtype=np.uint8)

    code, decoded = parse_record(encode_record(0xB1, image))

    assert code == 0xB1
    assert decoded.shape == (HEIGHT, WIDTH)
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, image)


def test_nibble_order_is_high_then_low():
    image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image[0, 0] = 0xA
    image[0, 1] = 0x3

    _, decoded = parse_record(encode_record(0xB1, image))

    assert decoded[0, 0] == 0xA
    assert decoded[0, 1] == 0x3


def test_wrong_record_length_is_rejected():
    with pytest.raises(ValueError, match="2952"):
        parse_record(b"\x00" * 100)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0xB1, "あ"), (0xB2, "い"), (0xB3, "う"), (0xB4, "え"), (0xB5, "お")],
)
def test_known_codes_decode_to_hiragana(code, expected):
    assert decode_hiragana(code) == expected


def test_non_kana_codes_return_none():
    assert decode_hiragana(0x41) is None


def test_decoded_hiragana_are_distinct_across_the_kana_range():
    decoded = [decode_hiragana(code) for code in range(0xA1, 0xE0)]
    hiragana = [char for char in decoded if char is not None]
    assert len(hiragana) >= 45
    assert len(set(hiragana)) == len(hiragana)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_etl4_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data.sources.etl4_records'`

- [ ] **Step 3: Write the implementation**

Create `jvqvae/data/sources/etl4_records.py`:

```python
"""ETL4 C-type record parsing.

Record layout (struct ">9xB206x2736s"): 9 padding bytes, one character-code
byte, 206 padding bytes, then a 72x76 image at 4 bits per pixel.

The layout and the JIS X 0201 character set are documented by AIST and were
cross-checked against the MIT-licensed choo/etlcdb-image-extractor
(https://github.com/choo/etlcdb-image-extractor).
"""

import unicodedata

import numpy as np

RECORD_SIZE = 2952
IMAGE_BYTES = 2736
WIDTH = 72
HEIGHT = 76
CHAR_CODE_OFFSET = 9
MAX_PIXEL_VALUE = 15

_IMAGE_OFFSET = RECORD_SIZE - IMAGE_BYTES

# JIS X 0201 halfwidth katakana occupy 0xA1-0xDF, mapping to U+FF61-U+FF9F.
_HALFWIDTH_KATAKANA_START = 0xA1
_HALFWIDTH_KATAKANA_END = 0xDF
_HALFWIDTH_UNICODE_START = 0xFF61
# Distance from a katakana codepoint to its hiragana counterpart.
_KATAKANA_TO_HIRAGANA = 0x60


def parse_record(raw: bytes) -> tuple[int, np.ndarray]:
    """Return ``(character_code, image)`` for one 2,952-byte record."""
    if len(raw) != RECORD_SIZE:
        raise ValueError(f"record must be {RECORD_SIZE} bytes, got {len(raw)}")

    char_code = raw[CHAR_CODE_OFFSET]
    packed = np.frombuffer(raw[_IMAGE_OFFSET:], dtype=np.uint8)

    pixels = np.empty(packed.size * 2, dtype=np.uint8)
    pixels[0::2] = packed >> 4
    pixels[1::2] = packed & 0x0F
    return char_code, pixels.reshape(HEIGHT, WIDTH)


def encode_record(char_code: int, image: np.ndarray) -> bytes:
    """Build a synthetic record. Used by tests while AIST is unreachable."""
    if image.shape != (HEIGHT, WIDTH):
        raise ValueError(f"image must be ({HEIGHT}, {WIDTH}), got {image.shape}")
    if image.max(initial=0) > MAX_PIXEL_VALUE:
        raise ValueError("pixel values must fit in 4 bits")

    flat = image.reshape(-1).astype(np.uint8)
    packed = ((flat[0::2] << 4) | flat[1::2]).astype(np.uint8)
    return b"\x00" * CHAR_CODE_OFFSET + bytes([char_code]) + b"\x00" * 206 + packed.tobytes()


def decode_hiragana(char_code: int) -> str | None:
    """Map a JIS X 0201 character code to hiragana, or ``None`` if not kana.

    ETL4 holds hiragana but encodes them with JIS X 0201, which only defines
    halfwidth katakana. NFKC widens the halfwidth form, and subtracting 0x60
    converts katakana to hiragana.
    """
    if not _HALFWIDTH_KATAKANA_START <= char_code <= _HALFWIDTH_KATAKANA_END:
        return None

    halfwidth = chr(_HALFWIDTH_UNICODE_START + char_code - _HALFWIDTH_KATAKANA_START)
    widened = unicodedata.normalize("NFKC", halfwidth)
    if len(widened) != 1:
        return None

    codepoint = ord(widened)
    if not 0x30A1 <= codepoint <= 0x30F6:
        return None
    return chr(codepoint - _KATAKANA_TO_HIRAGANA)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_etl4_records.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add jvqvae/data/sources/etl4_records.py tests/test_etl4_records.py
git commit -m "feat: parse ETL4 C-type records and decode hiragana codes"
```

---

### Task 7: ETL4 ingestion with verification and writer inference

**Files:**
- Create: `jvqvae/data/sources/etl4.py`
- Test: `tests/test_etl4.py`

**Interfaces:**
- Consumes: `parse_record`, `encode_record`, `decode_hiragana`, `WIDTH`, `HEIGHT`,
  `RECORD_SIZE`, `MAX_PIXEL_VALUE` (Task 6); `CharacterData` (Task 3); `pad_to_square`,
  `resize` (Task 3).
- Produces:
  - `EXPECTED_ZIP_BYTES = 4_949_352`, `EXPECTED_ZIP_MD5 = "cede897313b3617e364ddb17b8fad499"`,
    `EXPECTED_RECORDS = 6120`, `EXPECTED_CLASSES = 51`, `EXPECTED_WRITERS = 120`.
  - `verify_zip(path: str | Path) -> None` — raises `ValueError` on size or MD5 mismatch.
  - `infer_writer_ids(class_ids: np.ndarray, n_classes: int, n_writers: int) -> np.ndarray`
  - `parse_etl4_stream(raw: bytes) -> tuple[np.ndarray, np.ndarray]` returning
    `(images_uint8 (N, 76, 72), char_codes (N,))`.
  - `ingest_etl4(zip_path: str | Path, cache_path: str | Path, *, image_size: int = 64,
    verify: bool = True, strict: bool = True) -> CharacterData`
  - `load_cached_etl4(cache_path: str | Path) -> CharacterData`

**Why `strict`.** With the real file unavailable, `strict=True` enforces the spec's
assertions (6,120 records, 51 classes, 120 samples each). Tests build small synthetic
archives and pass `strict=False`. When the real zip arrives, run with the default and the
assertions either pass or tell you exactly which assumption was wrong.

**Writer inference.** Records are in a fixed writer-major order, so writer identity is
recoverable from the record index. Rather than trust that, `infer_writer_ids` tries
writer-major (`index // n_classes`) and then character-major (`index % n_writers`),
accepting whichever yields groups that each contain every class exactly once, and raising
if neither does.

- [ ] **Step 1: Write the failing test**

Create `tests/test_etl4.py`:

```python
import hashlib
import zipfile

import numpy as np
import pytest

from jvqvae.data.sources.etl4 import (
    EXPECTED_ZIP_BYTES,
    EXPECTED_ZIP_MD5,
    infer_writer_ids,
    ingest_etl4,
    load_cached_etl4,
    parse_etl4_stream,
    verify_zip,
)
from jvqvae.data.sources.etl4_records import HEIGHT, WIDTH, encode_record

N_CLASSES = 4
N_WRITERS = 3
CODES = [0xB1, 0xB2, 0xB3, 0xB4]  # あいうえ


def build_stream(writer_major=True):
    rng = np.random.default_rng(0)
    records = []
    if writer_major:
        order = [(w, c) for w in range(N_WRITERS) for c in range(N_CLASSES)]
    else:
        order = [(w, c) for c in range(N_CLASSES) for w in range(N_WRITERS)]
    for _, class_index in order:
        image = rng.integers(0, 16, size=(HEIGHT, WIDTH), dtype=np.uint8)
        records.append(encode_record(CODES[class_index], image))
    return b"".join(records)


@pytest.fixture
def fake_zip(tmp_path):
    path = tmp_path / "ETL4.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ETL4/ETL4C", build_stream())
    return path


def test_expected_fixity_constants_match_aist_listing():
    assert EXPECTED_ZIP_BYTES == 4_949_352
    assert EXPECTED_ZIP_MD5 == "cede897313b3617e364ddb17b8fad499"


def test_verify_rejects_a_wrong_sized_file(tmp_path):
    path = tmp_path / "ETL4.zip"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="size"):
        verify_zip(path)


def test_verify_accepts_a_file_matching_size_and_md5(tmp_path, monkeypatch):
    payload = b"x" * 32
    path = tmp_path / "ETL4.zip"
    path.write_bytes(payload)
    monkeypatch.setattr(
        "jvqvae.data.sources.etl4.EXPECTED_ZIP_BYTES", len(payload)
    )
    monkeypatch.setattr(
        "jvqvae.data.sources.etl4.EXPECTED_ZIP_MD5",
        hashlib.md5(payload).hexdigest(),
    )
    verify_zip(path)  # must not raise


def test_parse_stream_returns_all_records():
    images, codes = parse_etl4_stream(build_stream())
    assert images.shape == (N_CLASSES * N_WRITERS, HEIGHT, WIDTH)
    assert sorted(set(codes.tolist())) == CODES


def test_truncated_stream_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        parse_etl4_stream(build_stream()[:-10])


def test_writer_major_order_is_detected():
    class_ids = np.array([0, 1, 2, 3] * N_WRITERS, dtype=np.int64)
    writers = infer_writer_ids(class_ids, N_CLASSES, N_WRITERS)
    assert writers.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_character_major_order_is_detected():
    class_ids = np.repeat(np.arange(N_CLASSES), N_WRITERS).astype(np.int64)
    writers = infer_writer_ids(class_ids, N_CLASSES, N_WRITERS)
    assert writers.tolist() == [0, 1, 2] * N_CLASSES


def test_unrecognisable_order_raises():
    class_ids = np.zeros(N_CLASSES * N_WRITERS, dtype=np.int64)
    with pytest.raises(ValueError, match="writer order"):
        infer_writer_ids(class_ids, N_CLASSES, N_WRITERS)


def test_ingest_produces_normalized_square_images(fake_zip, tmp_path):
    data = ingest_etl4(
        fake_zip, tmp_path / "etl4.npz", image_size=64, verify=False, strict=False
    )

    assert len(data) == N_CLASSES * N_WRITERS
    assert data.image_shape == (64, 64)
    assert data.images.dtype == np.float32
    assert data.images.min() >= 0.0 and data.images.max() <= 1.0
    assert data.num_classes == N_CLASSES
    assert data.classes == ["あ", "い", "う", "え"]
    assert data.writer_ids is not None
    assert sorted(set(data.writer_ids.tolist())) == [0, 1, 2]


def test_ingest_writes_a_reloadable_cache(fake_zip, tmp_path):
    cache = tmp_path / "etl4.npz"
    original = ingest_etl4(fake_zip, cache, verify=False, strict=False)

    assert cache.exists()
    reloaded = load_cached_etl4(cache)
    assert np.array_equal(original.images, reloaded.images)
    assert np.array_equal(original.class_ids, reloaded.class_ids)
    assert np.array_equal(original.writer_ids, reloaded.writer_ids)
    assert original.classes == reloaded.classes


def test_strict_mode_rejects_a_short_archive(fake_zip, tmp_path):
    with pytest.raises(ValueError, match="6120"):
        ingest_etl4(fake_zip, tmp_path / "etl4.npz", verify=False, strict=True)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_etl4.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data.sources.etl4'`

- [ ] **Step 3: Write the implementation**

Create `jvqvae/data/sources/etl4.py`:

```python
"""ETL4 ingestion.

The repository never ships or downloads ETL4. AIST's Terms of Use clause 5
prohibits redistribution and direct URL links, so the user registers at
https://etlcdb.db.aist.go.jp/ and supplies the zip. Ingestion verifies it
against AIST's published fixity before parsing a byte.

Citation required by clause 4: Electrotechnical Laboratory, Japanese Technical
Committee for Optical Character Recognition, ETL Character Database, 1973-1984.
"""

import hashlib
import zipfile
from pathlib import Path

import numpy as np

from jvqvae.data.dataset import CharacterData
from jvqvae.data.sources.etl4_records import (
    MAX_PIXEL_VALUE,
    RECORD_SIZE,
    decode_hiragana,
    parse_record,
)
from jvqvae.data.transforms import pad_to_square, resize

EXPECTED_ZIP_BYTES = 4_949_352
EXPECTED_ZIP_MD5 = "cede897313b3617e364ddb17b8fad499"
EXPECTED_RECORDS = 6120
EXPECTED_CLASSES = 51
EXPECTED_WRITERS = 120

_ACQUISITION_HINT = (
    "Register at https://etlcdb.db.aist.go.jp/ to obtain ETL4. The repository "
    "cannot download or redistribute it."
)


def verify_zip(path: str | Path) -> None:
    """Check the supplied archive against AIST's published size and MD5."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. {_ACQUISITION_HINT}")

    actual_size = path.stat().st_size
    if actual_size != EXPECTED_ZIP_BYTES:
        raise ValueError(
            f"unexpected size for {path}: got {actual_size} bytes, "
            f"expected {EXPECTED_ZIP_BYTES}. {_ACQUISITION_HINT}"
        )

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_ZIP_MD5:
        raise ValueError(
            f"MD5 mismatch for {path}: got {digest.hexdigest()}, "
            f"expected {EXPECTED_ZIP_MD5}. {_ACQUISITION_HINT}"
        )


def parse_etl4_stream(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse concatenated fixed-size records into images and character codes."""
    if len(raw) % RECORD_SIZE != 0:
        raise ValueError(
            f"data length {len(raw)} is not a multiple of the "
            f"{RECORD_SIZE}-byte record size"
        )

    count = len(raw) // RECORD_SIZE
    images, codes = [], []
    for index in range(count):
        code, image = parse_record(raw[index * RECORD_SIZE : (index + 1) * RECORD_SIZE])
        codes.append(code)
        images.append(image)
    return np.stack(images), np.array(codes, dtype=np.int64)


def _groups_are_complete(
    class_ids: np.ndarray, group_ids: np.ndarray, n_classes: int
) -> bool:
    for group in np.unique(group_ids):
        members = class_ids[group_ids == group]
        if len(np.unique(members)) != n_classes or len(members) != n_classes:
            return False
    return True


def infer_writer_ids(
    class_ids: np.ndarray, n_classes: int, n_writers: int
) -> np.ndarray:
    """Recover writer identity from record order, validating the assumption.

    ETL4 is documented as writer-major, but rather than trusting that, both
    plausible orderings are tested against the decoded class sequence and the
    one that produces complete per-writer groups wins.
    """
    count = len(class_ids)
    candidates = {
        "writer-major": np.arange(count) // n_classes,
        "character-major": np.arange(count) % n_writers,
    }
    for group_ids in candidates.values():
        if _groups_are_complete(class_ids, group_ids, n_classes):
            return group_ids.astype(np.int64)

    raise ValueError(
        "could not determine writer order: neither writer-major nor "
        "character-major grouping yields one sample per class per writer"
    )


def _check_strict(class_ids: np.ndarray, classes: list[str]) -> None:
    if len(class_ids) != EXPECTED_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_RECORDS} records, found {len(class_ids)}. "
            "The 2018 notebook read 6113 and silently dropped seven."
        )
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"expected {EXPECTED_CLASSES} classes, found {len(classes)}")

    counts = np.bincount(class_ids, minlength=len(classes))
    if not np.all(counts == EXPECTED_WRITERS):
        offenders = {
            classes[i]: int(count)
            for i, count in enumerate(counts)
            if count != EXPECTED_WRITERS
        }
        raise ValueError(
            f"expected {EXPECTED_WRITERS} samples per class; these differ: {offenders}"
        )


def ingest_etl4(
    zip_path: str | Path,
    cache_path: str | Path,
    *,
    image_size: int = 64,
    verify: bool = True,
    strict: bool = True,
) -> CharacterData:
    """Verify, parse and cache ETL4 into a `CharacterData`."""
    if verify:
        verify_zip(zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        data_members = [name for name in members if "INFO" not in name.upper()]
        if len(data_members) != 1:
            raise ValueError(
                f"expected exactly one data member in {zip_path}, "
                f"found {data_members}"
            )
        raw = archive.read(data_members[0])

    raw_images, char_codes = parse_etl4_stream(raw)

    unique_codes = np.unique(char_codes)
    class_ids = np.searchsorted(unique_codes, char_codes).astype(np.int64)
    classes = [decode_hiragana(int(code)) or f"U+{code:04X}" for code in unique_codes]

    if strict:
        _check_strict(class_ids, classes)

    n_writers = len(class_ids) // max(len(classes), 1)
    writer_ids = infer_writer_ids(class_ids, len(classes), n_writers)

    images = raw_images.astype(np.float32) / float(MAX_PIXEL_VALUE)
    images = resize(pad_to_square(images), image_size)

    data = CharacterData(
        images=images,
        class_ids=class_ids,
        writer_ids=writer_ids,
        classes=classes,
    )
    _save_cache(data, cache_path)
    return data


def _save_cache(data: CharacterData, cache_path: str | Path) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=data.images,
        class_ids=data.class_ids,
        writer_ids=data.writer_ids,
        classes=np.array(data.classes, dtype=object),
    )


def load_cached_etl4(cache_path: str | Path) -> CharacterData:
    with np.load(cache_path, allow_pickle=True) as cached:
        return CharacterData(
            images=cached["images"],
            class_ids=cached["class_ids"],
            writer_ids=cached["writer_ids"],
            classes=[str(char) for char in cached["classes"]],
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_etl4.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS, all tests green, no network access, no ETL4 file present.

- [ ] **Step 6: Commit**

```bash
git add jvqvae/data/sources/etl4.py tests/test_etl4.py
git commit -m "feat: add verified ETL4 ingestion with writer inference"
```

---

### Task 8: PyTorch adapter, CLI, and README

**Files:**
- Create: `jvqvae/data/torch_dataset.py`
- Create: `jvqvae/cli.py`
- Create: `README.md`
- Modify: `pyproject.toml` (add the console script entry point)
- Test: `tests/test_torch_dataset.py`

**Interfaces:**
- Consumes: `CharacterData` (Task 3), `split_by_writer` / `carve_validation` (Task 4),
  `load_k49` / `download_k49` (Task 5), `ingest_etl4` (Task 7), `DataConfig` (Task 2).
- Produces:
  - `jvqvae.data.torch_dataset.CharacterTorchDataset(data: CharacterData,
    indices: np.ndarray | None = None)` — a `torch.utils.data.Dataset` yielding
    `(image: FloatTensor (1, H, W), class_id: LongTensor scalar)`.
  - `jvqvae.cli.main(argv: list[str] | None = None) -> int` with subcommands
    `fetch-k49` and `ingest-etl4`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_torch_dataset.py`:

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from jvqvae.data.dataset import CharacterData
from jvqvae.data.torch_dataset import CharacterTorchDataset


def make(n=8, n_classes=4):
    rng = np.random.default_rng(0)
    return CharacterData(
        images=rng.random((n, 8, 8)).astype(np.float32),
        class_ids=np.arange(n, dtype=np.int64) % n_classes,
        writer_ids=np.arange(n, dtype=np.int64) // n_classes,
        classes=[chr(0x3042 + i) for i in range(n_classes)],
    )


def test_item_has_a_channel_dimension():
    dataset = CharacterTorchDataset(make())
    image, class_id = dataset[0]

    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert class_id.dtype == torch.int64
    assert class_id.item() == 0


def test_indices_restrict_the_view():
    dataset = CharacterTorchDataset(make(), indices=np.array([5, 6]))

    assert len(dataset) == 2
    assert dataset[0][1].item() == 1  # original index 5, class 5 % 4


def test_works_with_a_dataloader():
    loader = DataLoader(CharacterTorchDataset(make()), batch_size=4)
    images, class_ids = next(iter(loader))

    assert images.shape == (4, 1, 8, 8)
    assert class_ids.shape == (4,)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_torch_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.data.torch_dataset'`

- [ ] **Step 3: Implement the adapter**

Create `jvqvae/data/torch_dataset.py`:

```python
"""PyTorch adapter over `CharacterData`.

Kept deliberately thin: the numpy container stays the single source of truth,
and this only adds the channel dimension and tensor conversion.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from jvqvae.data.dataset import CharacterData


class CharacterTorchDataset(Dataset):
    def __init__(self, data: CharacterData, indices: np.ndarray | None = None) -> None:
        self._data = data
        self._indices = (
            np.arange(len(data), dtype=np.int64) if indices is None else indices
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self._indices[position])
        image = torch.from_numpy(self._data.images[index]).unsqueeze(0)
        class_id = torch.tensor(int(self._data.class_ids[index]), dtype=torch.int64)
        return image, class_id
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_torch_dataset.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Implement the CLI**

Create `jvqvae/cli.py`:

```python
"""Command line entry points for data preparation."""

import argparse

from jvqvae.config import load_data_config
from jvqvae.data.sources.etl4 import ingest_etl4
from jvqvae.data.sources.k49 import download_k49, load_k49


def _fetch_k49(args: argparse.Namespace) -> int:
    directory = download_k49(args.dest)
    data = load_k49(directory, split="train", max_per_class=10)
    print(f"ready: {len(data)} sampled images, {data.num_classes} classes")
    return 0


def _ingest_etl4(args: argparse.Namespace) -> int:
    config = load_data_config(args.config)
    if config.etl4_zip is None:
        raise SystemExit(f"{args.config} does not set etl4_zip")

    cache = f"{config.cache_dir}/etl4_{config.image_size}.npz"
    data = ingest_etl4(
        config.etl4_zip,
        cache,
        image_size=config.image_size,
        verify=not args.no_verify,
        strict=not args.lenient,
    )
    print(
        f"ingested {len(data)} images, {data.num_classes} classes, "
        f"{len(set(data.writer_ids.tolist()))} writers -> {cache}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jvqvae")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch-k49", help="download Kuzushiji-49")
    fetch.add_argument("--dest", default="data/k49")
    fetch.set_defaults(handler=_fetch_k49)

    ingest = subparsers.add_parser(
        "ingest-etl4", help="verify and ingest a user-supplied ETL4 zip"
    )
    ingest.add_argument("--config", default="configs/etl4.yaml")
    ingest.add_argument(
        "--no-verify", action="store_true", help="skip the size and MD5 check"
    )
    ingest.add_argument(
        "--lenient", action="store_true", help="skip record and class count assertions"
    )
    ingest.set_defaults(handler=_ingest_etl4)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Register the console script**

In `pyproject.toml`, add immediately after the `dependencies` list:

```toml
[project.scripts]
jvqvae = "jvqvae.cli:main"
```

Then re-sync and check the CLI is wired up:

```bash
uv sync
uv run jvqvae --help
```

Expected: usage text listing `fetch-k49` and `ingest-etl4`.

- [ ] **Step 7: Download K49 and confirm the pipeline end to end**

```bash
uv run jvqvae fetch-k49 --dest data/k49
```

Expected: five files downloaded (the images archive is roughly 63 MB), then
`ready: 490 sampled images, 49 classes`.

Then confirm splitting works on real data:

```bash
uv run python -c "
from jvqvae.data.sources.k49 import load_k49
from jvqvae.data.splits import carve_validation
d = load_k49('data/k49', split='train', max_per_class=100)
tr, va = carve_validation(len(d), seed=0, n_val=200)
print(len(d), d.image_shape, d.num_classes, len(tr), len(va))
"
```

Expected: `4900 (32, 32) 49 4700 200`

- [ ] **Step 8: Write the README**

Create `README.md`:

````markdown
# Japanese handwriting with a VQ-VAE

A rebuild of a 2018 project that trained a convolutional VAE on ETL4 handwritten
hiragana. This version uses a VQ-VAE with a class-conditional transformer prior, so
you can ask for a specific character instead of sampling blindly.

Original write-up: [Computer made Japanese letters through Variational
Autoencoder](https://medium.com/@tomernahshon/computer-made-japanese-letters-through-variational-autoencoder-2fdb5b6a0990)

Design: `docs/superpowers/specs/2026-11-05-japanese-vq-vae-design.md`

## Setup

```bash
uv sync --python 3.12
uv run pytest
```

## Data

### Kuzushiji-49 (development)

Downloads without registration:

```bash
uv run jvqvae fetch-k49 --dest data/k49
```

### ETL4 (the real dataset)

ETL4 cannot be downloaded by this repository. AIST's Terms of Use permit
distribution only through their own site, so you must obtain it yourself:

1. Go to <https://etlcdb.db.aist.go.jp/>, accept the Terms of Use, and register to
   receive the download password.
2. Download ETL4 and place the zip at `data/raw/ETL4.zip`.
3. Run `uv run jvqvae ingest-etl4 --config configs/etl4.yaml`

Ingestion verifies the archive against AIST's published size and MD5 before parsing,
and asserts that it contains 6,120 records across 51 classes with 120 samples each.

## Attribution

ETL4: Electrotechnical Laboratory, Japanese Technical Committee for Optical Character
Recognition, ETL Character Database, 1973-1984.

Kuzushiji-49: "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset"
(created by NIJL and others), doi:10.20676/00000341.

ETL record layout and character set details were cross-checked against the
MIT-licensed [choo/etlcdb-image-extractor](https://github.com/choo/etlcdb-image-extractor).
````

- [ ] **Step 9: Run the full suite one more time**

Run: `uv run pytest -v`
Expected: PASS, all tests green.

- [ ] **Step 10: Commit and push**

```bash
git add jvqvae/data/torch_dataset.py jvqvae/cli.py README.md pyproject.toml tests/test_torch_dataset.py uv.lock
git commit -m "feat: add torch dataset adapter, data CLI, and README"
git push
```

---

## Definition of done for this plan

- `uv run pytest` is green with no network access and no ETL4 file present.
- `uv run jvqvae fetch-k49` downloads K49 and `load_k49` returns 32x32 normalized images
  across 49 classes.
- `uv run jvqvae ingest-etl4` exists, verifies fixity, and fails with an actionable
  message when the archive is absent or wrong.
- `CharacterTorchDataset` feeds a `DataLoader` with `(1, H, W)` float tensors, ready for
  plan 2.

## Deferred to plan 2

The VQ-VAE itself: quantizer with EMA updates and dead-code restarts, encoder and
decoder, the training loop, checkpointing, and codebook utilization logging.

## Open question for when AIST returns

`decode_hiragana` assumes ETL4's character-code byte is JIS X 0201 halfwidth katakana
mapped to hiragana. If the strict assertions in `ingest_etl4` fail on the real archive,
that assumption is where to look first — specifically whether the code byte is T56 rather
than JIS X 0201, since the 2018 notebook defined a T56 table but never used it.
