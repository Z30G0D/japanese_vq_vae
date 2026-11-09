# VQ-VAE Model and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable, checkpointable VQ-VAE that turns a character image into
an 8x8 grid of discrete codes and back, together with everything needed to train it and
to hand its codes to the prior in plan 3.

**Architecture:** An encoder downsamples to an 8x8 grid, an EMA vector quantizer snaps
each grid cell to one of 512 codebook entries, and a mirrored decoder reconstructs. The
quantizer is the interesting part: the codebook is a buffer updated by exponential
moving average rather than a learned parameter, gradients reach the encoder through a
straight-through estimator, and codes that go unused for too long are restarted from
live encoder outputs. Training, checkpointing, code extraction and figures are written
against the assembled model, never against its internals.

**Tech Stack:** Python 3.12, uv, PyTorch, numpy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-japanese-vq-vae-design.md`
**Plan 1 (complete):** `docs/superpowers/plans/2026-11-09-foundation-and-data-pipeline.md`

## Global Constraints

- Python 3.12. Dependencies limited to: torch, numpy, pillow, pyyaml, matplotlib,
  pytest. No experiment-tracking service.
- Package name is `jvqvae`. Run everything through `uv run`.
- The repository must never contain ETL4 data and must never download it. AIST's Terms
  of Use clause 5 prohibits redistribution and direct URL links.
- **The full test suite must pass with no network access, no ETL4 file present, and no
  trained checkpoint present.** Every model test builds its own tiny tensors.
- **No test may run more than a few hundred optimizer steps on tensors larger than a
  handful of images.** The suite's total runtime must stay under about 60 seconds. Tests
  that need a model use a deliberately tiny config (see the `tiny_config` fixture in
  Task 4), never the shipped ETL4 or K49 configs.
- All randomness goes through an explicit seed. No global `np.random` calls; torch
  randomness is seeded once per run through `set_seed`.
- Tests must run on CPU regardless of what hardware is present. Never call
  `select_device()` inside a test; pass `torch.device("cpu")` explicitly.
- Checkpoints load with `torch.load(..., weights_only=True)`. Never unpickle arbitrary
  objects from a checkpoint file.
- Everything written at runtime goes under `checkpoints/` or `results/`, both already
  gitignored. Never write into the repository root or `jvqvae/`.

### Execution note: do not train

The user will run the real training themselves. **No task in this plan runs a training
job on ETL4 or K49.** The definition of done is that `uv run jvqvae train-vqvae` works
and a few-step smoke run on a synthetic fixture drives the loss down. Do not launch long
jobs, do not leave background processes running, and do not commit checkpoints or
figures.

### Known deviations from the spec

Spec section 6 sketches configs as `etl4_vqvae.yaml`, `etl4_prior.yaml`, `k49_dev.yaml`
and a CLI of `ingest-etl4`, `train vqvae`, `train prior`, `sample`. This plan deviates
twice, both for consistency with what plan 1 already shipped:

1. **One run config per dataset, not per model.** `configs/etl4.yaml` and
   `configs/k49_dev.yaml` each grow `model:` and `train:` sections alongside their
   existing data fields, which move under a `data:` key. A separate `etl4_vqvae.yaml`
   would duplicate the entire data block, and the two copies would drift. Plan 3 adds a
   `prior:` section to the same files.
2. **Flat hyphenated CLI verbs.** `train-vqvae` and `extract-codes`, matching the
   existing `fetch-k49` and `ingest-etl4`, rather than a nested `train vqvae`.

Spec section 5 says the decoder mirrors the encoder "back to 64x64x1" without naming an
output activation. This plan uses a sigmoid, because `CharacterData` guarantees inputs
in [0, 1] and a bounded decoder output means every reconstruction and every sample is a
valid image without clipping.

---

## File Structure

| File | Responsibility |
|---|---|
| `jvqvae/config.py` | **Modify.** Add `VQVAEConfig`, `TrainConfig`, `RunConfig`, `load_run_config` |
| `jvqvae/models/__init__.py` | New package marker |
| `jvqvae/models/blocks.py` | Residual block, encoder, decoder |
| `jvqvae/models/quantizer.py` | EMA vector quantizer with dead-code restarts |
| `jvqvae/models/vqvae.py` | Model assembly, checkpoint save and load |
| `jvqvae/train/__init__.py` | New package marker |
| `jvqvae/train/seed.py` | `set_seed` across python, numpy and torch |
| `jvqvae/train/vqvae.py` | Training loop, per-epoch metrics, JSON run record |
| `jvqvae/codes.py` | Extract code grids from a trained model; save and load them |
| `jvqvae/figures.py` | Reconstruction grid figure |
| `jvqvae/cli.py` | **Modify.** Add `train-vqvae` and `extract-codes` |
| `configs/etl4.yaml`, `configs/k49_dev.yaml` | **Modify.** Nest under `data:`, add `model:` and `train:` |
| `README.md` | **Modify.** Document training and the config layout |
| `tests/test_config.py` | **Modify.** Cover the new sections |
| `tests/test_blocks.py`, `test_quantizer.py`, `test_vqvae.py`, `test_train_vqvae.py`, `test_codes.py`, `test_figures.py` | New |

---

## Task 1: Run configuration

**Files:**
- Modify: `jvqvae/config.py`
- Modify: `jvqvae/cli.py`
- Modify: `configs/etl4.yaml`, `configs/k49_dev.yaml`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `DataConfig`, `SplitConfig`, `_build` from `jvqvae/config.py` (plan 1).
- Produces:
  - `VQVAEConfig(num_codes=512, embedding_dim=64, hidden_channels=128, residual_hidden=32, num_residual_blocks=2, num_downsample=3, commitment_cost=0.25, ema_decay=0.99, dead_code_steps=50)`, frozen.
  - `TrainConfig(seed=0, epochs=50, batch_size=64, learning_rate=3e-4, max_steps=None, device="auto", checkpoint_dir="checkpoints", results_dir="results")`, frozen.
  - `RunConfig(name, data, model, train)`, frozen, with property `latent_size -> int`.
  - `load_run_config(path: str | Path) -> RunConfig`.
  - `load_data_config` is **removed**; callers use `load_run_config(path).data`.

This task is the migration. Everything after it assumes `RunConfig` exists.

- [ ] **Step 1: Write the failing tests**

Replace the whole of `tests/test_config.py` with this. It keeps plan 1's coverage,
re-rooted onto `RunConfig`, and fixes a plan 1 finding: the shipped-config test used to
resolve `configs/` relative to the current working directory.

```python
from pathlib import Path

import pytest

from jvqvae.config import (
    DataConfig,
    RunConfig,
    SplitConfig,
    TrainConfig,
    VQVAEConfig,
    load_run_config,
)

REPO_ROOT = Path(__file__).parent.parent


def write(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(text)
    return path


MINIMAL = "data:\n  source: k49\n  image_size: 32\n"


def test_defaults_fill_in_every_section(tmp_path):
    config = load_run_config(write(tmp_path, MINIMAL))

    assert config.data.source == "k49"
    assert config.model.num_codes == 512
    assert config.model.embedding_dim == 64
    assert config.train.epochs == 50
    assert config.data.split.seed == 0
    assert config.name == "run"


def test_sections_override_defaults(tmp_path):
    config = load_run_config(
        write(
            tmp_path,
            "name: experiment\n"
            "data:\n  source: k49\n  image_size: 32\n"
            "model:\n  num_codes: 64\n  num_downsample: 2\n"
            "train:\n  epochs: 3\n  batch_size: 8\n",
        )
    )

    assert config.name == "experiment"
    assert config.model.num_codes == 64
    assert config.train.epochs == 3
    assert config.train.batch_size == 8
    assert config.train.learning_rate == 3e-4


def test_latent_size_follows_image_size_and_downsampling(tmp_path):
    config = load_run_config(
        write(
            tmp_path,
            "data:\n  source: k49\n  image_size: 32\n"
            "model:\n  num_downsample: 2\n",
        )
    )
    assert config.latent_size == 8


def test_a_missing_data_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="data"):
        load_run_config(write(tmp_path, "model:\n  num_codes: 64\n"))


def test_unknown_top_level_sections_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="prior"):
        load_run_config(write(tmp_path, MINIMAL + "prior:\n  layers: 4\n"))


def test_unknown_keys_inside_a_section_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="codebook_size"):
        load_run_config(write(tmp_path, MINIMAL + "model:\n  codebook_size: 64\n"))


def test_a_non_mapping_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="model"):
        load_run_config(write(tmp_path, MINIMAL + "model: 0\n"))


def test_error_messages_name_the_offending_file(tmp_path):
    path = write(tmp_path, MINIMAL + "model:\n  codebook_size: 64\n")
    with pytest.raises(ValueError, match=str(path)):
        load_run_config(path)


def test_an_image_size_that_does_not_reach_the_grid_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        load_run_config(
            write(
                tmp_path,
                "data:\n  source: k49\n  image_size: 20\n"
                "model:\n  num_downsample: 3\n",
            )
        )


@pytest.mark.parametrize(
    "section,text",
    [
        ("model", "model:\n  num_codes: 0\n"),
        ("model", "model:\n  embedding_dim: -1\n"),
        ("model", "model:\n  num_downsample: 0\n"),
        ("model", "model:\n  ema_decay: 1.5\n"),
        ("model", "model:\n  commitment_cost: -0.1\n"),
        ("train", "train:\n  epochs: 0\n"),
        ("train", "train:\n  batch_size: 0\n"),
        ("train", "train:\n  learning_rate: 0\n"),
        ("train", "train:\n  max_steps: 0\n"),
    ],
)
def test_out_of_range_scalars_are_rejected(tmp_path, section, text):
    with pytest.raises(ValueError):
        load_run_config(write(tmp_path, MINIMAL + text))


def test_split_config_still_validates():
    with pytest.raises(ValueError, match="negative"):
        SplitConfig(val_writers=-1)


@pytest.mark.parametrize("name", ["etl4.yaml", "k49_dev.yaml"])
def test_shipped_configs_load_and_reach_an_eight_by_eight_grid(name):
    config = load_run_config(REPO_ROOT / "configs" / name)
    assert config.latent_size == 8
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, VQVAEConfig)
    assert isinstance(config.train, TrainConfig)


def test_the_etl4_config_holds_the_ninety_six_twelve_twelve_split():
    config = load_run_config(REPO_ROOT / "configs" / "etl4.yaml")
    assert config.data.source == "etl4"
    assert config.data.image_size == 64
    assert config.data.split.val_writers == 12
    assert config.data.split.test_writers == 12


def test_run_config_can_be_built_directly():
    config = RunConfig(
        name="direct",
        data=DataConfig(source="k49", image_size=32),
        model=VQVAEConfig(num_downsample=2),
        train=TrainConfig(),
    )
    assert config.latent_size == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -x -q`
Expected: FAIL at import with `ImportError: cannot import name 'RunConfig' from 'jvqvae.config'`.

- [ ] **Step 3: Add the new config dataclasses**

In `jvqvae/config.py`, after the existing `DataConfig`, add:

```python
@dataclass(frozen=True)
class VQVAEConfig:
    """Shape and behaviour of the VQ-VAE.

    ``num_downsample`` sets how far the encoder reduces the image, so with an
    ``image_size`` of 64 three stages give the 8x8 code grid the prior expects.
    It is a field rather than a constant so a 16x16 grid is available as an
    ablation without touching the model.
    """

    num_codes: int = 512
    embedding_dim: int = 64
    hidden_channels: int = 128
    residual_hidden: int = 32
    num_residual_blocks: int = 2
    num_downsample: int = 3
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    dead_code_steps: int = 50

    def __post_init__(self) -> None:
        positive = {
            "num_codes": self.num_codes,
            "embedding_dim": self.embedding_dim,
            "hidden_channels": self.hidden_channels,
            "residual_hidden": self.residual_hidden,
            "num_downsample": self.num_downsample,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_residual_blocks < 0:
            raise ValueError(
                f"num_residual_blocks must not be negative, "
                f"got {self.num_residual_blocks}"
            )
        if self.commitment_cost < 0:
            raise ValueError(
                f"commitment_cost must not be negative, got {self.commitment_cost}"
            )
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")
        if self.dead_code_steps < 0:
            raise ValueError(
                f"dead_code_steps must not be negative, got {self.dead_code_steps}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """Optimization settings.

    ``max_steps`` caps total optimizer steps regardless of ``epochs``, which is
    what makes a smoke run cheap.
    """

    seed: int = 0
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_steps: int | None = None
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                f"max_steps must be positive when set, got {self.max_steps}"
            )


@dataclass(frozen=True)
class RunConfig:
    """Everything one training run needs, from one file."""

    name: str
    data: DataConfig
    model: VQVAEConfig = field(default_factory=VQVAEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        stride = 2**self.model.num_downsample
        if self.data.image_size % stride:
            raise ValueError(
                f"image_size {self.data.image_size} is not divisible by "
                f"{stride} ({self.model.num_downsample} downsampling stages), "
                "so the encoder cannot reach a whole code grid"
            )

    @property
    def latent_size(self) -> int:
        """Side length of the square code grid the encoder produces."""
        return self.data.image_size // 2**self.model.num_downsample
```

- [ ] **Step 4: Replace the loader**

Delete `load_data_config` and add in its place:

```python
_SECTIONS = ("name", "data", "model", "train")


def _build_data_config(raw: Any, *, where: str) -> DataConfig:
    if not isinstance(raw, dict):
        raise ValueError(
            f"{where}: expected a mapping for DataConfig, got {type(raw).__name__}"
        )

    raw = dict(raw)
    split_raw = raw.get("split")
    raw["split"] = _build(
        SplitConfig, {} if split_raw is None else split_raw, where=f"{where}:split"
    )
    return _build(DataConfig, raw, where=where)


def load_run_config(path: str | Path) -> RunConfig:
    """Parse one YAML file into the full configuration for a run."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: expected a mapping at the top level, got {type(raw).__name__}"
        )

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(
            f"{path}: unknown top-level section(s): {', '.join(sorted(unknown))}. "
            f"Expected any of: {', '.join(_SECTIONS)}"
        )
    if "data" not in raw:
        raise ValueError(f"{path}: missing the required 'data' section")

    def section(name: str) -> Any:
        # An absent section defaults, but an explicitly wrong one must still be
        # rejected, so only None is replaced -- not every falsy value.
        value = raw.get(name)
        return {} if value is None else value

    return RunConfig(
        name=raw.get("name") or path.stem,
        data=_build_data_config(raw["data"], where=f"{path}:data"),
        model=_build(VQVAEConfig, section("model"), where=f"{path}:model"),
        train=_build(TrainConfig, section("train"), where=f"{path}:train"),
    )
```

- [ ] **Step 5: Migrate the shipped configs**

`configs/etl4.yaml`:

```yaml
name: etl4
data:
  source: etl4
  image_size: 64
  etl4_zip: data/raw/ETL4.zip
  split:
    seed: 0
    val_writers: 12
    test_writers: 12
model:
  num_codes: 512
  embedding_dim: 64
  num_downsample: 3
train:
  epochs: 50
  batch_size: 64
  learning_rate: 0.0003
```

`configs/k49_dev.yaml`:

```yaml
name: k49_dev
data:
  source: k49
  image_size: 32
  k49_dir: data/k49
  k49_max_per_class: 500
  split:
    seed: 0
model:
  num_codes: 512
  embedding_dim: 64
  # 32x32 needs two stages, not three, to land on the same 8x8 grid as ETL4,
  # which is what keeps the prior identical across the two sources.
  num_downsample: 2
train:
  epochs: 20
  batch_size: 128
  learning_rate: 0.0003
```

- [ ] **Step 6: Update the CLI to the new loader**

In `jvqvae/cli.py`, change the import and the two uses in `_ingest_etl4`:

```python
from jvqvae.config import load_run_config
```

```python
def _ingest_etl4(args: argparse.Namespace) -> int:
    config = load_run_config(args.config).data
    if config.etl4_zip is None:
        raise SystemExit(
            f"{args.config} does not set data.etl4_zip. Add a line such as "
            "'etl4_zip: data/raw/ETL4.zip' under the 'data:' section, pointing "
            "at the archive you downloaded from AIST."
        )
```

The rest of the function is unchanged.

- [ ] **Step 7: Pin the Python version**

In `pyproject.toml`, change `requires-python = ">=3.12"` to:

```toml
requires-python = ">=3.12,<3.13"
```

- [ ] **Step 8: Fix the CLI test for the new config shape**

In `tests/test_cli.py`, both config-writing tests must nest under `data:`. Replace each
occurrence of

```python
    config.write_text(
        "source: etl4\nimage_size: 64\n"
        f"cache_dir: {tmp_path}\netl4_zip: {tmp_path}/ETL4.zip\n"
    )
```

with

```python
    config.write_text(
        "data:\n  source: etl4\n  image_size: 64\n"
        f"  cache_dir: {tmp_path}\n  etl4_zip: {tmp_path}/ETL4.zip\n"
    )
```

and in `test_a_config_without_a_zip_path_says_what_to_add` replace

```python
    config.write_text("source: etl4\nimage_size: 64\n")
```

with

```python
    config.write_text("data:\n  source: etl4\n  image_size: 64\n")
```

That last test matches on `etl4_zip: data/raw/ETL4.zip`, which the new message still
contains, so its assertion is unchanged.

- [ ] **Step 9: Update the README example, which calls the deleted loader**

In `README.md`, under `## Using the data`, change the import and the call:

```python
from jvqvae.config import load_run_config
from jvqvae.data.pipeline import load_splits
from jvqvae.data.torch_dataset import CharacterTorchDataset

splits = load_splits(load_run_config("configs/etl4.yaml").data)
```

- [ ] **Step 10: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS. `tests/test_pipeline.py` builds `DataConfig` directly rather than going
through the loader, so it needs no change. Any failure is a real one; fix it rather than
loosening the test.

Then confirm nothing still references the removed loader:

Run: `rg 'load_data_config' --glob '!docs/**'`
Expected: no matches.

- [ ] **Step 11: Verify the CLI still works**

Run: `uv run jvqvae ingest-etl4 --config configs/etl4.yaml`
Expected: a clean `FileNotFoundError`-style message about `data/raw/ETL4.zip` not
existing and how to obtain ETL4 — **not** a `TypeError` or a traceback about config
parsing.

- [ ] **Step 12: Commit**

```bash
git add jvqvae/config.py jvqvae/cli.py configs tests/test_config.py tests/test_cli.py pyproject.toml README.md
git commit -m "feat: add run configuration covering model and training

One file per dataset now carries data, model and training settings, so a
run is reproducible from a single path and the data block is not duplicated
across a config per model. Scalar ranges are validated where they enter."
```

---

## Task 2: Residual blocks, encoder and decoder

**Files:**
- Create: `jvqvae/models/__init__.py`
- Create: `jvqvae/models/blocks.py`
- Test: `tests/test_blocks.py`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces:
  - `ResidualBlock(channels: int, hidden: int)`, an `nn.Module`, shape-preserving.
  - `Encoder(in_channels, hidden_channels, embedding_dim, num_downsample, num_residual_blocks, residual_hidden)` mapping `(B, in_channels, S, S)` to `(B, embedding_dim, S // 2**num_downsample, ...)`.
  - `Decoder(embedding_dim, hidden_channels, out_channels, num_downsample, num_residual_blocks, residual_hidden)` mapping `(B, embedding_dim, G, G)` to `(B, out_channels, G * 2**num_downsample, ...)` with a sigmoid output.

All constructor arguments after the first are keyword-only in both `Encoder` and
`Decoder`, so a transposed pair of integers is a `TypeError` rather than a silently
wrong model.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_blocks.py`:

```python
import pytest
import torch

from jvqvae.models.blocks import Decoder, Encoder, ResidualBlock


def test_residual_block_preserves_shape():
    block = ResidualBlock(8, 4)
    x = torch.randn(2, 8, 5, 5)
    assert block(x).shape == x.shape


def test_residual_block_actually_adds_a_residual():
    """A zeroed branch must leave the input untouched, which proves the skip."""
    block = ResidualBlock(8, 4)
    for parameter in block.parameters():
        torch.nn.init.zeros_(parameter)

    x = torch.randn(2, 8, 5, 5)
    assert torch.allclose(block(x), x)


@pytest.mark.parametrize("image_size,num_downsample", [(64, 3), (32, 2), (16, 1)])
def test_encoder_reaches_the_expected_grid(image_size, num_downsample):
    encoder = Encoder(
        1,
        hidden_channels=16,
        embedding_dim=6,
        num_downsample=num_downsample,
        num_residual_blocks=2,
        residual_hidden=4,
    )
    out = encoder(torch.rand(2, 1, image_size, image_size))

    expected = image_size // 2**num_downsample
    assert out.shape == (2, 6, expected, expected)


@pytest.mark.parametrize("grid,num_downsample", [(8, 3), (8, 2), (4, 1)])
def test_decoder_restores_the_image_size(grid, num_downsample):
    decoder = Decoder(
        6,
        hidden_channels=16,
        out_channels=1,
        num_downsample=num_downsample,
        num_residual_blocks=2,
        residual_hidden=4,
    )
    out = decoder(torch.randn(2, 6, grid, grid))

    expected = grid * 2**num_downsample
    assert out.shape == (2, 1, expected, expected)


def test_decoder_output_is_a_valid_image():
    decoder = Decoder(
        6,
        hidden_channels=16,
        out_channels=1,
        num_downsample=2,
        num_residual_blocks=1,
        residual_hidden=4,
    )
    out = decoder(torch.randn(4, 6, 8, 8) * 10)

    assert out.min() >= 0.0 and out.max() <= 1.0


def test_encoder_and_decoder_round_trip_the_shape():
    shared = dict(num_downsample=3, num_residual_blocks=2, residual_hidden=4)
    encoder = Encoder(1, hidden_channels=16, embedding_dim=6, **shared)
    decoder = Decoder(6, hidden_channels=16, out_channels=1, **shared)

    images = torch.rand(2, 1, 64, 64)
    assert decoder(encoder(images)).shape == images.shape


def test_positional_misuse_is_rejected():
    with pytest.raises(TypeError):
        Encoder(1, 16, 6, 3, 2, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_blocks.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.models'`.

- [ ] **Step 3: Create the package marker**

Create `jvqvae/models/__init__.py` as an empty file.

- [ ] **Step 4: Implement the blocks**

Create `jvqvae/models/blocks.py`:

```python
"""Convolutional encoder and decoder around the quantizer.

Both are plain stacks with no normalization layers: at this data scale the
model is small enough that batch statistics add variance without buying
stability, and keeping them out means a batch of one behaves like a batch of
many at sampling time.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block, 3x3 then 1x1."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


class Encoder(nn.Module):
    """Downsample an image to a grid of embedding vectors."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        embedding_dim: int,
        num_downsample: int,
        num_residual_blocks: int,
        residual_hidden: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for stage in range(num_downsample):
            # Widen once at the start, then hold: the first stage sees raw
            # pixels and does not need the full width.
            out = hidden_channels // 2 if stage == 0 else hidden_channels
            layers += [
                nn.Conv2d(channels, out, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ]
            channels = out

        layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1))
        layers += [
            ResidualBlock(hidden_channels, residual_hidden)
            for _ in range(num_residual_blocks)
        ]
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Decoder(nn.Module):
    """Mirror of the encoder, from a grid of embedding vectors to an image."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_channels: int,
        out_channels: int,
        num_downsample: int,
        num_residual_blocks: int,
        residual_hidden: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1)
        ]
        layers += [
            ResidualBlock(hidden_channels, residual_hidden)
            for _ in range(num_residual_blocks)
        ]
        layers.append(nn.ReLU())

        channels = hidden_channels
        for stage in range(num_downsample):
            is_last = stage == num_downsample - 1
            out = out_channels if is_last else hidden_channels // 2
            layers.append(
                nn.ConvTranspose2d(channels, out, kernel_size=4, stride=2, padding=1)
            )
            if not is_last:
                layers.append(nn.ReLU())
            channels = out

        # Inputs are guaranteed to lie in [0, 1], so a bounded output means every
        # reconstruction and every sample is a valid image without clipping.
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_blocks.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add jvqvae/models tests/test_blocks.py
git commit -m "feat: add residual encoder and decoder blocks"
```

---

## Task 3: EMA vector quantizer

**Files:**
- Create: `jvqvae/models/quantizer.py`
- Test: `tests/test_quantizer.py`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces:
  - `QuantizerOutput`, a dataclass with fields `quantized` (B, D, H, W), `codes` (B, H, W) int64, `commitment_loss` scalar tensor, `perplexity` scalar tensor, `num_active` int, `num_restarted` int.
  - `VectorQuantizerEMA(num_codes, embedding_dim, *, commitment_cost=0.25, decay=0.99, eps=1e-5, dead_code_steps=50)`, an `nn.Module` whose `forward(inputs) -> QuantizerOutput`.
  - `VectorQuantizerEMA.lookup(codes: torch.Tensor) -> torch.Tensor` mapping `(B, H, W)` int64 to `(B, D, H, W)`.

This is the subtlest module in the plan. Three things are easy to get wrong and each has
a test below: the straight-through estimator must pass gradients to the encoder while
the codebook stays out of autograd; the EMA update must move a code toward the mean of
the vectors assigned to it; and a dead-code restart must reset the code's EMA state too,
or the moving average drags it straight back to where it died.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quantizer.py`:

```python
import pytest
import torch

from jvqvae.models.quantizer import VectorQuantizerEMA


def make(num_codes=4, dim=3, **kwargs):
    torch.manual_seed(0)
    return VectorQuantizerEMA(num_codes, dim, **kwargs)


def place_codebook(quantizer, rows):
    """Set the codebook *and* the EMA state that reproduces it.

    Writing only `codebook` is not enough: the next EMA step recomputes the
    codebook from `ema_weight`, so a test that sets one and not the other is
    testing a state the model can never actually be in, and would pass even if
    restarts were broken.
    """
    table = torch.tensor(rows)
    quantizer.codebook.copy_(table)
    quantizer.ema_weight.copy_(table)
    quantizer.ema_cluster_size.copy_(torch.ones(len(rows)))


def test_output_shapes_and_dtypes():
    quantizer = make()
    out = quantizer(torch.randn(2, 3, 4, 4))

    assert out.quantized.shape == (2, 3, 4, 4)
    assert out.codes.shape == (2, 4, 4)
    assert out.codes.dtype == torch.int64
    assert out.commitment_loss.ndim == 0
    assert out.perplexity.ndim == 0


def test_every_output_vector_is_exactly_a_codebook_entry():
    quantizer = make().eval()
    inputs = torch.randn(2, 3, 4, 4)
    out = quantizer(inputs)

    expected = quantizer.codebook[out.codes]  # (B, H, W, D)
    assert torch.allclose(out.quantized.permute(0, 2, 3, 1), expected, atol=1e-6)


def test_codes_are_the_nearest_codebook_entry():
    quantizer = make().eval()
    # Feed the codebook entries themselves; each must map to its own index.
    inputs = quantizer.codebook.t().reshape(1, 3, 1, 4)
    assert quantizer(inputs).codes.flatten().tolist() == [0, 1, 2, 3]


def test_the_codebook_is_not_a_gradient_parameter():
    """EMA owns the codebook, so autograd must not also be updating it."""
    quantizer = make()
    assert "codebook" not in dict(quantizer.named_parameters())
    assert "codebook" in dict(quantizer.named_buffers())
    assert list(quantizer.parameters()) == []


def test_gradients_reach_the_encoder_through_the_quantizer():
    """The straight-through estimator is the only path for them."""
    quantizer = make().eval()
    encoder = torch.nn.Conv2d(3, 3, kernel_size=1)
    inputs = encoder(torch.randn(2, 3, 4, 4))

    quantizer(inputs).quantized.sum().backward()

    assert encoder.weight.grad is not None
    assert encoder.weight.grad.abs().sum() > 0


def test_the_straight_through_gradient_is_the_identity():
    quantizer = make().eval()
    inputs = torch.randn(2, 3, 4, 4, requires_grad=True)

    quantizer(inputs).quantized.sum().backward()

    assert torch.allclose(inputs.grad, torch.ones_like(inputs))


def test_commitment_loss_is_zero_when_inputs_sit_on_the_codebook():
    quantizer = make().eval()
    inputs = quantizer.codebook.t().reshape(1, 3, 1, 4)

    assert quantizer(inputs).commitment_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_commitment_loss_scales_with_the_commitment_cost():
    inputs = torch.randn(2, 3, 4, 4)
    cheap = make(commitment_cost=0.25).eval()
    dear = make(commitment_cost=1.0).eval()

    assert dear(inputs).commitment_loss.item() == pytest.approx(
        4 * cheap(inputs).commitment_loss.item(), rel=1e-5
    )


def test_perplexity_is_one_when_every_vector_picks_the_same_code():
    quantizer = make().eval()
    single = quantizer.codebook[2].view(1, 3, 1, 1).expand(4, 3, 2, 2).contiguous()
    out = quantizer(single)

    assert out.codes.unique().tolist() == [2]
    assert out.perplexity.item() == pytest.approx(1.0, abs=1e-4)
    assert out.num_active == 1


def test_perplexity_equals_the_code_count_under_uniform_use():
    quantizer = make(num_codes=4, dim=3).eval()
    inputs = quantizer.codebook.t().reshape(1, 3, 1, 4)

    assert quantizer(inputs).perplexity.item() == pytest.approx(4.0, abs=1e-4)
    assert quantizer(inputs).num_active == 4


def test_ema_moves_a_code_toward_the_vectors_assigned_to_it():
    quantizer = make(num_codes=2, dim=2, decay=0.5, dead_code_steps=0).train()
    place_codebook(quantizer, [[0.0, 0.0], [10.0, 10.0]])

    target = torch.tensor([1.0, 1.0])
    before = (quantizer.codebook[0] - target).norm().item()
    batch = target.view(1, 2, 1, 1).expand(8, 2, 1, 1).contiguous()
    for _ in range(5):
        quantizer(batch)
    after = (quantizer.codebook[0] - target).norm().item()

    assert after < before
    assert after < 0.1


def test_the_codebook_does_not_move_in_eval_mode():
    quantizer = make(dead_code_steps=0).eval()
    before = quantizer.codebook.clone()

    quantizer(torch.randn(4, 3, 4, 4))

    assert torch.equal(quantizer.codebook, before)


def test_unused_codes_are_restarted_from_live_encoder_outputs():
    quantizer = make(num_codes=4, dim=2, dead_code_steps=2).train()
    # Put code 3 far away so nothing ever selects it.
    place_codebook(
        quantizer, [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [500.0, 500.0]]
    )
    batch = torch.randn(16, 2, 2, 2) * 0.05

    restarted = 0
    for _ in range(4):
        restarted += quantizer(batch).num_restarted

    assert restarted >= 1
    assert quantizer.codebook[3].abs().max() < 10.0


def test_a_restarted_code_does_not_drift_back_to_where_it_died():
    """The restart must reset the code's EMA state, not just the codebook row.

    Resetting only the row leaves `ema_weight` pointing at the dead position,
    and the next update recomputes the row straight back to roughly 900.
    """
    quantizer = make(num_codes=2, dim=2, decay=0.99, dead_code_steps=1).train()
    place_codebook(quantizer, [[0.0, 0.0], [900.0, 900.0]])
    batch = torch.zeros(8, 2, 2, 2)

    for _ in range(10):
        quantizer(batch)

    # A restart happens at the end of a step, so the codebook always looks
    # clean immediately after one -- asserting here would pass even with the
    # bug. Stop restarting and take one more step: if the EMA state was not
    # reset alongside the row, the code springs back to where it died.
    quantizer.dead_code_steps = 0
    quantizer(batch)

    assert quantizer.codebook[1].abs().max() < 1.0


def test_restarts_are_disabled_when_the_threshold_is_zero():
    quantizer = make(num_codes=4, dim=2, dead_code_steps=0).train()
    place_codebook(
        quantizer, [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [500.0, 500.0]]
    )
    batch = torch.randn(16, 2, 2, 2) * 0.05

    for _ in range(5):
        assert quantizer(batch).num_restarted == 0
    assert quantizer.codebook[3].abs().max() > 100.0


def test_lookup_inverts_the_code_assignment():
    quantizer = make().eval()
    inputs = torch.randn(2, 3, 4, 4)
    out = quantizer(inputs)

    assert torch.allclose(quantizer.lookup(out.codes), out.quantized, atol=1e-6)


def test_lookup_rejects_out_of_range_codes():
    quantizer = make(num_codes=4, dim=3)
    with pytest.raises(ValueError, match="out of range"):
        quantizer.lookup(torch.tensor([[[4]]]))


def test_buffers_survive_a_state_dict_round_trip():
    original = make(num_codes=4, dim=2, dead_code_steps=0).train()
    original(torch.randn(8, 2, 3, 3))

    restored = make(num_codes=4, dim=2, dead_code_steps=0)
    restored.load_state_dict(original.state_dict())

    assert torch.equal(restored.codebook, original.codebook)
    assert torch.equal(restored.ema_cluster_size, original.ema_cluster_size)
    assert torch.equal(restored.steps_since_used, original.steps_since_used)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_quantizer.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.models.quantizer'`.

- [ ] **Step 3: Implement the quantizer**

Create `jvqvae/models/quantizer.py`:

```python
"""Vector quantizer with an EMA-updated codebook.

The codebook is a buffer, not a parameter. It is updated by an exponential
moving average of the encoder outputs assigned to each code, which at this data
scale is markedly more stable than learning it by gradient alongside everything
else. Gradients still reach the encoder, through a straight-through estimator.

Codebook collapse -- most codes going permanently unused -- is the expected
failure mode with roughly 4,900 training images against 512 codes. Dead-code
restarts plus the perplexity reported on every batch turn that from a silent
failure into a visible one.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor  # (B, D, H, W), straight-through
    codes: torch.Tensor  # (B, H, W) int64
    commitment_loss: torch.Tensor  # scalar
    perplexity: torch.Tensor  # scalar, effective number of codes in use
    num_active: int  # codes selected at least once in this batch
    num_restarted: int  # dead codes reinitialized on this step


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: int,
        embedding_dim: int,
        *,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_steps: int = 50,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.dead_code_steps = dead_code_steps

        codebook = torch.randn(num_codes, embedding_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_weight", codebook.clone())
        self.register_buffer(
            "steps_since_used", torch.zeros(num_codes, dtype=torch.long)
        )

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """Map a grid of code indices back to embedding vectors."""
        if codes.numel() and (
            int(codes.max()) >= self.num_codes or int(codes.min()) < 0
        ):
            raise ValueError(
                f"code index out of range for a codebook of {self.num_codes}: "
                f"got [{int(codes.min())}, {int(codes.max())}]"
            )
        return self.codebook[codes].permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs: torch.Tensor) -> QuantizerOutput:
        batch, dim, height, width = inputs.shape
        if dim != self.embedding_dim:
            raise ValueError(
                f"expected {self.embedding_dim} channels, got {dim}"
            )

        flat = inputs.permute(0, 2, 3, 1).reshape(-1, dim)

        # |a - b|^2 expanded, so the codebook is touched once rather than per row.
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        encodings = F.one_hot(indices, self.num_codes).to(flat.dtype)
        quantized_flat = self.codebook[indices]

        num_restarted = self._update(flat, encodings) if self.training else 0

        # Pull the encoder toward the code it chose. The other direction is the
        # EMA's job, so the codebook side is detached.
        commitment_loss = self.commitment_cost * F.mse_loss(
            flat, quantized_flat.detach()
        )

        quantized = quantized_flat.view(batch, height, width, dim).permute(0, 3, 1, 2)
        # Straight-through: the decoder sees the quantized grid, the encoder sees
        # the gradient as though quantization never happened.
        quantized = inputs + (quantized - inputs).detach()

        probabilities = encodings.mean(dim=0)
        perplexity = torch.exp(
            -(probabilities * torch.log(probabilities + 1e-10)).sum()
        )

        return QuantizerOutput(
            quantized=quantized,
            codes=indices.view(batch, height, width),
            commitment_loss=commitment_loss,
            perplexity=perplexity,
            num_active=int((probabilities > 0).sum()),
            num_restarted=num_restarted,
        )

    @torch.no_grad()
    def _update(self, flat: torch.Tensor, encodings: torch.Tensor) -> int:
        cluster_size = encodings.sum(dim=0)
        self.ema_cluster_size.mul_(self.decay).add_(
            cluster_size, alpha=1 - self.decay
        )
        self.ema_weight.mul_(self.decay).add_(
            encodings.t() @ flat, alpha=1 - self.decay
        )

        # Laplace smoothing, so a code with a near-zero count does not explode
        # when its accumulated weight is divided by it.
        total = self.ema_cluster_size.sum()
        normalized = (
            (self.ema_cluster_size + self.eps)
            / (total + self.num_codes * self.eps)
            * total
        )
        self.codebook.copy_(self.ema_weight / normalized.unsqueeze(1))

        used = cluster_size > 0
        self.steps_since_used[used] = 0
        self.steps_since_used[~used] += 1
        return self._restart_dead_codes(flat)

    @torch.no_grad()
    def _restart_dead_codes(self, flat: torch.Tensor) -> int:
        if self.dead_code_steps < 1:
            return 0
        dead = (self.steps_since_used >= self.dead_code_steps).nonzero().flatten()
        if dead.numel() == 0:
            return 0

        picks = torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)
        replacements = flat[picks]
        self.codebook[dead] = replacements
        # The EMA state has to move with the code. Resetting only the codebook
        # row leaves the moving average pointing at the dead position, which
        # drags the code straight back there over the next few steps.
        self.ema_weight[dead] = replacements
        self.ema_cluster_size[dead] = 1.0
        self.steps_since_used[dead] = 0
        return int(dead.numel())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_quantizer.py -q`
Expected: PASS, 18 tests.

If `test_a_restarted_code_does_not_drift_back_to_where_it_died` or
`test_unused_codes_are_restarted_from_live_encoder_outputs` fails, do **not** relax the
threshold. It means the EMA state is not being reset alongside the codebook row, which
is the exact bug both tests exist to catch. The whole file above was run against a
prototype of this implementation and all 18 tests passed; both of those two were also
confirmed to fail when the two `ema_*` lines in `_restart_dead_codes` are removed.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/models/quantizer.py tests/test_quantizer.py
git commit -m "feat: add EMA vector quantizer with dead-code restarts"
```

---

## Task 4: VQ-VAE assembly and checkpoints

**Files:**
- Create: `jvqvae/models/vqvae.py`
- Create: `tests/conftest.py`
- Test: `tests/test_vqvae.py`

**Interfaces:**
- Consumes: `Encoder`, `Decoder` (Task 2); `VectorQuantizerEMA`, `QuantizerOutput` (Task 3); `VQVAEConfig`, `RunConfig` (Task 1).
- Produces:
  - `VQVAEOutput`, a dataclass with `reconstruction`, `loss`, `reconstruction_loss`, `commitment_loss`, `perplexity`, `num_active`, `num_restarted`, `codes`.
  - `VQVAE(config: VQVAEConfig, in_channels: int = 1)`, an `nn.Module` with `forward(images) -> VQVAEOutput`, `encode(images) -> codes`, `decode(codes) -> images`.
  - `save_checkpoint(model, path, *, config: VQVAEConfig, epoch: int, metrics: dict) -> None`.
  - `load_checkpoint(path, *, device) -> tuple[VQVAE, dict]` where the dict holds `config`, `epoch` and `metrics`.
- Also produces the shared `tiny_config` fixture every later task uses.

- [ ] **Step 1: Write the shared fixture**

Create `tests/conftest.py`:

```python
import numpy as np
import pytest

from jvqvae.config import DataConfig, RunConfig, TrainConfig, VQVAEConfig
from jvqvae.data.dataset import CharacterData


@pytest.fixture
def tiny_config():
    """A model small enough that a full training step costs milliseconds.

    Every model test uses this. Never load the shipped ETL4 or K49 configs in a
    test: they describe real runs and would make the suite minutes long.
    """
    return RunConfig(
        name="tiny",
        data=DataConfig(source="k49", image_size=16, cache_dir="unused"),
        model=VQVAEConfig(
            num_codes=8,
            embedding_dim=4,
            hidden_channels=8,
            residual_hidden=4,
            num_residual_blocks=1,
            num_downsample=2,
            dead_code_steps=0,
        ),
        train=TrainConfig(epochs=1, batch_size=4, max_steps=2),
    )


@pytest.fixture
def tiny_data():
    """Twelve 16x16 images across three classes, deterministic."""
    rng = np.random.default_rng(0)
    return CharacterData(
        images=rng.random((12, 16, 16), dtype=np.float32),
        class_ids=np.tile(np.arange(3), 4).astype(np.int64),
        writer_ids=np.repeat(np.arange(4), 3).astype(np.int64),
        classes=["あ", "い", "う"],
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_vqvae.py`:

```python
import pytest
import torch

from jvqvae.config import VQVAEConfig
from jvqvae.models.vqvae import VQVAE, load_checkpoint, save_checkpoint


def test_forward_returns_a_reconstruction_of_the_input_shape(tiny_config):
    model = VQVAE(tiny_config.model)
    images = torch.rand(2, 1, 16, 16)

    out = model(images)

    assert out.reconstruction.shape == images.shape
    assert out.codes.shape == (2, 4, 4)


def test_reconstructions_are_valid_images(tiny_config):
    model = VQVAE(tiny_config.model)
    out = model(torch.rand(2, 1, 16, 16))

    assert out.reconstruction.min() >= 0.0
    assert out.reconstruction.max() <= 1.0


def test_the_code_grid_matches_the_configured_latent_size(tiny_config):
    model = VQVAE(tiny_config.model)
    codes = model.encode(torch.rand(3, 1, 16, 16))

    assert codes.shape == (3, tiny_config.latent_size, tiny_config.latent_size)
    assert codes.dtype == torch.int64
    assert int(codes.max()) < tiny_config.model.num_codes


def test_loss_is_the_sum_of_its_reported_parts(tiny_config):
    model = VQVAE(tiny_config.model)
    out = model(torch.rand(4, 1, 16, 16))

    assert out.loss.item() == pytest.approx(
        out.reconstruction_loss.item() + out.commitment_loss.item(), rel=1e-5
    )


def test_backward_reaches_both_the_encoder_and_the_decoder(tiny_config):
    model = VQVAE(tiny_config.model)
    model(torch.rand(4, 1, 16, 16)).loss.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    decoder_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]

    assert encoder_grads and sum(g.abs().sum() for g in encoder_grads) > 0
    assert decoder_grads and sum(g.abs().sum() for g in decoder_grads) > 0


def test_decode_accepts_what_encode_produces(tiny_config):
    model = VQVAE(tiny_config.model).eval()
    images = torch.rand(2, 1, 16, 16)

    assert model.decode(model.encode(images)).shape == images.shape


def test_encode_and_decode_round_trip_through_the_full_forward(tiny_config):
    model = VQVAE(tiny_config.model).eval()
    images = torch.rand(2, 1, 16, 16)

    out = model(images)

    assert torch.allclose(model.decode(out.codes), out.reconstruction, atol=1e-5)


def test_a_few_steps_of_training_drive_the_loss_down(tiny_config):
    """A tiny model on four fixed images must be able to overfit them."""
    torch.manual_seed(0)
    model = VQVAE(tiny_config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    images = torch.rand(4, 1, 16, 16)

    first = model(images).reconstruction_loss.item()
    for _ in range(30):
        optimizer.zero_grad()
        out = model(images)
        out.loss.backward()
        optimizer.step()

    assert out.reconstruction_loss.item() < first


def test_checkpoint_round_trip_reproduces_identical_outputs(tiny_config, tmp_path):
    torch.manual_seed(0)
    model = VQVAE(tiny_config.model).eval()
    images = torch.rand(2, 1, 16, 16)
    expected = model(images).reconstruction

    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        model, path, config=tiny_config.model, epoch=7, metrics={"val_loss": 0.5}
    )
    restored, meta = load_checkpoint(path, device=torch.device("cpu"))

    assert torch.allclose(restored.eval()(images).reconstruction, expected, atol=1e-6)
    assert meta["epoch"] == 7
    assert meta["metrics"]["val_loss"] == 0.5
    assert meta["config"] == tiny_config.model


def test_the_checkpoint_carries_the_codebook(tiny_config, tmp_path):
    model = VQVAE(tiny_config.model).train()
    model(torch.rand(8, 1, 16, 16))
    path = tmp_path / "vqvae.pt"
    save_checkpoint(model, path, config=tiny_config.model, epoch=0, metrics={})

    restored, _ = load_checkpoint(path, device=torch.device("cpu"))

    assert torch.equal(restored.quantizer.codebook, model.quantizer.codebook)


def test_a_checkpoint_rebuilds_the_model_without_being_handed_a_config(
    tiny_config, tmp_path
):
    """Plan 3 loads a checkpoint knowing only its path."""
    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        VQVAE(tiny_config.model), path, config=tiny_config.model, epoch=0, metrics={}
    )

    restored, meta = load_checkpoint(path, device=torch.device("cpu"))

    assert isinstance(meta["config"], VQVAEConfig)
    assert restored.config.num_codes == tiny_config.model.num_codes


def test_a_loaded_checkpoint_comes_back_in_eval_mode(tiny_config, tmp_path):
    """Otherwise a fresh module defaults to training and keeps moving the codebook."""
    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        VQVAE(tiny_config.model).train(),
        path,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )

    model, _ = load_checkpoint(path, device=torch.device("cpu"))

    assert not model.training
    before = model.quantizer.codebook.clone()
    model(torch.rand(8, 1, 16, 16))
    assert torch.equal(model.quantizer.codebook, before)


def test_loading_a_checkpoint_does_not_unpickle_arbitrary_objects(
    tiny_config, tmp_path
):
    path = tmp_path / "vqvae.pt"
    save_checkpoint(
        VQVAE(tiny_config.model), path, config=tiny_config.model, epoch=0, metrics={}
    )

    # weights_only=True must be enough to read it back.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"model", "config", "epoch", "metrics"}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_vqvae.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.models.vqvae'`.

- [ ] **Step 4: Implement the model**

Create `jvqvae/models/vqvae.py`:

```python
"""The VQ-VAE itself: encoder, quantizer, decoder, and its checkpoint format.

A VQ-VAE is not generative on its own. It learns a discrete code for an image
and how to decode one, but it holds no distribution over code grids, so a grid
sampled uniformly decodes to noise. Learning that distribution is the prior's
job, in plan 3; this model's output is the code grids it trains on.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from jvqvae.config import VQVAEConfig
from jvqvae.models.blocks import Decoder, Encoder
from jvqvae.models.quantizer import VectorQuantizerEMA


@dataclass
class VQVAEOutput:
    reconstruction: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor
    num_active: int
    num_restarted: int
    codes: torch.Tensor


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig, in_channels: int = 1) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels

        shared = dict(
            num_downsample=config.num_downsample,
            num_residual_blocks=config.num_residual_blocks,
            residual_hidden=config.residual_hidden,
        )
        self.encoder = Encoder(
            in_channels,
            hidden_channels=config.hidden_channels,
            embedding_dim=config.embedding_dim,
            **shared,
        )
        self.quantizer = VectorQuantizerEMA(
            config.num_codes,
            config.embedding_dim,
            commitment_cost=config.commitment_cost,
            decay=config.ema_decay,
            dead_code_steps=config.dead_code_steps,
        )
        self.decoder = Decoder(
            config.embedding_dim,
            hidden_channels=config.hidden_channels,
            out_channels=in_channels,
            **shared,
        )

    def forward(self, images: torch.Tensor) -> VQVAEOutput:
        quantized = self.quantizer(self.encoder(images))
        reconstruction = self.decoder(quantized.quantized)
        reconstruction_loss = F.mse_loss(reconstruction, images)

        return VQVAEOutput(
            reconstruction=reconstruction,
            loss=reconstruction_loss + quantized.commitment_loss,
            reconstruction_loss=reconstruction_loss,
            commitment_loss=quantized.commitment_loss,
            perplexity=quantized.perplexity,
            num_active=quantized.num_active,
            num_restarted=quantized.num_restarted,
            codes=quantized.codes,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return the grid of code indices for a batch of images."""
        return self.quantizer(self.encoder(images)).codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from a grid of code indices."""
        return self.decoder(self.quantizer.lookup(codes))


def save_checkpoint(
    model: VQVAE,
    path: str | Path,
    *,
    config: VQVAEConfig,
    epoch: int,
    metrics: dict,
) -> None:
    """Write a checkpoint that can rebuild the model from the file alone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        # Stored as a plain dict so the file reads back with weights_only=True.
        "config": asdict(config),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[VQVAE, dict]:
    """Rebuild a model and its metadata from a checkpoint.

    The model comes back in eval mode. Training mode is not part of a
    `state_dict`, so a freshly built module would otherwise default to
    training and quietly keep updating the codebook on whoever used it.
    """
    raw = torch.load(path, map_location=device, weights_only=True)
    config = VQVAEConfig(**raw["config"])

    model = VQVAE(config)
    model.load_state_dict(raw["model"])
    model.to(device)
    model.eval()

    return model, {
        "config": config,
        "epoch": raw["epoch"],
        "metrics": raw["metrics"],
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_vqvae.py -q`
Expected: PASS, 13 tests.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS. The new `tests/conftest.py` fixtures must not disturb the plan 1 tests.

- [ ] **Step 7: Commit**

```bash
git add jvqvae/models/vqvae.py tests/test_vqvae.py tests/conftest.py
git commit -m "feat: assemble the VQ-VAE with a self-describing checkpoint format"
```

---

## Task 5: Training loop

**Files:**
- Create: `jvqvae/train/__init__.py`
- Create: `jvqvae/train/seed.py`
- Create: `jvqvae/train/vqvae.py`
- Test: `tests/test_train_vqvae.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 1); `VQVAE`, `save_checkpoint` (Task 4); `CharacterData` (plan 1); `CharacterTorchDataset` (plan 1).
- Produces:
  - `set_seed(seed: int) -> None`.
  - `EpochMetrics`, a dataclass with `epoch`, `train_loss`, `train_reconstruction`, `train_commitment`, `val_loss`, `perplexity`, `num_active`, `num_restarted`.
  - `train_vqvae(config, splits, *, device, output_root=".", log=print) -> dict` returning the run record it also writes to disk.
  - Artifacts: `<checkpoint_dir>/<name>/vqvae_best.pt`, `vqvae_last.pt`, and `<results_dir>/<name>/vqvae_run.json`.

`train_vqvae` takes `splits` rather than loading them, so a test can hand it four
synthetic images and the CLI can hand it the real pipeline output.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_vqvae.py`:

```python
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from jvqvae.data.dataset import CharacterData
from jvqvae.models.vqvae import load_checkpoint
from jvqvae.train.seed import set_seed
from jvqvae.train.vqvae import train_vqvae

CPU = torch.device("cpu")


@pytest.fixture
def splits(tiny_data):
    return {"train": tiny_data, "val": tiny_data.subset(np.arange(4))}


def run(config, splits, tmp_path, **overrides):
    if overrides:
        config = replace(config, train=replace(config.train, **overrides))
    return train_vqvae(
        config, splits, device=CPU, output_root=tmp_path, log=lambda *a: None
    )


def test_set_seed_makes_torch_reproducible():
    set_seed(3)
    first = torch.randn(4)
    set_seed(3)
    assert torch.equal(torch.randn(4), first)


def test_set_seed_makes_numpy_reproducible():
    set_seed(3)
    first = np.random.rand(4)
    set_seed(3)
    assert np.array_equal(np.random.rand(4), first)


def test_training_returns_one_record_per_epoch(tiny_config, splits, tmp_path):
    record = run(tiny_config, splits, tmp_path, epochs=3, max_steps=None)

    assert len(record["epochs"]) == 3
    assert [e["epoch"] for e in record["epochs"]] == [0, 1, 2]


def test_max_steps_caps_the_run(tiny_config, splits, tmp_path):
    record = run(tiny_config, splits, tmp_path, epochs=100, max_steps=3)

    assert record["total_steps"] == 3
    assert len(record["epochs"]) < 100


def test_every_epoch_reports_the_metrics_the_blog_post_needs(
    tiny_config, splits, tmp_path
):
    record = run(tiny_config, splits, tmp_path, epochs=1, max_steps=None)
    epoch = record["epochs"][0]

    for key in (
        "train_loss",
        "train_reconstruction",
        "train_commitment",
        "val_loss",
        "perplexity",
        "num_active",
        "num_restarted",
    ):
        assert key in epoch, key
    assert epoch["num_active"] >= 1


def test_training_reduces_the_loss(tiny_config, splits, tmp_path):
    record = run(
        tiny_config, splits, tmp_path, epochs=25, max_steps=None, learning_rate=1e-2
    )
    losses = [e["train_loss"] for e in record["epochs"]]

    assert losses[-1] < losses[0]


def test_both_checkpoints_are_written(tiny_config, splits, tmp_path):
    run(tiny_config, splits, tmp_path, epochs=2, max_steps=None)
    directory = tmp_path / "checkpoints" / "tiny"

    assert (directory / "vqvae_best.pt").exists()
    assert (directory / "vqvae_last.pt").exists()


def test_the_best_checkpoint_is_loadable_and_knows_its_epoch(
    tiny_config, splits, tmp_path
):
    run(tiny_config, splits, tmp_path, epochs=2, max_steps=None)

    model, meta = load_checkpoint(
        tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt", device=CPU
    )

    assert meta["config"].num_codes == tiny_config.model.num_codes
    assert 0 <= meta["epoch"] <= 1
    assert model.eval()(torch.rand(1, 1, 16, 16)).reconstruction.shape == (1, 1, 16, 16)


def test_the_best_checkpoint_tracks_the_lowest_validation_loss(
    tiny_config, splits, tmp_path
):
    record = run(tiny_config, splits, tmp_path, epochs=4, max_steps=None)

    best = min(record["epochs"], key=lambda e: e["val_loss"])
    assert record["best_epoch"] == best["epoch"]
    assert record["best_val_loss"] == pytest.approx(best["val_loss"])

    _, meta = load_checkpoint(
        tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt", device=CPU
    )
    assert meta["epoch"] == best["epoch"]


def test_the_run_record_is_written_as_json(tiny_config, splits, tmp_path):
    returned = run(tiny_config, splits, tmp_path, epochs=1, max_steps=None)
    path = tmp_path / "results" / "tiny" / "vqvae_run.json"

    assert path.exists()
    assert json.loads(path.read_text()) == returned


def test_the_run_record_carries_enough_to_reproduce_the_run(
    tiny_config, splits, tmp_path
):
    record = run(tiny_config, splits, tmp_path, epochs=1, max_steps=None)

    assert record["name"] == "tiny"
    assert record["device"] == "cpu"
    assert record["config"]["model"]["num_codes"] == tiny_config.model.num_codes
    assert record["config"]["data"]["source"] == "k49"
    assert record["config"]["train"]["seed"] == tiny_config.train.seed
    assert record["num_train"] == 12
    assert record["num_val"] == 4
    assert "torch_version" in record


def test_the_same_seed_gives_the_same_losses(tiny_config, splits, tmp_path):
    first = run(tiny_config, splits, tmp_path / "a", epochs=2, max_steps=None)
    second = run(tiny_config, splits, tmp_path / "b", epochs=2, max_steps=None)

    assert [e["train_loss"] for e in first["epochs"]] == [
        e["train_loss"] for e in second["epochs"]
    ]


def test_different_seeds_give_different_losses(tiny_config, splits, tmp_path):
    first = run(tiny_config, splits, tmp_path / "a", epochs=2, max_steps=None, seed=0)
    second = run(tiny_config, splits, tmp_path / "b", epochs=2, max_steps=None, seed=1)

    assert [e["train_loss"] for e in first["epochs"]] != [
        e["train_loss"] for e in second["epochs"]
    ]


def test_training_without_a_validation_split_still_works(
    tiny_config, tiny_data, tmp_path
):
    record = train_vqvae(
        tiny_config,
        {"train": tiny_data},
        device=CPU,
        output_root=tmp_path,
        log=lambda *a: None,
    )

    assert record["epochs"][0]["val_loss"] is None
    assert record["best_val_loss"] is None
    # With nothing to select on, "best" is simply the final state.
    assert (tmp_path / "checkpoints" / "tiny" / "vqvae_best.pt").exists()


def test_an_empty_training_split_is_refused(tiny_config, tiny_data, tmp_path):
    empty = tiny_data.subset(np.arange(0))
    with pytest.raises(ValueError, match="empty"):
        train_vqvae(
            tiny_config,
            {"train": empty},
            device=CPU,
            output_root=tmp_path,
            log=lambda *a: None,
        )


def test_an_image_size_disagreeing_with_the_config_is_refused(
    tiny_config, tmp_path
):
    wrong = CharacterData(
        images=np.zeros((4, 32, 32), dtype=np.float32),
        class_ids=np.zeros(4, dtype=np.int64),
        writer_ids=None,
        classes=["あ"],
    )
    with pytest.raises(ValueError, match="image_size"):
        train_vqvae(
            tiny_config,
            {"train": wrong},
            device=CPU,
            output_root=tmp_path,
            log=lambda *a: None,
        )


def test_the_log_callback_sees_every_epoch(tiny_config, splits, tmp_path):
    lines = []
    config = replace(tiny_config, train=replace(tiny_config.train, epochs=3,
                                                max_steps=None))

    train_vqvae(config, splits, device=CPU, output_root=tmp_path, log=lines.append)

    assert len(lines) == 3
    assert "perplexity" in lines[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train_vqvae.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.train'`.

- [ ] **Step 3: Create the package marker and the seeding helper**

Create `jvqvae/train/__init__.py` as an empty file.

Create `jvqvae/train/seed.py`:

```python
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
```

- [ ] **Step 4: Implement the training loop**

Create `jvqvae/train/vqvae.py`:

```python
"""VQ-VAE training.

Writes two checkpoints (best by validation loss, and last) plus a JSON run
record. The record exists from phase 1 even though no evaluation metrics are
computed yet, so phase 2 has somewhere to put them.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from jvqvae.config import RunConfig
from jvqvae.data.dataset import CharacterData
from jvqvae.data.torch_dataset import CharacterTorchDataset
from jvqvae.models.vqvae import VQVAE, save_checkpoint
from jvqvae.train.seed import set_seed


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_reconstruction: float
    train_commitment: float
    val_loss: float | None
    perplexity: float
    num_active: int
    num_restarted: int


def _check(config: RunConfig, splits: dict[str, CharacterData]) -> None:
    train = splits.get("train")
    if train is None or len(train) == 0:
        raise ValueError("the training split is empty, so there is nothing to train on")

    height, width = train.image_shape
    if (height, width) != (config.data.image_size, config.data.image_size):
        raise ValueError(
            f"config image_size is {config.data.image_size} but the data is "
            f"{height}x{width}; they must agree"
        )


def _loader(data: CharacterData, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        CharacterTorchDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def _validate(model: VQVAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, _ in loader:
        images = images.to(device)
        total += model(images).reconstruction_loss.item() * len(images)
        count += len(images)
    return total / max(count, 1)


def train_vqvae(
    config: RunConfig,
    splits: dict[str, CharacterData],
    *,
    device: torch.device,
    output_root: str | Path = ".",
    log: Callable[[str], None] = print,
) -> dict:
    """Train the VQ-VAE and write its checkpoints and run record."""
    _check(config, splits)
    set_seed(config.train.seed)

    root = Path(output_root)
    checkpoint_dir = root / config.train.checkpoint_dir / config.name
    results_dir = root / config.train.results_dir / config.name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = VQVAE(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

    train_loader = _loader(splits["train"], config.train.batch_size, shuffle=True)
    val = splits.get("val")
    val_loader = (
        _loader(val, config.train.batch_size, shuffle=False)
        if val is not None and len(val)
        else None
    )

    history: list[EpochMetrics] = []
    best_val, best_epoch = None, None
    steps = 0

    for epoch in range(config.train.epochs):
        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "commit": 0.0, "perplexity": 0.0}
        active, restarted, batches = 0, 0, 0

        for images, _ in train_loader:
            if config.train.max_steps is not None and steps >= config.train.max_steps:
                break

            images = images.to(device)
            optimizer.zero_grad()
            out = model(images)
            out.loss.backward()
            optimizer.step()
            steps += 1
            batches += 1

            sums["loss"] += out.loss.item()
            sums["recon"] += out.reconstruction_loss.item()
            sums["commit"] += out.commitment_loss.item()
            sums["perplexity"] += out.perplexity.item()
            active = max(active, out.num_active)
            restarted += out.num_restarted

        if batches == 0:
            break

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=sums["loss"] / batches,
            train_reconstruction=sums["recon"] / batches,
            train_commitment=sums["commit"] / batches,
            val_loss=_validate(model, val_loader, device) if val_loader else None,
            perplexity=sums["perplexity"] / batches,
            num_active=active,
            num_restarted=restarted,
        )
        history.append(metrics)

        log(
            f"epoch {epoch}: train {metrics.train_loss:.4f}"
            + (f" val {metrics.val_loss:.4f}" if metrics.val_loss is not None else "")
            + f" perplexity {metrics.perplexity:.1f}"
            f" active {metrics.num_active}/{config.model.num_codes}"
            + (f" restarted {metrics.num_restarted}" if metrics.num_restarted else "")
        )

        if metrics.val_loss is not None and (
            best_val is None or metrics.val_loss < best_val
        ):
            best_val, best_epoch = metrics.val_loss, epoch
            save_checkpoint(
                model,
                checkpoint_dir / "vqvae_best.pt",
                config=config.model,
                epoch=epoch,
                metrics=asdict(metrics),
            )

        if config.train.max_steps is not None and steps >= config.train.max_steps:
            break

    # Leave the model in eval mode: whatever runs next should not keep mutating
    # the codebook by accident.
    model.eval()
    last_epoch = history[-1].epoch if history else -1
    save_checkpoint(
        model,
        checkpoint_dir / "vqvae_last.pt",
        config=config.model,
        epoch=last_epoch,
        metrics=asdict(history[-1]) if history else {},
    )
    if best_epoch is None:
        # No validation split, so "best" is simply the final state.
        save_checkpoint(
            model,
            checkpoint_dir / "vqvae_best.pt",
            config=config.model,
            epoch=last_epoch,
            metrics=asdict(history[-1]) if history else {},
        )

    record = {
        "name": config.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "device": device.type,
        "torch_version": torch.__version__,
        "config": {
            "data": asdict(config.data),
            "model": asdict(config.model),
            "train": asdict(config.train),
        },
        "num_train": len(splits["train"]),
        "num_val": len(val) if val is not None else 0,
        "total_steps": steps,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs": [asdict(m) for m in history],
    }
    (results_dir / "vqvae_run.json").write_text(json.dumps(record, indent=2))
    return record
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train_vqvae.py -q`
Expected: PASS, 16 tests.

- [ ] **Step 6: Check the suite's runtime**

Run: `uv run pytest -q --durations=5`
Expected: PASS, and no single test above about 5 seconds. If one is slower, reduce its
epoch count rather than accepting it.

- [ ] **Step 7: Commit**

```bash
git add jvqvae/train tests/test_train_vqvae.py
git commit -m "feat: add VQ-VAE training with checkpoints and a JSON run record"
```

---

## Task 6: Code extraction and the reconstruction figure

**Files:**
- Create: `jvqvae/codes.py`
- Create: `jvqvae/figures.py`
- Test: `tests/test_codes.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `VQVAE` (Task 4); `CharacterData` (plan 1).
- Produces:
  - `CodeData(codes, class_ids, classes, num_codes)`, frozen, where `codes` is int64 `(N, G, G)`, with properties `latent_size`, `num_classes` and `sequence_length`.
  - `extract_codes(model, data, *, device, batch_size=64) -> CodeData`.
  - `save_codes(data: CodeData, path) -> None` and `load_codes(path) -> CodeData`.
  - `save_reconstruction_grid(model, data, path, *, device, num_images=8, seed=0) -> Path`.

`CodeData` is the hand-off to plan 3: the prior trains on exactly this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_codes.py`:

```python
import numpy as np
import pytest
import torch

from jvqvae.codes import CodeData, extract_codes, load_codes, save_codes
from jvqvae.models.vqvae import VQVAE

CPU = torch.device("cpu")


def test_extraction_produces_one_grid_per_image(tiny_config, tiny_data):
    model = VQVAE(tiny_config.model)

    codes = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert codes.codes.shape == (12, 4, 4)
    assert codes.codes.dtype == np.int64
    assert codes.latent_size == 4


def test_extracted_codes_stay_inside_the_codebook(tiny_config, tiny_data):
    model = VQVAE(tiny_config.model)

    codes = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert codes.codes.min() >= 0
    assert codes.codes.max() < tiny_config.model.num_codes
    assert codes.num_codes == tiny_config.model.num_codes


def test_labels_survive_extraction_in_order(tiny_config, tiny_data):
    model = VQVAE(tiny_config.model)

    codes = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert np.array_equal(codes.class_ids, tiny_data.class_ids)
    assert codes.classes == tiny_data.classes


def test_batch_size_does_not_change_the_result(tiny_config, tiny_data):
    """A ragged final batch is where an off-by-one would show up."""
    model = VQVAE(tiny_config.model).eval()

    one = extract_codes(model, tiny_data, device=CPU, batch_size=12)
    many = extract_codes(model, tiny_data, device=CPU, batch_size=5)

    assert np.array_equal(one.codes, many.codes)


def test_extraction_leaves_the_codebook_untouched(tiny_config, tiny_data):
    """Extraction must not keep training the quantizer."""
    model = VQVAE(tiny_config.model).train()
    before = model.quantizer.codebook.clone()

    extract_codes(model, tiny_data, device=CPU)

    assert torch.equal(model.quantizer.codebook, before)


def test_codes_round_trip_through_a_file(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    original = extract_codes(model, tiny_data, device=CPU)

    path = tmp_path / "codes.npz"
    save_codes(original, path)
    restored = load_codes(path)

    assert np.array_equal(restored.codes, original.codes)
    assert np.array_equal(restored.class_ids, original.class_ids)
    assert restored.classes == original.classes
    assert restored.num_codes == original.num_codes
    assert restored.latent_size == original.latent_size


def test_the_saved_file_does_not_need_pickle(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    path = tmp_path / "codes.npz"
    save_codes(extract_codes(model, tiny_data, device=CPU), path)

    with np.load(path, allow_pickle=False) as raw:
        assert "codes" in raw


def test_code_data_rejects_a_mismatched_length():
    with pytest.raises(ValueError, match="length"):
        CodeData(
            codes=np.zeros((3, 4, 4), dtype=np.int64),
            class_ids=np.zeros(2, dtype=np.int64),
            classes=["あ"],
            num_codes=8,
        )


def test_code_data_rejects_codes_outside_the_codebook():
    with pytest.raises(ValueError, match="codebook"):
        CodeData(
            codes=np.full((2, 4, 4), 99, dtype=np.int64),
            class_ids=np.zeros(2, dtype=np.int64),
            classes=["あ"],
            num_codes=8,
        )


def test_code_data_rejects_a_non_square_grid():
    with pytest.raises(ValueError, match="square"):
        CodeData(
            codes=np.zeros((2, 4, 5), dtype=np.int64),
            class_ids=np.zeros(2, dtype=np.int64),
            classes=["あ"],
            num_codes=8,
        )
```

Create `tests/test_figures.py`:

```python
import torch

from jvqvae.figures import save_reconstruction_grid
from jvqvae.models.vqvae import VQVAE

CPU = torch.device("cpu")


def test_the_figure_is_written(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    path = tmp_path / "nested" / "recon.png"

    returned = save_reconstruction_grid(
        model, tiny_data, path, device=CPU, num_images=4
    )

    assert returned == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_asking_for_more_images_than_exist_is_clamped(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model)
    path = tmp_path / "recon.png"

    save_reconstruction_grid(model, tiny_data, path, device=CPU, num_images=500)

    assert path.exists()


def test_the_figure_does_not_leave_the_model_training(tiny_config, tiny_data, tmp_path):
    model = VQVAE(tiny_config.model).train()
    before = model.quantizer.codebook.clone()

    save_reconstruction_grid(
        model, tiny_data, tmp_path / "recon.png", device=CPU, num_images=4
    )

    assert torch.equal(model.quantizer.codebook, before)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_codes.py tests/test_figures.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jvqvae.codes'`.

- [ ] **Step 3: Implement code extraction**

Create `jvqvae/codes.py`:

```python
"""Code grids extracted from a trained VQ-VAE.

This is the hand-off to the prior: the prior never sees an image, only these
grids and their class labels.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from jvqvae.data.dataset import CharacterData
from jvqvae.models.vqvae import VQVAE


@dataclass(frozen=True)
class CodeData:
    codes: np.ndarray  # int64 (N, G, G)
    class_ids: np.ndarray  # int64 (N,)
    classes: list[str]
    num_codes: int

    def __post_init__(self) -> None:
        if self.codes.ndim != 3:
            raise ValueError(f"codes must be (N, G, G), got {self.codes.shape}")
        if self.codes.shape[1] != self.codes.shape[2]:
            raise ValueError(f"code grid must be square, got {self.codes.shape[1:]}")
        if len(self.class_ids) != len(self.codes):
            raise ValueError(
                f"length mismatch: {len(self.codes)} grids, "
                f"{len(self.class_ids)} class ids"
            )
        if len(self.codes) and (
            int(self.codes.max()) >= self.num_codes or int(self.codes.min()) < 0
        ):
            raise ValueError(
                f"codes must index a codebook of {self.num_codes}, got "
                f"[{int(self.codes.min())}, {int(self.codes.max())}]"
            )
        if len(self.class_ids) and int(self.class_ids.max()) >= len(self.classes):
            raise ValueError(
                f"class id {int(self.class_ids.max())} exceeds class table of "
                f"size {len(self.classes)}"
            )

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def latent_size(self) -> int:
        return self.codes.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def sequence_length(self) -> int:
        """Tokens per image once the grid is flattened in raster order."""
        return self.codes.shape[1] * self.codes.shape[2]


@torch.no_grad()
def extract_codes(
    model: VQVAE,
    data: CharacterData,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> CodeData:
    """Run a trained model over a dataset and keep only the code grids."""
    was_training = model.training
    model.eval().to(device)

    grids = []
    for start in range(0, len(data), batch_size):
        images = data.images[start : start + batch_size]
        tensor = torch.from_numpy(images).unsqueeze(1).to(device)
        grids.append(model.encode(tensor).cpu().numpy())

    if was_training:
        model.train()

    codes = (
        np.concatenate(grids).astype(np.int64)
        if grids
        else np.zeros((0, 0, 0), dtype=np.int64)
    )
    return CodeData(
        codes=codes,
        class_ids=data.class_ids.astype(np.int64),
        classes=list(data.classes),
        num_codes=model.config.num_codes,
    )


def save_codes(data: CodeData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        codes=data.codes,
        class_ids=data.class_ids,
        classes=np.array(data.classes),
        num_codes=np.array(data.num_codes),
    )


def load_codes(path: str | Path) -> CodeData:
    with np.load(path, allow_pickle=False) as raw:
        return CodeData(
            codes=raw["codes"],
            class_ids=raw["class_ids"],
            classes=[str(char) for char in raw["classes"]],
            num_codes=int(raw["num_codes"]),
        )
```

- [ ] **Step 4: Implement the figure**

Create `jvqvae/figures.py`:

```python
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
    picks = np.random.default_rng(seed).choice(len(data), count, replace=False)
    picks.sort()

    originals = data.images[picks]
    tensor = torch.from_numpy(originals).unsqueeze(1).to(device)
    reconstructions = model(tensor).reconstruction.squeeze(1).cpu().numpy()

    if was_training:
        model.train()

    figure, axes = plt.subplots(2, count, figsize=(count * 1.2, 2.8), squeeze=False)
    for column in range(count):
        for row, image in enumerate((originals[column], reconstructions[column])):
            axis = axes[row][column]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
        label = data.classes[int(data.class_ids[picks[column]])]
        axes[0][column].set_title(label, fontsize=9)

    axes[0][0].set_ylabel("original", fontsize=8)
    axes[1][0].set_ylabel("decoded", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_codes.py tests/test_figures.py -q`
Expected: PASS, 13 tests.

A missing CJK glyph may print a matplotlib font warning for the title. If it does, that
is cosmetic for a test fixture, but confirm it does not appear as a warning in the suite
summary; if it does, drop the `set_title` call rather than suppressing warnings
globally.

- [ ] **Step 6: Commit**

```bash
git add jvqvae/codes.py jvqvae/figures.py tests/test_codes.py tests/test_figures.py
git commit -m "feat: extract code grids and render reconstruction figures"
```

---

## Task 7: CLI, README and deferred cleanup

**Files:**
- Modify: `jvqvae/cli.py`
- Modify: `README.md`
- Modify: `jvqvae/data/dataset.py`
- Modify: `jvqvae/data/torch_dataset.py`
- Modify: `jvqvae/data/sources/k49.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_torch_dataset.py`
- Modify: `tests/test_etl4.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `jvqvae train-vqvae --config PATH [--device D] [--max-steps N] [--output-root DIR]` and `jvqvae extract-codes --config PATH [--checkpoint PATH] [--split S] [--out PATH]`.

The second half of this task clears deferred findings recorded during plan 1, listed in
`.superpowers/sdd/progress.md`. They are small and they touch the containers the models
now depend on.

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_cli.py`:

```python
def test_train_vqvae_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["train-vqvae", "--help"])
    out = capsys.readouterr().out

    assert excinfo.value.code == 0
    for flag in ("--config", "--device", "--max-steps", "--output-root"):
        assert flag in out


def test_extract_codes_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["extract-codes", "--help"])
    out = capsys.readouterr().out

    assert excinfo.value.code == 0
    for flag in ("--config", "--checkpoint", "--split", "--out"):
        assert flag in out


def test_train_vqvae_runs_end_to_end(monkeypatch, tmp_path, tiny_config, tiny_data):
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    monkeypatch.setattr(
        "jvqvae.cli.load_splits", lambda config: {"train": tiny_data, "val": tiny_data}
    )

    code = main(
        [
            "train-vqvae",
            "--config",
            "unused.yaml",
            "--device",
            "cpu",
            "--max-steps",
            "2",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert (tmp_path / "checkpoints" / "tiny" / "vqvae_last.pt").exists()
    assert (tmp_path / "results" / "tiny" / "vqvae_run.json").exists()
    assert (tmp_path / "results" / "tiny" / "reconstructions.png").exists()


def test_extract_codes_writes_a_loadable_file(
    monkeypatch, tmp_path, tiny_config, tiny_data
):
    from jvqvae.codes import load_codes
    from jvqvae.models.vqvae import VQVAE, save_checkpoint

    checkpoint = tmp_path / "vqvae_best.pt"
    save_checkpoint(
        VQVAE(tiny_config.model),
        checkpoint,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    monkeypatch.setattr("jvqvae.cli.load_splits", lambda config: {"train": tiny_data})

    out = tmp_path / "codes.npz"
    code = main(
        [
            "extract-codes",
            "--config",
            "unused.yaml",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "train",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    assert load_codes(out).codes.shape == (12, 4, 4)


def test_extract_codes_rejects_an_unknown_split(
    monkeypatch, tmp_path, tiny_config, tiny_data
):
    from jvqvae.models.vqvae import VQVAE, save_checkpoint

    checkpoint = tmp_path / "vqvae_best.pt"
    save_checkpoint(
        VQVAE(tiny_config.model),
        checkpoint,
        config=tiny_config.model,
        epoch=0,
        metrics={},
    )
    monkeypatch.setattr("jvqvae.cli.load_run_config", lambda path: tiny_config)
    monkeypatch.setattr("jvqvae.cli.load_splits", lambda config: {"train": tiny_data})

    with pytest.raises(SystemExit, match="train"):
        main(
            [
                "extract-codes",
                "--config",
                "unused.yaml",
                "--checkpoint",
                str(checkpoint),
                "--split",
                "nonsense",
                "--out",
                str(tmp_path / "codes.npz"),
            ]
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -x -q`
Expected: FAIL — argparse exits with code 2 on the unknown command `train-vqvae`, so the
help assertion fails.

- [ ] **Step 3: Add the two subcommands**

Replace the imports at the top of `jvqvae/cli.py` with:

```python
"""Command line entry points for data preparation and training."""

import argparse
from dataclasses import replace
from pathlib import Path

from jvqvae.codes import extract_codes, save_codes
from jvqvae.config import load_run_config
from jvqvae.data.pipeline import load_splits
from jvqvae.data.sources.etl4 import ingest_etl4
from jvqvae.data.sources.k49 import download_k49, load_k49
from jvqvae.device import select_device
from jvqvae.figures import save_reconstruction_grid
from jvqvae.models.vqvae import load_checkpoint
from jvqvae.train.vqvae import train_vqvae
```

Add these two handlers after `_ingest_etl4`:

```python
def _train_vqvae(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    if args.max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=args.max_steps))

    device = select_device(args.device or config.train.device)
    print(f"training {config.name} on {device}")

    splits = load_splits(config.data)
    record = train_vqvae(
        config, splits, device=device, output_root=args.output_root
    )

    results = Path(args.output_root) / config.train.results_dir / config.name
    model, _ = load_checkpoint(
        Path(args.output_root)
        / config.train.checkpoint_dir
        / config.name
        / "vqvae_best.pt",
        device=device,
    )
    figure = save_reconstruction_grid(
        model,
        splits.get("val") or splits["train"],
        results / "reconstructions.png",
        device=device,
    )

    print(
        f"done after {record['total_steps']} steps; "
        f"best val {record['best_val_loss']}; figure at {figure}"
    )
    return 0


def _extract_codes(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    device = select_device(args.device or config.train.device)

    checkpoint = Path(
        args.checkpoint
        or Path(args.output_root)
        / config.train.checkpoint_dir
        / config.name
        / "vqvae_best.pt"
    )
    if not checkpoint.exists():
        raise SystemExit(
            f"{checkpoint} does not exist. Run `jvqvae train-vqvae --config "
            f"{args.config}` first."
        )

    model, _ = load_checkpoint(checkpoint, device=device)
    splits = load_splits(config.data)
    if args.split not in splits:
        raise SystemExit(
            f"unknown split {args.split!r}; this config has: "
            f"{', '.join(sorted(splits))}"
        )

    codes = extract_codes(model, splits[args.split], device=device)
    out = Path(
        args.out
        or Path(args.output_root)
        / config.train.results_dir
        / config.name
        / f"codes_{args.split}.npz"
    )
    save_codes(codes, out)
    print(
        f"wrote {len(codes)} code grids of "
        f"{codes.latent_size}x{codes.latent_size} to {out}"
    )
    return 0
```

Then register both subparsers in `main`, immediately before
`args = parser.parse_args(argv)`:

```python
    train = subparsers.add_parser("train-vqvae", help="train the VQ-VAE")
    train.add_argument("--config", default="configs/etl4.yaml")
    train.add_argument("--device", default=None, help="cpu, mps, cuda or auto")
    train.add_argument("--max-steps", type=int, default=None, help="cap for a smoke run")
    train.add_argument("--output-root", default=".")
    train.set_defaults(handler=_train_vqvae)

    extract = subparsers.add_parser(
        "extract-codes", help="write the code grids the prior trains on"
    )
    extract.add_argument("--config", default="configs/etl4.yaml")
    extract.add_argument("--checkpoint", default=None)
    extract.add_argument("--split", default="train")
    extract.add_argument("--out", default=None)
    extract.add_argument("--device", default=None)
    extract.add_argument("--output-root", default=".")
    extract.set_defaults(handler=_extract_codes)
```

- [ ] **Step 4: Run the CLI tests**

Run: `uv run pytest tests/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Clear the deferred findings**

In `jvqvae/data/dataset.py`, extend `__post_init__`. Insert after the `images.dtype`
check:

```python
        for name, values in (
            ("class_ids", self.class_ids),
            ("writer_ids", self.writer_ids),
        ):
            if values is None:
                continue
            if values.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got {values.shape}")
            if values.dtype != np.int64:
                raise ValueError(f"{name} must be int64, got {values.dtype}")
```

and replace the final class-id bounds check with one that catches negatives too:

```python
        if len(self.class_ids):
            if int(self.class_ids.min()) < 0:
                raise ValueError(
                    f"class ids must not be negative, got {int(self.class_ids.min())}"
                )
            if int(self.class_ids.max()) >= len(self.classes):
                raise ValueError(
                    f"class id {int(self.class_ids.max())} exceeds class table of "
                    f"size {len(self.classes)}"
                )
```

In `jvqvae/data/torch_dataset.py`, validate indices in `__init__`:

```python
    def __init__(self, data: CharacterData, indices: np.ndarray | None = None) -> None:
        self._data = data
        if indices is None:
            indices = np.arange(len(data), dtype=np.int64)
        elif len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(data)):
            raise ValueError(
                f"indices must lie in [0, {len(data)}), got "
                f"[{int(indices.min())}, {int(indices.max())}]"
            )
        self._indices = indices
```

In `jvqvae/data/sources/k49.py`, give `load_k49` the docstring it is missing, directly
under its signature:

```python
    """Load one official K49 split into a `CharacterData`.

    K49 has no writer identities, so `writer_ids` is always None.
    """
```

In `tests/test_etl4.py`, replace `test_expected_fixity_constants_match_aist_listing`
with a comment on the constants instead. Delete the test, and in
`jvqvae/data/sources/etl4.py` change the constant block to:

```python
# From AIST's download listing for ETL4. Re-confirm against the real archive the
# first time ingestion runs against it.
EXPECTED_ZIP_BYTES = 4_949_352
EXPECTED_ZIP_MD5 = "cede897313b3617e364ddb17b8fad499"
```

- [ ] **Step 6: Add tests for the tightened validation**

Append to `tests/test_dataset.py`:

```python
def test_negative_class_ids_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        CharacterData(
            images=np.zeros((1, 4, 4), dtype=np.float32),
            class_ids=np.array([-1], dtype=np.int64),
            writer_ids=None,
            classes=["あ"],
        )


def test_class_ids_must_be_int64():
    with pytest.raises(ValueError, match="int64"):
        CharacterData(
            images=np.zeros((1, 4, 4), dtype=np.float32),
            class_ids=np.array([0], dtype=np.int32),
            writer_ids=None,
            classes=["あ"],
        )
```

Append to `tests/test_torch_dataset.py`:

```python
def test_out_of_range_indices_are_rejected():
    import numpy as np
    import pytest

    from jvqvae.data.dataset import CharacterData
    from jvqvae.data.torch_dataset import CharacterTorchDataset

    data = CharacterData(
        images=np.zeros((2, 4, 4), dtype=np.float32),
        class_ids=np.zeros(2, dtype=np.int64),
        writer_ids=None,
        classes=["あ"],
    )
    with pytest.raises(ValueError, match="indices"):
        CharacterTorchDataset(data, np.array([0, 5]))
```

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS. If the tightened `CharacterData` validation breaks a plan 1 test, the
test was relying on a loose dtype — fix the test's fixture, not the validation.

- [ ] **Step 8: Verify the commands by hand**

Run each and confirm clean output, not a traceback:

```bash
uv run jvqvae --help
uv run jvqvae train-vqvae --help
uv run jvqvae extract-codes --help
```

Then a genuine smoke run on K49 if `data/k49` is present, capped so it finishes in
seconds. **This is the only real-data run in the plan; do not raise the cap.**

```bash
uv run jvqvae train-vqvae --config configs/k49_dev.yaml --device cpu \
  --max-steps 3 --output-root /tmp/jvqvae_smoke
```

Expected: a device line, one epoch line reporting loss and perplexity, and a final line
naming the figure. Confirm `/tmp/jvqvae_smoke/results/k49_dev/vqvae_run.json` exists,
then delete `/tmp/jvqvae_smoke`. If `data/k49` is absent, skip this and say so in your
report.

- [ ] **Step 9: Update the README**

Replace the status line near the top with:

```markdown
**Status: the data pipeline and the VQ-VAE are complete; the transformer prior is not
written yet.** You can train the VQ-VAE and extract its codes today. There is no
sampling command, because sampling needs the prior.
```

Add a `## Training` section immediately after the `## Using the data` section:

````markdown
## Training

One config file per dataset holds everything a run needs — the data source, the model
and the optimizer settings — so a run is reproducible from a single path.

```bash
uv run jvqvae train-vqvae --config configs/k49_dev.yaml
```

Add `--max-steps 20` for a smoke run, or `--device cpu` to override the automatic
MPS/CUDA/CPU choice. Each run writes two checkpoints to
`checkpoints/<name>/` (best by validation loss, and last), a reconstruction figure and a
JSON record of every epoch to `results/<name>/`.

The per-epoch line reports codebook perplexity and how many of the 512 codes are in use.
Watch it: codebook collapse, where nearly all the codes go unused, is the expected
failure mode at this data scale. Codes that stay unused are automatically restarted from
live encoder outputs, and the count of restarts is logged.

Once the VQ-VAE is trained, extract the discrete codes the prior will learn from:

```bash
uv run jvqvae extract-codes --config configs/k49_dev.yaml --split train
```

A VQ-VAE cannot generate on its own. It learns a code for an image and how to decode
one, but holds no distribution over code grids, so a grid sampled uniformly decodes to
noise. Learning that distribution is the prior's job, and it is next.
````

- [ ] **Step 10: Commit**

```bash
git add jvqvae tests README.md
git commit -m "feat: add train-vqvae and extract-codes commands

Also clears deferred plan 1 findings in the containers the models now
depend on: CharacterData enforces int64 one-dimensional label arrays and
rejects negative class ids, and the torch adapter bounds-checks indices
at construction rather than at access."
```

---

## Definition of done

- `uv run pytest -q` passes with no network, no ETL4 file, and no checkpoint present, in
  under about 60 seconds.
- `uv run jvqvae train-vqvae --config configs/k49_dev.yaml --max-steps 3` completes and
  writes a checkpoint, a JSON record and a figure.
- `uv run jvqvae extract-codes` writes a `CodeData` file that `load_codes` reads back.
- No long training run has been launched, and no checkpoint, figure or dataset is
  committed.
- `README.md` states accurately what works.

## What plan 3 will consume

- `load_run_config(path) -> RunConfig`, extended there with a `prior:` section.
- `CodeData` from `jvqvae/codes.py`: `codes` (N, G, G) int64, `class_ids`, `classes`,
  `num_codes`, `sequence_length`, `num_classes`.
- `load_checkpoint(path, device) -> (VQVAE, meta)` to decode sampled grids into images.
- `set_seed`, and the JSON-run-record and checkpoint conventions established in Task 5.
