# Transformer Prior and Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VQ-VAE generative: a class-conditional decoder-only transformer that
learns the distribution of 8x8 code grids, plus sampling with classifier-free guidance
that decodes those grids back into character images.

**Architecture:** The prior never sees pixels. It is trained teacher-forced on
`CodeData` extracted from a frozen VQ-VAE. A class token is prepended, a start token
follows, then the 64 codes in raster order. Positions on the grid use factorized row
and column embeddings. Ten percent label dropout at train time enables CFG at sample
time. Completed grids are decoded by the frozen VQ-VAE.

**Tech Stack:** Python 3.12, uv, PyTorch, numpy, matplotlib, pytest. No new
dependencies. No `nn.Transformer`: a small causal stack, because the default
Transformer is encoder-style and the causal-mask test has to be obviously true of the
code we write.

**Spec:** `docs/superpowers/specs/2026-11-05-japanese-vq-vae-design.md`
**Plan 1 (complete):** `docs/superpowers/plans/2026-11-09-foundation-and-data-pipeline.md`
**Plan 2 (complete):** `docs/superpowers/plans/2026-11-09-vqvae-model-and-training.md`

## Global Constraints

- Python 3.12. Dependencies limited to: torch, numpy, pillow, pyyaml, matplotlib,
  pytest. No experiment-tracking service. Do not add flash-attention, transformers, or
  any sampling library.
- Package name is `jvqvae`. Run everything through `uv run`.
- The repository must never contain ETL4 data and must never download it.
- **The full test suite must pass with no network access, no ETL4 file present, and no
  trained checkpoint present.** Every model test builds its own tiny tensors.
- **No test may run more than a few hundred optimizer steps on tensors larger than a
  handful of grids.** The suite's total runtime must stay under about 60 seconds. Tests
  that need a model use a deliberately tiny config (the `tiny_config` fixture), never
  the shipped ETL4 or K49 configs.
- All randomness goes through an explicit seed. No global `np.random` calls; torch
  randomness is seeded once per run through `set_seed`.
- Tests must run on CPU regardless of what hardware is present. Never call
  `select_device()` inside a test; pass `torch.device("cpu")` explicitly.
- Checkpoints load with `torch.load(..., weights_only=True)`. Never unpickle arbitrary
  objects from a checkpoint file.
- Everything written at runtime goes under `checkpoints/` or `results/`, both already
  gitignored. Never write into the repository root or `jvqvae/`.
- Decoder outputs are unbounded (plan 2 dropped the sigmoid). Clip to `[0, 1]` when
  drawing sample figures, the same way `save_reconstruction_grid` already does.

### Execution note: do not train

The user will run the real prior training themselves. **No task in this plan trains on
extracted K49 or ETL4 codes, and no task loads `checkpoints/k49_dev/`.** The definition
of done is that `uv run jvqvae train-prior` and `uv run jvqvae sample` work, and a
few-step smoke run on a synthetic `CodeData` drives the loss down. Do not launch long
jobs, do not leave background processes running, and do not commit checkpoints or
figures.

### Known deviations from the spec

Spec section 6 sketches `etl4_prior.yaml` and a nested CLI of `train prior` / `sample`.
This plan keeps the two deviations plan 2 already shipped:

1. **One run config per dataset.** `configs/etl4.yaml` and `configs/k49_dev.yaml` each
   grow a `prior:` section. A separate prior YAML would duplicate the data block.
2. **Flat hyphenated CLI verbs.** `train-prior` and `sample`, matching `train-vqvae`.

Spec section 5 sizes the class table to the dataset (51 / 49). The prior is constructed
from `CodeData.num_classes`, not from a YAML field, so one config serves both sources.
The unconditional (dropped-label) class is an extra row, index `num_classes`.

Spec section 5 does not name nucleus `top_p`, CFG scale, or prior epoch counts. Those
are this plan's call, recorded in `PriorConfig` defaults below.

## Review Focus

- A broken causal mask: perturbing token *i* must leave logits at positions `< i`
  unchanged. Owned by Task 2 (`test_perturbing_a_later_token_does_not_change_earlier_logits`).
- CFG scale 1 must equal the conditional forward, scale 0 the unconditional. Owned by
  Task 3 (`test_guidance_scale_one_matches_conditional`,
  `test_guidance_scale_zero_matches_unconditional`).
- Sampled indices must lie in `[0, num_codes)`, never BOS. Owned by Task 3
  (`test_generate_returns_a_grid_of_codebook_indices`).
- A prior checkpoint must rebuild the model from the file alone, including
  `num_classes` / `num_codes` / `latent_size`, with `weights_only=True`. Owned by
  Task 2 (`test_a_checkpoint_rebuilds_the_prior_without_being_handed_sizes`).
- `sample` / `train-prior` must fail with a path-naming `SystemExit` when the
  VQ-VAE checkpoint or codes file is missing, not a raw `FileNotFoundError`. Owned by
  Task 7 (`test_train_prior_rejects_a_missing_codes_file`,
  `test_sample_rejects_a_missing_vqvae_checkpoint`).

---

## File Structure

| File | Responsibility |
|---|---|
| `jvqvae/config.py` | **Modify.** Add `PriorConfig`; `RunConfig.prior`; allow `prior:` in YAML |
| `tests/conftest.py` | **Modify.** Tiny prior on `tiny_config` |
| `jvqvae/models/prior.py` | Causal transformer, checkpoints, generate |
| `jvqvae/codes.py` | **Modify.** `CodeTorchDataset` |
| `jvqvae/train/prior.py` | Training loop, JSON run record |
| `jvqvae/sample.py` | Decode generated grids through a frozen VQ-VAE |
| `jvqvae/figures.py` | **Modify.** Conditional sample grid and guidance-strength sweep |
| `jvqvae/cli.py` | **Modify.** `train-prior`, `sample` |
| `configs/etl4.yaml`, `configs/k49_dev.yaml` | **Modify.** Add `prior:` |
| `README.md` | **Modify.** Document prior training and sampling |
| `tests/test_config.py` | **Modify.** `prior:` is a real section; pick a new unknown-section name |
| `tests/test_prior.py`, `test_train_prior.py`, `test_sample.py` | New |
| `tests/test_figures.py`, `test_cli.py`, `test_codes.py` | **Modify.** |

## Sequence layout (locked)

For a grid of side `G` (`latent_size`), `N = G * G` codes in raster order
(row-major: `(0,0), (0,1), …, (G-1,G-1)`):

| Position | Token | Embedding |
|---|---|---|
| 0 | class id, or `num_classes` when dropped | `class_embed[id] + class_pos` |
| 1 | BOS = `num_codes` | `tok_embed[BOS] + bos_pos` |
| `2 + r*G + c` | code at `(r, c)` | `tok_embed[code] + row_embed[r] + col_embed[c]` |

Input length is `N + 2`. Next-token logits at every position; the output head has
size `num_codes` (codes only — never BOS, never class). Cross-entropy is applied at
positions `1 .. N` (inclusive), whose targets are the `N` codes. Position 0 is never
in the loss.

`UNCOND = num_classes`. `BOS = num_codes`. Label dropout replaces the class id with
`UNCOND` independently per example with probability `label_dropout`.

Classifier-free guidance, at each sample step:

```
logits = uncond + scale * (cond - uncond)
```

`scale=1` is the conditional model; `scale=0` is the unconditional model.

---

### Task 1: Prior configuration

**Files:**
- Modify: `jvqvae/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_config.py`
- Modify: `configs/etl4.yaml`, `configs/k49_dev.yaml`

**Interfaces:**
- Consumes: `RunConfig`, `_build`, `_SECTIONS`, `load_run_config`.
- Produces:
  - Frozen `PriorConfig(num_layers=4, num_heads=4, width=256, label_dropout=0.1, temperature=1.0, top_p=0.9, guidance_scale=1.5, epochs=30, batch_size=64, learning_rate=3e-4, max_steps=None)`.
  - `RunConfig` gains `prior: PriorConfig = field(default_factory=PriorConfig)`.
  - `_SECTIONS` becomes `("name", "data", "model", "train", "prior")`.
  - Seed, device, `checkpoint_dir`, `results_dir` stay on `TrainConfig` and are shared.
    Prior-specific epochs / batch / lr / max_steps live on `PriorConfig` so the two
    models can disagree in one file.

`num_heads` must divide `width`. `label_dropout` and `top_p` in `[0, 1]`. `temperature`
and `guidance_scale` must be `> 0`. Other scalars follow the same positivity rules as
`TrainConfig` / `VQVAEConfig`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`:

- Change `test_unknown_top_level_sections_are_rejected` so the unknown section is
  `optim:`, not `prior:`. `prior:` is about to become valid; leaving this test as-is
  makes Task 1's implementation look like a regression.
- `test_defaults_fill_in_every_section`: also assert `config.prior.num_layers == 4`,
  `num_heads == 4`, `width == 256`, `label_dropout == 0.1`, `top_p == 0.9`,
  `guidance_scale == 1.5`, `epochs == 30`.
- `test_sections_override_defaults`: a YAML `prior:\n  num_layers: 2\n  epochs: 5\n`
  must stick.
- `test_every_config_is_immutable`: add `PriorConfig()` to the parametrize list.
- `test_shipped_configs_load_and_reach_an_eight_by_eight_grid`: also
  `isinstance(config.prior, PriorConfig)`.
- Extend `test_out_of_range_scalars_are_rejected` with:
  `("prior", "prior:\n  num_layers: 0\n")`,
  `("prior", "prior:\n  num_heads: 0\n")`,
  `("prior", "prior:\n  width: 0\n")`,
  `("prior", "prior:\n  label_dropout: 1.5\n")`,
  `("prior", "prior:\n  top_p: -0.1\n")`,
  `("prior", "prior:\n  temperature: 0\n")`,
  `("prior", "prior:\n  guidance_scale: 0\n")`,
  `("prior", "prior:\n  epochs: 0\n")`.
- New: `test_num_heads_must_divide_width` — `PriorConfig(num_heads=3, width=256)`
  raises `ValueError` matching `divisible`.
- New: `test_run_config_default_prior_is_the_spec_transformer` — a `RunConfig`
  built with only `name` and `data` has `prior == PriorConfig()`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL on the new prior assertions (and the old `prior:` unknown-section
test no longer applies once rewritten).

- [ ] **Step 3: Implement `PriorConfig` and wire it into `load_run_config`**

Add the dataclass next to `VQVAEConfig`. Validate in `__post_init__`. Parse the
`prior` section the same way as `model` / `train`. Put a `prior:` block with the
defaults written out in both shipped YAMLs (so a reader can see the knobs without
opening the dataclass).

In `tests/conftest.py`, give `tiny_config` a tiny prior so later tasks stay fast:

```python
prior=PriorConfig(
    num_layers=1, num_heads=2, width=16, epochs=1, batch_size=4, max_steps=2
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/conftest.py -q`
Expected: PASS. Then `uv run pytest -q` — the rest of the suite must still pass;
`tiny_config` grew a field with a default so existing `RunConfig(...)` calls that
omit `prior=` keep working.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/config.py tests/conftest.py tests/test_config.py configs/etl4.yaml configs/k49_dev.yaml
git commit -m "feat: add prior configuration to the run YAML"
```

---

### Task 2: Causal transformer prior

**Files:**
- Create: `jvqvae/models/prior.py`
- Test: `tests/test_prior.py`

**Interfaces:**
- Consumes: `PriorConfig`.
- Produces:
  - `PriorOutput(logits: Tensor, loss: Tensor | None)` — `logits` is
    `(B, N+2, num_codes)`.
  - `class Prior(nn.Module)`:
    - `__init__(self, config: PriorConfig, *, num_codes: int, num_classes: int, latent_size: int)`
    - `forward(self, codes: Tensor, class_ids: Tensor, *, dropout_labels: bool = True) -> PriorOutput`
      `codes` is `(B, G, G)` int64 in `[0, num_codes)`; `class_ids` is `(B,)` int64.
      When `dropout_labels` is true, drop each label independently with
      `config.label_dropout`. Loss is mean token cross-entropy on the `N` code
      positions.
    - `bos_id` property = `num_codes`; `uncond_id` property = `num_classes`.
  - `save_prior_checkpoint(model, path, *, config, num_codes, num_classes, latent_size, epoch, metrics) -> None`
  - `load_prior_checkpoint(path, *, device) -> tuple[Prior, dict]` — eval mode,
    `weights_only=True`. The dict contains `config`, `num_codes`, `num_classes`,
    `latent_size`, `epoch`, `metrics`.

No normalization layers beyond pre-norm inside each block (LayerNorm, then causal
attention, residual, LayerNorm, MLP, residual). Causal mask is a registered buffer,
`-inf` above the diagonal. One linear output head, no softmax.

- [ ] **Step 1: Write the failing tests**

`tests/test_prior.py`. Helper: `make_prior(tiny_config, num_classes=3) -> Prior`.

```python
def test_forward_logits_cover_the_full_sequence(tiny_config):
    prior = make_prior(tiny_config)
    codes = torch.randint(0, tiny_config.model.num_codes, (2, 4, 4))
    out = prior(codes, torch.tensor([0, 2]), dropout_labels=False)
    # 4x4 grid -> N=16, sequence = 18
    assert out.logits.shape == (2, 18, tiny_config.model.num_codes)
    assert out.loss is not None and out.loss.ndim == 0


def test_perturbing_a_later_token_does_not_change_earlier_logits(tiny_config):
    """The spec's silent-failure test. A broken mask trains well and samples garbage."""
    prior = make_prior(tiny_config).eval()
    codes = torch.randint(0, tiny_config.model.num_codes, (1, 4, 4))
    class_ids = torch.tensor([1])
    base = prior(codes, class_ids, dropout_labels=False).logits
    codes_perturbed = codes.clone()
    codes_perturbed[0, -1, -1] = (codes_perturbed[0, -1, -1] + 1) % tiny_config.model.num_codes
    other = prior(codes_perturbed, class_ids, dropout_labels=False).logits
    # Positions 0..N (class, BOS, and every code except the last) must match.
    # The last code sits at sequence index N+1 = 17; everything before it is causal.
    assert torch.allclose(base[:, :-1], other[:, :-1], atol=1e-5)


def test_the_loss_reaches_the_class_embedding(tiny_config):
    prior = make_prior(tiny_config)
    codes = torch.randint(0, tiny_config.model.num_codes, (2, 4, 4))
    prior(codes, torch.tensor([0, 1]), dropout_labels=False).loss.backward()
    assert prior.class_embed.weight.grad is not None
    assert prior.class_embed.weight.grad.abs().sum() > 0
```

Also:

- `test_forward_rejects_codes_outside_the_codebook` — `ValueError` matching `codebook`.
- `test_forward_rejects_a_class_id_outside_the_table` — `ValueError` matching `class`.
- `test_label_dropout_can_be_disabled` — with `label_dropout=1.0` and
  `dropout_labels=False`, two forwards with the same codes and different class ids
  produce **different** logits at position 0 (the class slot). With
  `dropout_labels=True` they produce **identical** logits at position 0, because every
  label becomes `uncond_id`.
- `test_a_few_steps_of_training_drive_the_loss_down` — 20 Adam steps at `1e-2` on four
  fixed grids, `dropout_labels=False`; final loss < first loss.
- `test_checkpoint_round_trip_reproduces_identical_logits`.
- `test_a_checkpoint_rebuilds_the_prior_without_being_handed_sizes`.
- `test_a_loaded_checkpoint_comes_back_in_eval_mode`.
- `test_loading_a_checkpoint_does_not_unpickle_arbitrary_objects` — keys exactly
  `{"model", "config", "num_codes", "num_classes", "latent_size", "epoch", "metrics"}`.
- `test_positional_misuse_is_rejected` — `Prior(config, 8, 3, 4)` raises `TypeError`
  (keyword-only after `config`, matching encoder/decoder).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prior.py -q`
Expected: FAIL with `Prior` not defined / import error.

- [ ] **Step 3: Implement `Prior`, `save_prior_checkpoint`, `load_prior_checkpoint`**

`jvqvae/models/prior.py`. Keep the causal mask construction in one place and use it in
`forward`; do not rely on `is_causal=True` of a library call unless the causal test
still passes if that flag is deleted — the test is the spec.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prior.py -q`
Expected: PASS. Then `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/models/prior.py tests/test_prior.py
git commit -m "feat: add a class-conditional causal transformer prior"
```

---

### Task 3: Sampling, nucleus, and classifier-free guidance

**Files:**
- Modify: `jvqvae/models/prior.py`
- Modify: `tests/test_prior.py`

**Interfaces:**
- Consumes: `Prior.forward`.
- Produces:
  - `nucleus(logits: Tensor, top_p: float) -> Tensor` — same shape as `logits`,
    disallowed logits set to `-inf`. `top_p >= 1` is a no-op; `top_p <= 0` is invalid
    (already rejected by config).
  - `apply_guidance(cond: Tensor, uncond: Tensor, scale: float) -> Tensor` equal to
    `uncond + scale * (cond - uncond)`.
  - `Prior.generate(self, class_ids: Tensor, *, temperature: float | None = None, top_p: float | None = None, guidance_scale: float | None = None) -> Tensor`
    in eval/`no_grad`. Returns `(B, G, G)` int64. `None` falls back to `self.config`.
    At each of the `N` steps, run a conditional and an unconditional forward (or one
    batched forward of size `2B`), take the last-position logits, apply guidance,
    divide by temperature, nucleus-filter, then `torch.multinomial` on the softmax.

- [ ] **Step 1: Write the failing tests**

```python
def test_nucleus_is_a_noop_at_top_p_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(nucleus(logits, 1.0), logits)


def test_nucleus_zeros_out_the_tail_below_top_p():
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    filtered = nucleus(logits, 0.5)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert filtered[0, 2] == logits[0, 2]


def test_guidance_scale_one_matches_conditional():
    cond, uncond = torch.randn(2, 5), torch.randn(2, 5)
    assert torch.allclose(apply_guidance(cond, uncond, 1.0), cond)


def test_guidance_scale_zero_matches_unconditional():
    cond, uncond = torch.randn(2, 5), torch.randn(2, 5)
    assert torch.allclose(apply_guidance(cond, uncond, 0.0), uncond)


def test_generate_returns_a_grid_of_codebook_indices(tiny_config):
    prior = make_prior(tiny_config).eval()
    codes = prior.generate(torch.tensor([0, 1, 2]), temperature=1.0, top_p=1.0, guidance_scale=1.0)
    assert codes.shape == (3, 4, 4)
    assert codes.dtype == torch.int64
    assert int(codes.min()) >= 0
    assert int(codes.max()) < tiny_config.model.num_codes


def test_generate_is_deterministic_under_a_fixed_seed(tiny_config):
    prior = make_prior(tiny_config).eval()
    class_ids = torch.tensor([0, 1])
    torch.manual_seed(0)
    first = prior.generate(class_ids, temperature=1.0, top_p=1.0, guidance_scale=1.0)
    torch.manual_seed(0)
    second = prior.generate(class_ids, temperature=1.0, top_p=1.0, guidance_scale=1.0)
    assert torch.equal(first, second)


def test_generate_does_not_update_parameters(tiny_config):
    prior = make_prior(tiny_config).train()
    before = {n: p.detach().clone() for n, p in prior.named_parameters()}
    prior.generate(torch.tensor([0]), temperature=1.0, top_p=1.0, guidance_scale=1.0)
    for name, param in prior.named_parameters():
        assert torch.equal(param, before[name])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prior.py -q`
Expected: FAIL on the new names.

- [ ] **Step 3: Implement `nucleus`, `apply_guidance`, and `Prior.generate`**

Keep `generate` on the module so sampling cannot drift from the sequence layout
`forward` uses. Do not softmax inside `nucleus`; it works on logits.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prior.py -q`
Expected: PASS. Then `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/models/prior.py tests/test_prior.py
git commit -m "feat: sample code grids with nucleus filtering and CFG"
```

---

### Task 4: Prior training loop

**Files:**
- Modify: `jvqvae/codes.py` — add `CodeTorchDataset`
- Create: `jvqvae/train/prior.py`
- Test: `tests/test_codes.py` (dataset), `tests/test_train_prior.py`

**Interfaces:**
- Consumes: `Prior`, `CodeData`, `save_prior_checkpoint`, `set_seed`, `RunConfig`.
- Produces:
  - `class CodeTorchDataset(torch.utils.data.Dataset)` wrapping `CodeData`, yielding
    `(codes: Tensor[G,G] int64, class_id: Tensor[] int64)`.
  - `train_prior(config: RunConfig, splits: dict[str, CodeData], *, device, output_root=".", log=print) -> dict`
    Uses `splits["train"]` (required, non-empty) and optional `splits["val"]`.
    Optimizer Adam at `config.prior.learning_rate`. Loop `config.prior.epochs`,
    capped by `config.prior.max_steps`. Per epoch: mean train CE, optional val CE,
    save `checkpoints/<name>/prior_best.pt` on best val (or last, if no val) and
    `prior_last.pt`. Write `results/<name>/prior_run.json` with the same metadata
    shape as the VQ-VAE record (`name`, `created`, `device`, `torch_version`,
    `config`, `num_train`, `num_val`, `total_steps`, `best_epoch`, `best_val_loss`,
    `epochs`). Leave the model in eval mode.
    Seed via `set_seed(config.train.seed)`. Batch size `config.prior.batch_size`.

- [ ] **Step 1: Write the failing tests**

`CodeTorchDataset`:

- `test_code_dataset_yields_a_grid_and_a_class_id` — length matches `CodeData`;
  `dataset[0]` is two tensors, grid shape `(G,G)`, dtype int64.
- `test_code_dataset_preserves_order`.

`tests/test_train_prior.py`, mirroring `tests/test_train_vqvae.py` at a smaller
surface. Fixture `code_splits` builds a `CodeData` of 12 random `4x4` grids,
3 classes, `num_codes=8`, split 8/4 train/val.

- `test_training_returns_one_record_per_epoch`
- `test_max_steps_caps_the_run` — `config.prior.max_steps`
- `test_every_epoch_reports_train_and_val_cross_entropy` — each epoch dict has
  `epoch`, `train_loss`, `val_loss`
- `test_best_checkpoint_is_the_lowest_validation_loss`
- `test_a_run_without_validation_still_writes_best`
- `test_empty_train_split_is_rejected`
- `test_the_json_record_is_written`
- `test_training_leaves_the_model_in_eval_mode` (load `prior_last.pt`, assert not
  training)
- `test_a_few_steps_drive_the_recorded_loss_down`

Do not assert on K49 files.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_prior.py tests/test_codes.py -q`
Expected: FAIL on the new names.

- [ ] **Step 3: Implement `CodeTorchDataset` and `train_prior`**

`train_prior` should look like `train_vqvae` on purpose: same record keys, same
two-checkpoint pattern, same `log=` injection. The loss is `out.loss` from
`Prior.forward(..., dropout_labels=True)` on train batches and
`dropout_labels=False` on val batches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_prior.py tests/test_codes.py tests/test_prior.py -q`
Expected: PASS. Then `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/codes.py jvqvae/train/prior.py tests/test_codes.py tests/test_train_prior.py
git commit -m "feat: train the prior on extracted code grids"
```

---

### Task 5: Decode samples through the frozen VQ-VAE

**Files:**
- Create: `jvqvae/sample.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `Prior.generate`, `VQVAE.decode`.
- Produces:
  - `sample_images(prior: Prior, vqvae: VQVAE, class_ids: Tensor, *, device, temperature=None, top_p=None, guidance_scale=None) -> Tensor`
    Shape `(B, 1, H, H)` where `H = latent_size * 2**vqvae.config.num_downsample`.
    Runs under `no_grad` in eval mode, then restores each module's original
    train/eval flag. Clip is **not** applied here (tensors stay raw); figures clip.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_images_match_the_vqvae_decode_size(tiny_config, tiny_data):
    vqvae = VQVAE(tiny_config.model).eval()
    prior = make_prior(tiny_config).eval()
    images = sample_images(prior, vqvae, torch.tensor([0, 1, 2]), device=CPU)
    assert images.shape == (3, 1, 16, 16)


def test_sample_images_do_not_move_the_codebook(tiny_config):
    vqvae = VQVAE(tiny_config.model).train()
    before = vqvae.quantizer.codebook.clone()
    sample_images(make_prior(tiny_config), vqvae, torch.tensor([0]), device=CPU)
    assert torch.equal(vqvae.quantizer.codebook, before)


def test_sample_images_restore_training_flags(tiny_config):
    vqvae = VQVAE(tiny_config.model).train()
    prior = make_prior(tiny_config).train()
    sample_images(prior, vqvae, torch.tensor([0]), device=CPU)
    assert vqvae.training
    assert prior.training
```

Also: `test_sampled_codes_decode_the_same_as_calling_decode_yourself` — capture
`prior.generate` under a fixed seed, `vqvae.decode` those codes, compare to
`sample_images` under the same seed (`atol=1e-5`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sample.py -q`
Expected: FAIL, `sample_images` not defined.

- [ ] **Step 3: Implement `sample_images` in `jvqvae/sample.py`**

Generate on `device`, decode on `device`, return the images tensor still on
`device` (callers `.cpu()` when they need to).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sample.py tests/test_prior.py -q`
Expected: PASS. Then `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/sample.py tests/test_sample.py
git commit -m "feat: decode prior samples through the frozen VQ-VAE"
```

---

### Task 6: Sample grid and guidance-strength sweep figures

**Files:**
- Modify: `jvqvae/figures.py`
- Modify: `tests/test_figures.py`

**Interfaces:**
- Consumes: `sample_images`.
- Produces:
  - `save_sample_grid(prior, vqvae, path, *, class_ids: Sequence[int], samples_per_class: int = 4, device, temperature=None, top_p=None, guidance_scale=None, seed=0) -> Path`
    Rows are classes, columns are independent samples. Titles are class indices,
    not glyphs (same CJK-font reason as reconstructions). Clip to `[0, 1]`.
    `vmin=0, vmax=1`. `set_seed(seed)` before generating.
  - `save_guidance_sweep(prior, vqvae, path, *, class_id: int, scales: Sequence[float] = (0.0, 1.0, 1.5, 2.0, 3.0), num_samples: int = 4, device, temperature=None, top_p=None, seed=0) -> Path`
    Rows are samples (shared noise per row: generate cond/uncond logits from the
    same prefix by running `generate` once per scale is acceptable; shared-noise
    CFG is nicer but not required). Columns are `scales`. Column titles are the
    scale values. Spec: "a guidance-strength sweep figure for a single character."

- [ ] **Step 1: Write the failing tests**

- `test_the_sample_grid_is_written` — path exists, size > 0, return value is the path.
- `test_the_guidance_sweep_is_written`.
- `test_sample_grid_does_not_leave_the_vqvae_training`.
- `test_an_empty_class_list_is_rejected` — `ValueError` matching `empty`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_figures.py -q`
Expected: FAIL on the new names.

- [ ] **Step 3: Implement the two figure functions**

Follow `save_reconstruction_grid`: Agg backend already selected at import.
Restore each module's train/eval flag.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_figures.py -q`
Expected: PASS. Then `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/figures.py tests/test_figures.py
git commit -m "feat: render conditional samples and a guidance-strength sweep"
```

---

### Task 7: CLI, README, and missing-file errors

**Files:**
- Modify: `jvqvae/cli.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `train_prior`, `load_codes`, `load_checkpoint` (VQ-VAE),
  `load_prior_checkpoint`, `sample_images`, `save_sample_grid`,
  `save_guidance_sweep`.
- Produces:
  - `jvqvae train-prior --config --device --max-steps --output-root --codes`
    Default codes path: `<output-root>/<results_dir>/<name>/codes_train.npz`.
    `--max-steps` overrides `config.prior.max_steps`. After training, do **not**
    sample; sampling is a separate command (the VQ-VAE checkpoint is a second
    input and the user may want to inspect the loss first).
  - `jvqvae sample --config --device --output-root --vqvae --prior --class --n --guidance --temperature --top-p --seed --sweep-class`
    Defaults: VQ-VAE `checkpoints/<name>/vqvae_best.pt`, prior
    `checkpoints/<name>/prior_best.pt`. `--class` may be repeated; omitting it
    samples every class in the prior checkpoint's class table.
    `--n` is samples per class (default 4). Writes
    `results/<name>/samples.png`. If `--sweep-class` is given, also writes
    `results/<name>/guidance_sweep.png` for that class id.
  - Missing codes / VQ-VAE / prior files raise `SystemExit` with the path and
    the command that produces it (`extract-codes` / `train-vqvae` /
    `train-prior`).

- [ ] **Step 1: Write the failing tests**

Help:

- `test_train_prior_help_lists_its_flags` — `--config`, `--device`, `--max-steps`,
  `--output-root`, `--codes`.
- `test_sample_help_lists_its_flags` — `--config`, `--vqvae`, `--prior`,
  `--class`, `--n`, `--guidance`, `--sweep-class`.
- `test_help_exits_cleanly` currently asserts `fetch-k49` in `--help`; also
  assert `train-prior` and `sample`.

End-to-end, monkeypatched like the existing train/extract tests:

- `test_train_prior_runs_end_to_end` — monkeypatch `load_run_config`, write a
  tiny `codes_train.npz` via `save_codes` from a `CodeData` matching
  `tiny_config`, run `train-prior --device cpu --max-steps 2 --output-root tmp`.
  Assert `checkpoints/tiny/prior_last.pt` and `results/tiny/prior_run.json`.
- `test_sample_runs_end_to_end` — write a VQ-VAE checkpoint and a prior
  checkpoint into tmp, monkeypatch config, run `sample --class 0 --n 2 --device cpu`.
  Assert `results/tiny/samples.png`.
- `test_train_prior_rejects_a_missing_codes_file` — `SystemExit` matching
  `extract-codes`.
- `test_sample_rejects_a_missing_vqvae_checkpoint` — `SystemExit` matching
  `train-vqvae`.
- `test_sample_rejects_a_missing_prior_checkpoint` — `SystemExit` matching
  `train-prior`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL on the new names.

- [ ] **Step 3: Implement the two subcommands and update the README**

README: replace the closing "Learning that distribution is the prior's job, and
it is next." with the two commands, the default paths, and a one-line reminder
that sampling needs **both** checkpoints. Keep the ETL4 / K49 attribution
blocks untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -q`
Expected: PASS. Then `uv run pytest -q` for the whole suite.

- [ ] **Step 5: Commit**

```bash
git add jvqvae/cli.py README.md tests/test_cli.py
git commit -m "feat: add train-prior and sample commands"
```

---

## Self-review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| Decoder-only causal transformer, 4/4/256 | 1 (config defaults), 2 (model) |
| Vocab = 512 codes + start token | 2 (`bos_id = num_codes`) |
| Factorized row/column embeddings | 2 (sequence layout) |
| Class embedding prepended | 2 |
| Teacher-forced CE on extracted codes | 2, 4 |
| 10% label dropout / CFG | 1 default, 2 dropout, 3 guidance |
| Temperature + nucleus, then decode | 3, 5 |
| Causal-mask test | 2 |
| Conditional sample grid, all classes | 6, 7 (`--class` omitted) |
| Guidance-strength sweep, one character | 6, 7 (`--sweep-class`) |
| JSON run records | 4 |
| CLI train prior + sample | 7 |
| Suite without ETL4 / network / checkpoints | Global + every test |
| No sigmoid on decoded samples | Global; Task 6 clips in figures |

**Step scan:** each step names one test, one signature, or one command. Sampling
algorithm lives in Task 3 because `forward`'s layout does not determine nucleus
or CFG.

**Type consistency:** `PriorConfig` fields match YAML and `tiny_config`.
`train_prior` takes `dict[str, CodeData]`, not images. Checkpoints named
`prior_best.pt` / `prior_last.pt` in Tasks 4 and 7.

**Review Focus:** each of the five lines has a named test in the owning task.

**Proportion:** signatures, sequence layout, and test names; no transcribed
attention implementation.
