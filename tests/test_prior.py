from dataclasses import replace

import pytest
import torch

from jvqvae.config import PriorConfig
from jvqvae.models.prior import (
    Prior,
    apply_guidance,
    load_prior_checkpoint,
    nucleus,
    save_prior_checkpoint,
)

CPU = torch.device("cpu")


def make_prior(config, num_classes=3) -> Prior:
    return Prior(
        config.prior,
        num_codes=config.model.num_codes,
        num_classes=num_classes,
        latent_size=config.latent_size,
    )


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
    codes_perturbed[0, -1, -1] = (
        codes_perturbed[0, -1, -1] + 1
    ) % tiny_config.model.num_codes
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


def test_forward_rejects_codes_outside_the_codebook(tiny_config):
    prior = make_prior(tiny_config)
    codes = torch.zeros(1, 4, 4, dtype=torch.int64)
    codes[0, 0, 0] = tiny_config.model.num_codes
    with pytest.raises(ValueError, match="codebook"):
        prior(codes, torch.tensor([0]), dropout_labels=False)


def test_forward_rejects_a_class_id_outside_the_table(tiny_config):
    prior = make_prior(tiny_config)
    codes = torch.zeros(1, 4, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="class"):
        prior(codes, torch.tensor([3]), dropout_labels=False)


def test_label_dropout_can_be_disabled(tiny_config):
    prior = Prior(
        replace(tiny_config.prior, label_dropout=1.0),
        num_codes=tiny_config.model.num_codes,
        num_classes=3,
        latent_size=tiny_config.latent_size,
    ).eval()
    codes = torch.randint(0, tiny_config.model.num_codes, (1, 4, 4))

    kept_a = prior(codes, torch.tensor([0]), dropout_labels=False).logits
    kept_b = prior(codes, torch.tensor([1]), dropout_labels=False).logits
    assert not torch.allclose(kept_a[:, 0], kept_b[:, 0])

    dropped_a = prior(codes, torch.tensor([0]), dropout_labels=True).logits
    dropped_b = prior(codes, torch.tensor([1]), dropout_labels=True).logits
    assert torch.allclose(dropped_a[:, 0], dropped_b[:, 0])


def test_a_few_steps_of_training_drive_the_loss_down(tiny_config):
    torch.manual_seed(0)
    prior = make_prior(tiny_config)
    codes = torch.randint(0, tiny_config.model.num_codes, (4, 4, 4))
    class_ids = torch.tensor([0, 1, 2, 0])
    optimizer = torch.optim.Adam(prior.parameters(), lr=1e-2)

    first = prior(codes, class_ids, dropout_labels=False).loss.item()
    loss = first
    for _ in range(20):
        optimizer.zero_grad()
        loss = prior(codes, class_ids, dropout_labels=False).loss
        loss.backward()
        optimizer.step()

    assert loss.item() < first


def test_checkpoint_round_trip_reproduces_identical_logits(tiny_config, tmp_path):
    torch.manual_seed(0)
    prior = make_prior(tiny_config).eval()
    codes = torch.randint(0, tiny_config.model.num_codes, (2, 4, 4))
    class_ids = torch.tensor([0, 2])
    expected = prior(codes, class_ids, dropout_labels=False).logits

    path = tmp_path / "prior.pt"
    save_prior_checkpoint(
        prior,
        path,
        config=tiny_config.prior,
        num_codes=tiny_config.model.num_codes,
        num_classes=3,
        latent_size=tiny_config.latent_size,
        epoch=7,
        metrics={"val_loss": 0.5},
    )
    restored, meta = load_prior_checkpoint(path, device=CPU)

    got = restored(codes, class_ids, dropout_labels=False).logits
    assert torch.allclose(got, expected, atol=1e-6)
    assert meta["epoch"] == 7
    assert meta["metrics"]["val_loss"] == 0.5
    assert meta["config"] == tiny_config.prior


def test_a_checkpoint_rebuilds_the_prior_without_being_handed_sizes(
    tiny_config, tmp_path
):
    path = tmp_path / "prior.pt"
    save_prior_checkpoint(
        make_prior(tiny_config),
        path,
        config=tiny_config.prior,
        num_codes=tiny_config.model.num_codes,
        num_classes=3,
        latent_size=tiny_config.latent_size,
        epoch=0,
        metrics={},
    )

    restored, meta = load_prior_checkpoint(path, device=CPU)

    assert isinstance(meta["config"], PriorConfig)
    assert meta["num_codes"] == tiny_config.model.num_codes
    assert meta["num_classes"] == 3
    assert meta["latent_size"] == tiny_config.latent_size
    assert restored.bos_id == tiny_config.model.num_codes
    assert restored.uncond_id == 3


def test_a_loaded_checkpoint_comes_back_in_eval_mode(tiny_config, tmp_path):
    path = tmp_path / "prior.pt"
    save_prior_checkpoint(
        make_prior(tiny_config).train(),
        path,
        config=tiny_config.prior,
        num_codes=tiny_config.model.num_codes,
        num_classes=3,
        latent_size=tiny_config.latent_size,
        epoch=0,
        metrics={},
    )

    model, _ = load_prior_checkpoint(path, device=CPU)

    assert not model.training


def test_loading_a_checkpoint_does_not_unpickle_arbitrary_objects(
    tiny_config, tmp_path
):
    path = tmp_path / "prior.pt"
    save_prior_checkpoint(
        make_prior(tiny_config),
        path,
        config=tiny_config.prior,
        num_codes=tiny_config.model.num_codes,
        num_classes=3,
        latent_size=tiny_config.latent_size,
        epoch=0,
        metrics={},
    )

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {
        "model",
        "config",
        "num_codes",
        "num_classes",
        "latent_size",
        "epoch",
        "metrics",
    }


def test_positional_misuse_is_rejected(tiny_config):
    with pytest.raises(TypeError):
        Prior(tiny_config.prior, 8, 3, 4)


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
