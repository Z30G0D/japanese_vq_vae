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
