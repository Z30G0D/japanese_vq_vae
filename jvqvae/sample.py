"""Decode class-conditional prior samples through a frozen VQ-VAE."""

import torch
from torch import Tensor

from jvqvae.models.prior import Prior
from jvqvae.models.vqvae import VQVAE


@torch.no_grad()
def sample_images(
    prior: Prior,
    vqvae: VQVAE,
    class_ids: Tensor,
    *,
    device: torch.device,
    temperature: float | None = None,
    top_p: float | None = None,
    guidance_scale: float | None = None,
) -> Tensor:
    """Draw code grids from the prior and decode them to raw images.

    Both modules and ``class_ids`` are moved to ``device``. Sampling and
    decoding run in eval mode; each module's original train/eval flag is
    restored afterwards. Images stay on ``device`` and are not clipped.
    """
    prior_training = prior.training
    vqvae_training = vqvae.training
    prior.eval().to(device)
    vqvae.eval().to(device)
    class_ids = class_ids.to(device)
    try:
        codes = prior.generate(
            class_ids,
            temperature=temperature,
            top_p=top_p,
            guidance_scale=guidance_scale,
        )
        return vqvae.decode(codes)
    finally:
        prior.train(prior_training)
        vqvae.train(vqvae_training)
