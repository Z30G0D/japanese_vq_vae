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
