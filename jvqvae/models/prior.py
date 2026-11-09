"""Class-conditional causal transformer over a grid of VQ codes.

The sequence is the class token, a start token, then the grid in raster order.
Code positions add a token embedding to factorized row and column embeddings,
so the model sees the grid instead of a flat index. The output head predicts
codebook indices only: never the class and never the start token.

The causal mask is a registered buffer of ``-inf`` above the diagonal and is
added inside attention. It is not an ``is_causal`` flag on a library module,
because dropping that flag would still train and then sample garbage.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from jvqvae.config import PriorConfig


@dataclass
class PriorOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


def _causal_mask(seq: int) -> torch.Tensor:
    """Additive attention mask: 0 on and below the diagonal, -inf above it."""
    allowed = torch.ones(seq, seq, dtype=torch.bool).tril()
    return torch.zeros(seq, seq).masked_fill(~allowed, float("-inf"))


class _CausalSelfAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        batch, seq, width = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores + causal_mask
        weights = scores.softmax(dim=-1)
        mixed = weights @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq, width)
        return self.proj(mixed)


class _TransformerBlock(nn.Module):
    """Pre-norm block: LayerNorm, causal attention, residual, LayerNorm, MLP, residual."""

    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = _CausalSelfAttention(width, num_heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Prior(nn.Module):
    def __init__(
        self,
        config: PriorConfig,
        *,
        num_codes: int,
        num_classes: int,
        latent_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_codes = num_codes
        self.num_classes = num_classes
        self.latent_size = latent_size

        width = config.width
        # BOS sits one past the codebook. UNCOND sits one past the class table.
        self.tok_embed = nn.Embedding(num_codes + 1, width)
        self.class_embed = nn.Embedding(num_classes + 1, width)
        self.row_embed = nn.Embedding(latent_size, width)
        self.col_embed = nn.Embedding(latent_size, width)
        self.class_pos = nn.Parameter(torch.empty(1, 1, width))
        self.bos_pos = nn.Parameter(torch.empty(1, 1, width))

        self.blocks = nn.ModuleList(
            _TransformerBlock(width, config.num_heads)
            for _ in range(config.num_layers)
        )
        self.head = nn.Linear(width, num_codes)
        self.register_buffer(
            "causal_mask", _causal_mask(latent_size * latent_size + 2)
        )
        self.apply(_init_weights)
        nn.init.normal_(self.class_pos, std=0.02)
        nn.init.normal_(self.bos_pos, std=0.02)

    @property
    def bos_id(self) -> int:
        return self.num_codes

    @property
    def uncond_id(self) -> int:
        return self.num_classes

    def forward(
        self,
        codes: torch.Tensor,
        class_ids: torch.Tensor,
        *,
        dropout_labels: bool = True,
    ) -> PriorOutput:
        codes, class_ids = self._check_inputs(codes, class_ids)
        if dropout_labels:
            class_ids = self._drop_labels(class_ids)

        hidden = self._embed(codes, class_ids)
        for block in self.blocks:
            hidden = block(hidden, self.causal_mask)
        logits = self.head(hidden)
        return PriorOutput(logits=logits, loss=self._token_loss(logits, codes))

    def _check_inputs(
        self, codes: torch.Tensor, class_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.latent_size
        if codes.dtype != torch.int64:
            raise ValueError(
                f"codes must be int64 indices into the codebook, got {codes.dtype}"
            )
        if codes.ndim != 3 or codes.shape[1] != grid or codes.shape[2] != grid:
            raise ValueError(
                f"codes must have shape (batch, {grid}, {grid}), "
                f"got {tuple(codes.shape)}"
            )
        if codes.numel() and (
            int(codes.min()) < 0 or int(codes.max()) >= self.num_codes
        ):
            raise ValueError(
                f"codes must lie inside the codebook [0, {self.num_codes}), "
                f"got values in [{int(codes.min())}, {int(codes.max())}]"
            )

        batch = codes.shape[0]
        if class_ids.dtype != torch.int64:
            raise ValueError(f"class ids must be int64, got {class_ids.dtype}")
        if class_ids.shape != (batch,):
            raise ValueError(
                f"class ids must have shape ({batch},), got {tuple(class_ids.shape)}"
            )
        if class_ids.numel() and (
            int(class_ids.min()) < 0 or int(class_ids.max()) >= self.num_classes
        ):
            raise ValueError(
                f"class ids must lie in the class table [0, {self.num_classes}), "
                f"got values in [{int(class_ids.min())}, {int(class_ids.max())}]"
            )
        return codes, class_ids

    def _drop_labels(self, class_ids: torch.Tensor) -> torch.Tensor:
        """Replace each label with UNCOND, independently, at ``label_dropout``.

        This follows the flag rather than ``train()`` / ``eval()``. Sampling
        asks for a conditional or unconditional pass explicitly, and a module
        left in eval mode must still be able to drop every label.
        """
        drop = (
            torch.rand(class_ids.shape, device=class_ids.device)
            < self.config.label_dropout
        )
        dropped = class_ids.clone()
        dropped[drop] = self.uncond_id
        return dropped

    def _embed(self, codes: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        batch, grid, _ = codes.shape
        flat = codes.reshape(batch, grid * grid)
        positions = torch.arange(grid * grid, device=codes.device)
        rows = torch.div(positions, grid, rounding_mode="floor")
        cols = positions % grid
        code_tokens = (
            self.tok_embed(flat)
            + self.row_embed(rows).unsqueeze(0)
            + self.col_embed(cols).unsqueeze(0)
        )

        class_token = self.class_embed(class_ids).unsqueeze(1) + self.class_pos
        bos = self.tok_embed(
            torch.full(
                (batch, 1), self.bos_id, dtype=torch.long, device=codes.device
            )
        )
        bos_token = bos + self.bos_pos
        return torch.cat((class_token, bos_token, code_tokens), dim=1)

    def _token_loss(self, logits: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # Positions 1..N predict the N codes. Position 0 is the class token and
        # is never a target; the logit on the final code has nothing left to predict.
        num_codes_in_grid = self.latent_size * self.latent_size
        predicted = logits[:, 1 : num_codes_in_grid + 1, :]
        return F.cross_entropy(
            predicted.reshape(-1, self.num_codes),
            codes.reshape(-1),
        )


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


def save_prior_checkpoint(
    model: Prior,
    path: str | Path,
    *,
    config: PriorConfig,
    num_codes: int,
    num_classes: int,
    latent_size: int,
    epoch: int,
    metrics: dict,
) -> None:
    """Write a checkpoint that can rebuild the prior from the file alone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        # Stored as a plain dict so the file reads back with weights_only=True.
        "config": asdict(config),
        "num_codes": num_codes,
        "num_classes": num_classes,
        "latent_size": latent_size,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_prior_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[Prior, dict]:
    """Rebuild a prior and its metadata from a checkpoint.

    The model comes back in eval mode. Training mode is not part of a
    ``state_dict``, so a freshly built module would otherwise default to
    training.
    """
    raw = torch.load(path, map_location=device, weights_only=True)
    config = PriorConfig(**raw["config"])
    model = Prior(
        config,
        num_codes=raw["num_codes"],
        num_classes=raw["num_classes"],
        latent_size=raw["latent_size"],
    )
    model.load_state_dict(raw["model"])
    model.to(device)
    model.eval()
    return model, {
        "config": config,
        "num_codes": raw["num_codes"],
        "num_classes": raw["num_classes"],
        "latent_size": raw["latent_size"],
        "epoch": raw["epoch"],
        "metrics": raw["metrics"],
    }
