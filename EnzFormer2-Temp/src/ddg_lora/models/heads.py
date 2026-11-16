"""Regression head for the checkpoint-compatible Siamese-delta regressor."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class DeltaRegressionHead(nn.Module):
    """Map a concat feature vector to a scalar ddG estimate.

    Default layer plan:

        LayerNorm -> Linear(in_dim, 512) -> GELU -> Dropout
                  -> Linear(512, 128)    -> GELU -> Dropout
                  -> Linear(128, 1)
    """

    def __init__(
        self,
        in_dim: int,
        *,
        dropout: float = 0.1,
        hidden_dims: Sequence[int] = (512, 128),
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        hidden_dims = tuple(int(d) for d in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(d <= 0 for d in hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {hidden_dims}")
        self.in_dim = int(in_dim)
        self.dropout = float(dropout)
        self.hidden_dims = hidden_dims

        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = int(in_dim)
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 1:
            return self.body(feat).squeeze(-1)
        if feat.dim() == 2:
            return self.body(feat).squeeze(-1)
        raise ValueError(
            f"DeltaRegressionHead expects [in_dim] or [B, in_dim], got shape {tuple(feat.shape)}"
        )
