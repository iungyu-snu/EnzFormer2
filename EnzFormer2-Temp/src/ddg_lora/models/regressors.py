"""ddG regressors built on top of an ESMC+LoRA backbone.

The checkpoint-compatible :class:`SiameseDeltaRegressor` is exported.

The regressor exposes this inference interface::

    predict_one(example: RegressionExample) -> torch.Tensor   # scalar ddG_hat

The same callable is used both during training (returns a tensor that retains
gradients into the LoRA + head parameters) and during evaluation (the trainer
wraps eval calls in ``torch.inference_mode``).

Backbone abstraction
--------------------

Each regressor takes a ``backbone`` argument that exposes a single method::

    backbone.forward_residue_hidden_states(seq: str) -> torch.Tensor  # [L, D]

In production this is :class:`ddg_lora.models.esmc_wrapper.ESMCMaskedLM`.
For unit tests we accept any object that provides the same method.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddg_lora.constants import AMINO_ACIDS_20
from ddg_lora.data.regression_dataset import RegressionExample
from ddg_lora.models.heads import DeltaRegressionHead
from ddg_lora.models.lora import lora_disabled


_NUM_AA = len(AMINO_ACIDS_20)  # 20


@dataclass(frozen=True)
class MaskedLogOddsConfig:
    enabled: bool = False
    source: str = "base_esmc_frozen"
    mode: str = "wt_context"
    transform: str = "train_zscore"
    fusion: str = "concat"
    feature_dropout: float = 0.0
    residual_gamma_init: float = 0.1
    mean: tuple[float, ...] = (0.0,)
    std: tuple[float, ...] = (1.0,)

    @classmethod
    def from_raw(cls, raw: dict | "MaskedLogOddsConfig" | None) -> "MaskedLogOddsConfig":
        if raw is None:
            return cls(enabled=False)
        if isinstance(raw, cls):
            return raw
        enabled = bool(raw.get("enabled", False))
        mode = str(raw.get("mode", "wt_context"))
        transform = str(raw.get("transform", "train_zscore"))
        fusion = str(raw.get("fusion", "concat"))
        source = str(raw.get("source", "base_esmc_frozen"))
        if mode not in {"wt_context", "symmetric", "both"}:
            raise ValueError(f"unknown masked_logodds.mode {mode!r}")
        if transform not in {"raw", "none", "train_zscore", "train_zscore_clip5"}:
            raise ValueError(f"unknown masked_logodds.transform {transform!r}")
        if fusion not in {"concat", "residual_prior"}:
            raise ValueError(f"unknown masked_logodds.fusion {fusion!r}")
        if source not in {"base_esmc_frozen", "lora_detached"}:
            raise ValueError(f"unknown masked_logodds.source {source!r}")
        if fusion == "residual_prior" and mode == "both":
            raise ValueError("masked_logodds.fusion=residual_prior requires a 1-D mode")
        dim = 2 if mode == "both" else 1
        mean = tuple(float(x) for x in raw.get("mean", [0.0] * dim))
        std = tuple(float(x) for x in raw.get("std", [1.0] * dim))
        feature_dropout = float(raw.get("feature_dropout", 0.0))
        residual_gamma_init = float(raw.get("residual_gamma_init", 0.1))
        if len(mean) != dim or len(std) != dim:
            raise ValueError(
                f"masked_logodds mean/std must have length {dim} for mode={mode!r}"
            )
        if any(x <= 0 for x in std):
            raise ValueError("masked_logodds std values must be positive")
        if not (0.0 <= feature_dropout <= 1.0):
            raise ValueError("masked_logodds.feature_dropout must be in [0, 1]")
        return cls(
            enabled=enabled,
            source=source,
            mode=mode,
            transform=transform,
            fusion=fusion,
            feature_dropout=feature_dropout,
            residual_gamma_init=residual_gamma_init,
            mean=mean,
            std=std,
        )

    @property
    def feature_dim(self) -> int:
        if not self.enabled:
            return 0
        return 2 if self.mode == "both" else 1

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "source": self.source,
            "mode": self.mode,
            "transform": self.transform,
            "fusion": self.fusion,
            "feature_dropout": self.feature_dropout,
            "residual_gamma_init": self.residual_gamma_init,
            "mean": list(self.mean),
            "std": list(self.std),
        }


class _HiddenStateBackbone(Protocol):
    def forward_residue_hidden_states(self, sequence: str) -> torch.Tensor: ...


def _onehot20(index: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not (0 <= index < _NUM_AA):
        raise IndexError(f"AA index {index} out of range [0, {_NUM_AA})")
    out = torch.zeros(_NUM_AA, device=device, dtype=dtype)
    out[index] = 1.0
    return out


class SiameseDeltaRegressor(nn.Module):
    """Shared ESMC(+LoRA) backbone over (wt_seq, mut_seq) + concat-feature MLP.

    Feature vector at the mutation position ``i``::

        feat = concat(
            h_wt_i,
            h_mut_i,
            h_mut_i - h_wt_i,
            h_mut_i * h_wt_i,
            onehot20(wt_aa),
            onehot20(mut_aa),
        )                                                  # [4D + 40]

    Optional context features (``use_context=True``)::

        local_delta  = mean(H_mut[i-k:i+k+1] - H_wt[i-k:i+k+1])    # [D]
        global_delta = mean(H_mut) - mean(H_wt)                     # [D]

    are appended in that order, growing the input to ``6D + 40``.
    """

    def __init__(
        self,
        backbone: _HiddenStateBackbone,
        hidden_dim: int,
        *,
        head_dropout: float = 0.1,
        head_hidden_dims: Sequence[int] = (512, 128),
        use_context: bool = False,
        context_window_k: int = 5,
        masked_logodds: dict | MaskedLogOddsConfig | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if context_window_k < 0:
            raise ValueError(f"context_window_k must be >= 0, got {context_window_k}")
        self.backbone = backbone
        self.hidden_dim = int(hidden_dim)
        self.use_context = bool(use_context)
        self.context_window_k = int(context_window_k)
        self.masked_logodds = MaskedLogOddsConfig.from_raw(masked_logodds)

        in_dim = 4 * self.hidden_dim + 2 * _NUM_AA
        if self.use_context:
            in_dim += 2 * self.hidden_dim
        if self.masked_logodds.enabled and self.masked_logodds.fusion == "concat":
            in_dim += self.masked_logodds.feature_dim
        self.head = DeltaRegressionHead(
            in_dim, dropout=head_dropout, hidden_dims=head_hidden_dims
        )
        self.masked_logodds_dropout = (
            nn.Dropout(self.masked_logodds.feature_dropout)
            if self.masked_logodds.enabled and self.masked_logodds.feature_dropout > 0
            else nn.Identity()
        )
        self.register_buffer(
            "masked_logodds_mean",
            torch.tensor(self.masked_logodds.mean, dtype=torch.float32),
        )
        self.register_buffer(
            "masked_logodds_std",
            torch.tensor(self.masked_logodds.std, dtype=torch.float32),
        )
        if self.masked_logodds.enabled and self.masked_logodds.fusion == "residual_prior":
            self.masked_logodds_gamma = nn.Parameter(
                torch.tensor(float(self.masked_logodds.residual_gamma_init))
            )
        else:
            self.register_parameter("masked_logodds_gamma", None)

    def set_masked_logodds_stats(
        self, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        """Update train-fit masked-logodds transform statistics."""
        mean_t = torch.tensor(
            list(mean), device=self.masked_logodds_mean.device, dtype=torch.float32
        )
        std_t = torch.tensor(
            list(std), device=self.masked_logodds_std.device, dtype=torch.float32
        )
        if mean_t.shape != self.masked_logodds_mean.shape:
            raise ValueError(
                f"masked_logodds mean shape mismatch: got {tuple(mean_t.shape)}, "
                f"expected {tuple(self.masked_logodds_mean.shape)}"
            )
        if torch.any(std_t <= 0):
            raise ValueError("masked_logodds std values must be positive")
        self.masked_logodds_mean.copy_(mean_t)
        self.masked_logodds_std.copy_(std_t)

    # ----- internal helpers --------------------------------------------------

    def _runtime_masked_logodds_pair(
        self,
        wt_sequence: str,
        mut_sequence: str,
        position_0idx: int,
        wt_aa: str,
        mut_aa: str,
    ) -> tuple[float, float]:
        if not hasattr(self.backbone, "masked_logodds_pair"):
            raise RuntimeError(
                "masked_logodds is enabled but examples have no precomputed "
                "log-odds and the backbone cannot compute them"
            )
        backbone_model = getattr(self.backbone, "model", None)
        was_training = (
            bool(backbone_model.training) if backbone_model is not None else False
        )
        disable_ctx = (
            lora_disabled(backbone_model)
            if (
                self.masked_logodds.source == "base_esmc_frozen"
                and backbone_model is not None
            )
            else nullcontext()
        )
        try:
            if backbone_model is not None:
                backbone_model.eval()
            with torch.inference_mode(), disable_ctx:
                return self.backbone.masked_logodds_pair(
                    wt_sequence,
                    mut_sequence,
                    position_0idx,
                    wt_aa=wt_aa,
                    mut_aa=mut_aa,
                )
        finally:
            if backbone_model is not None and was_training:
                backbone_model.train()

    def _raw_masked_logodds_from_values(
        self, forward: float, reverse: float
    ) -> torch.Tensor:
        if self.masked_logodds.mode == "wt_context":
            values = [float(forward)]
        elif self.masked_logodds.mode == "symmetric":
            values = [0.5 * (float(forward) - float(reverse))]
        elif self.masked_logodds.mode == "both":
            values = [float(forward), float(reverse)]
        else:  # pragma: no cover - validated at construction
            raise ValueError(f"unknown masked_logodds mode {self.masked_logodds.mode!r}")
        return torch.tensor(values, dtype=torch.float32)

    def _raw_masked_logodds_from_example(
        self, example: RegressionExample, *, reverse_direction: bool = False
    ) -> torch.Tensor | None:
        if not self.masked_logodds.enabled:
            return None
        forward = example.masked_logodds_forward
        reverse = example.masked_logodds_reverse
        if forward is None or reverse is None:
            forward, reverse = self._runtime_masked_logodds_pair(
                example.wt_sequence,
                example.mut_sequence,
                example.position_0idx,
                example.wt_aa,
                example.mut_aa,
            )
        if reverse_direction:
            forward, reverse = reverse, forward
        return self._raw_masked_logodds_from_values(float(forward), float(reverse))

    def _raw_masked_logodds_runtime(
        self,
        wt_sequence: str,
        mut_sequence: str,
        position_0idx: int,
        wt_token_index: int,
        mut_token_index: int,
    ) -> torch.Tensor | None:
        if not self.masked_logodds.enabled:
            return None
        wt_aa = AMINO_ACIDS_20[int(wt_token_index)]
        mut_aa = AMINO_ACIDS_20[int(mut_token_index)]
        forward, reverse = self._runtime_masked_logodds_pair(
            wt_sequence, mut_sequence, position_0idx, wt_aa, mut_aa
        )
        return self._raw_masked_logodds_from_values(forward, reverse)

    def _transform_masked_logodds(
        self,
        raw: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.masked_logodds.enabled:
            return None
        if raw is None:
            raise ValueError("masked_logodds is enabled but raw feature is missing")
        feat = raw.to(device=device, dtype=torch.float32)
        mean = self.masked_logodds_mean.to(device=device, dtype=torch.float32)
        std = self.masked_logodds_std.to(
            device=device, dtype=torch.float32
        ).clamp_min(1e-6)
        if self.masked_logodds.transform in {"train_zscore", "train_zscore_clip5"}:
            feat = (feat - mean) / std
            if self.masked_logodds.transform == "train_zscore_clip5":
                feat = feat.clamp(-5.0, 5.0)
        elif self.masked_logodds.transform in {"raw", "none"}:
            pass
        else:  # pragma: no cover - validated at construction
            raise ValueError(
                f"unknown masked_logodds transform {self.masked_logodds.transform!r}"
            )
        return feat.to(dtype=dtype)

    def _apply_masked_logodds_residual(
        self,
        pred: torch.Tensor,
        transformed_logodds: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            not self.masked_logodds.enabled
            or self.masked_logodds.fusion != "residual_prior"
        ):
            return pred
        if transformed_logodds is None:
            raise ValueError("masked_logodds residual requested but feature is missing")
        residual = self.masked_logodds_dropout(transformed_logodds).squeeze(-1)
        return pred + self.masked_logodds_gamma.to(dtype=pred.dtype) * (-residual)

    def _build_feature(
        self,
        H_wt: torch.Tensor,
        H_mut: torch.Tensor,
        position_0idx: int,
        wt_token_index: int,
        mut_token_index: int,
        masked_logodds_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if H_wt.shape != H_mut.shape:
            raise ValueError(
                f"WT/Mut hidden shape mismatch: {tuple(H_wt.shape)} vs {tuple(H_mut.shape)}"
            )
        if H_wt.dim() != 2:
            raise ValueError(f"hidden states must be 2-D [L, D], got {tuple(H_wt.shape)}")
        L, D = H_wt.shape
        if D != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: backbone gave D={D}, head expects {self.hidden_dim}"
            )
        if not (0 <= position_0idx < L):
            raise IndexError(
                f"position_0idx {position_0idx} out of range for L={L}"
            )

        h_wt = H_wt[position_0idx]
        h_mut = H_mut[position_0idx]

        device = h_wt.device
        # Use the same dtype as the hidden state so we don't fight autocast.
        dtype = h_wt.dtype
        oh_wt = _onehot20(wt_token_index, device=device, dtype=dtype)
        oh_mut = _onehot20(mut_token_index, device=device, dtype=dtype)

        parts = [h_wt, h_mut, h_mut - h_wt, h_mut * h_wt, oh_wt, oh_mut]

        if self.use_context:
            k = self.context_window_k
            lo = max(0, position_0idx - k)
            hi = min(L, position_0idx + k + 1)
            local_delta = (H_mut[lo:hi] - H_wt[lo:hi]).mean(dim=0)
            global_delta = H_mut.mean(dim=0) - H_wt.mean(dim=0)
            parts.append(local_delta)
            parts.append(global_delta)

        logodds_feat = self._transform_masked_logodds(
            masked_logodds_raw, device=device, dtype=dtype
        )
        if (
            self.masked_logodds.enabled
            and self.masked_logodds.fusion == "concat"
            and logodds_feat is not None
        ):
            parts.append(self.masked_logodds_dropout(logodds_feat))

        return torch.cat(parts, dim=-1), logodds_feat

    def _build_feature_batch(
        self,
        H_wt: torch.Tensor,
        H_mut: torch.Tensor,
        lengths: torch.Tensor,
        positions_0idx: torch.Tensor,
        wt_token_indices: torch.Tensor,
        mut_token_indices: torch.Tensor,
        masked_logodds_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Vectorized counterpart of ``_build_feature`` for padded batches."""
        if H_wt.shape != H_mut.shape:
            raise ValueError(
                f"WT/Mut hidden shape mismatch: {tuple(H_wt.shape)} vs {tuple(H_mut.shape)}"
            )
        if H_wt.dim() != 3:
            raise ValueError(
                f"hidden states must be 3-D [B, L, D], got {tuple(H_wt.shape)}"
            )
        B, L, D = H_wt.shape
        if D != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: backbone gave D={D}, head expects {self.hidden_dim}"
            )
        if positions_0idx.shape != (B,):
            raise ValueError(
                f"positions_0idx must have shape [{B}], got {tuple(positions_0idx.shape)}"
            )
        if torch.any(positions_0idx < 0) or torch.any(positions_0idx >= lengths):
            raise IndexError("one or more mutation positions are outside sequence lengths")

        batch_idx = torch.arange(B, device=H_wt.device)
        h_wt = H_wt[batch_idx, positions_0idx]
        h_mut = H_mut[batch_idx, positions_0idx]

        dtype = h_wt.dtype
        oh_wt = F.one_hot(wt_token_indices, num_classes=_NUM_AA).to(
            device=H_wt.device, dtype=dtype
        )
        oh_mut = F.one_hot(mut_token_indices, num_classes=_NUM_AA).to(
            device=H_wt.device, dtype=dtype
        )
        parts = [h_wt, h_mut, h_mut - h_wt, h_mut * h_wt, oh_wt, oh_mut]

        if self.use_context:
            diff = H_mut - H_wt
            local_delta = []
            k = self.context_window_k
            for b in range(B):
                pos = int(positions_0idx[b].item())
                length = int(lengths[b].item())
                lo = max(0, pos - k)
                hi = min(length, pos + k + 1)
                local_delta.append(diff[b, lo:hi].mean(dim=0))
            parts.append(torch.stack(local_delta, dim=0))

            residue_idx = torch.arange(L, device=H_wt.device).unsqueeze(0)
            mask = residue_idx < lengths.unsqueeze(1)
            denom = lengths.to(device=H_wt.device, dtype=diff.dtype).clamp_min(1).unsqueeze(1)
            global_delta = (diff * mask.unsqueeze(-1)).sum(dim=1) / denom
            parts.append(global_delta)

        logodds_feat = self._transform_masked_logodds(
            masked_logodds_raw, device=H_wt.device, dtype=dtype
        )
        if (
            self.masked_logodds.enabled
            and self.masked_logodds.fusion == "concat"
            and logodds_feat is not None
        ):
            parts.append(self.masked_logodds_dropout(logodds_feat))

        return torch.cat(parts, dim=-1), logodds_feat

    # ----- raw scoring (used by the antisymmetric wrapper) -------------------

    def score_raw(
        self,
        wt_sequence: str,
        mut_sequence: str,
        position_0idx: int,
        wt_token_index: int,
        mut_token_index: int,
        masked_logodds_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full forward path and return a scalar ddG_hat tensor."""
        if self.masked_logodds.enabled and masked_logodds_raw is None:
            masked_logodds_raw = self._raw_masked_logodds_runtime(
                wt_sequence,
                mut_sequence,
                position_0idx,
                wt_token_index,
                mut_token_index,
            )
        H_wt = self.backbone.forward_residue_hidden_states(wt_sequence)
        H_mut = self.backbone.forward_residue_hidden_states(mut_sequence)
        feat, logodds_feat = self._build_feature(
            H_wt, H_mut, position_0idx, wt_token_index, mut_token_index,
            masked_logodds_raw,
        )
        # Head squeezes the trailing dim; result is a 0-D tensor.
        return self._apply_masked_logodds_residual(self.head(feat), logodds_feat)

    def score_raw_batch(
        self,
        wt_sequences: Sequence[str],
        mut_sequences: Sequence[str],
        positions_0idx: torch.Tensor,
        wt_token_indices: torch.Tensor,
        mut_token_indices: torch.Tensor,
        masked_logodds_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(wt_sequences) != len(mut_sequences):
            raise ValueError("wt_sequences and mut_sequences must have the same length")
        if len(wt_sequences) == 0:
            raise ValueError("batch must be non-empty")
        if not hasattr(self.backbone, "forward_residue_hidden_states_batch"):
            raise RuntimeError("backbone does not support batched hidden-state forward")
        if self.masked_logodds.enabled and masked_logodds_raw is None:
            masked_logodds_raw = torch.stack(
                [
                    self._raw_masked_logodds_runtime(
                        wt_seq,
                        mut_seq,
                        int(pos),
                        int(wt_idx),
                        int(mut_idx),
                    )
                    for wt_seq, mut_seq, pos, wt_idx, mut_idx in zip(
                        wt_sequences,
                        mut_sequences,
                        positions_0idx.tolist(),
                        wt_token_indices.tolist(),
                        mut_token_indices.tolist(),
                        strict=True,
                    )
                ],
                dim=0,
            )
        H_wt, lengths_wt = self.backbone.forward_residue_hidden_states_batch(wt_sequences)
        H_mut, lengths_mut = self.backbone.forward_residue_hidden_states_batch(mut_sequences)
        if not torch.equal(lengths_wt, lengths_mut):
            raise ValueError("WT and mutant sequence lengths differ within batch")
        positions_0idx = positions_0idx.to(device=H_wt.device, dtype=torch.long)
        wt_token_indices = wt_token_indices.to(device=H_wt.device, dtype=torch.long)
        mut_token_indices = mut_token_indices.to(device=H_wt.device, dtype=torch.long)
        feat, logodds_feat = self._build_feature_batch(
            H_wt,
            H_mut,
            lengths_wt,
            positions_0idx,
            wt_token_indices,
            mut_token_indices,
            masked_logodds_raw,
        )
        return self._apply_masked_logodds_residual(self.head(feat), logodds_feat)

    def predict_one(self, example: RegressionExample) -> torch.Tensor:
        masked_logodds_raw = self._raw_masked_logodds_from_example(example)
        return self.score_raw(
            example.wt_sequence,
            example.mut_sequence,
            example.position_0idx,
            example.wt_token_index,
            example.mut_token_index,
            masked_logodds_raw,
        )

    def predict_batch(self, examples: Sequence[RegressionExample]) -> torch.Tensor:
        if not examples:
            raise ValueError("examples must be non-empty")
        if not hasattr(self.backbone, "forward_residue_hidden_states_batch"):
            return torch.stack([self.predict_one(ex) for ex in examples])
        masked_logodds_raw = None
        if self.masked_logodds.enabled:
            masked_logodds_raw = torch.stack(
                [self._raw_masked_logodds_from_example(ex) for ex in examples],
                dim=0,
            )
        return self.score_raw_batch(
            [ex.wt_sequence for ex in examples],
            [ex.mut_sequence for ex in examples],
            torch.tensor([ex.position_0idx for ex in examples], dtype=torch.long),
            torch.tensor([ex.wt_token_index for ex in examples], dtype=torch.long),
            torch.tensor([ex.mut_token_index for ex in examples], dtype=torch.long),
            masked_logodds_raw,
        )

REGRESSOR_KINDS = ("siamese_delta",)


def build_regressor(
    kind: str,
    *,
    backbone: _HiddenStateBackbone,
    hidden_dim: int,
    head_dropout: float = 0.1,
    head_hidden_dims: Sequence[int] = (512, 128),
    use_context: bool = False,
    context_window_k: int = 5,
    masked_logodds: dict | MaskedLogOddsConfig | None = None,
) -> nn.Module:
    """Instantiate the checkpoint-compatible Siamese-delta regressor."""
    if kind == "siamese_delta":
        return SiameseDeltaRegressor(
            backbone=backbone,
            hidden_dim=hidden_dim,
            head_dropout=head_dropout,
            head_hidden_dims=head_hidden_dims,
            use_context=use_context,
            context_window_k=context_window_k,
            masked_logodds=masked_logodds,
        )
    raise ValueError(
        f"unknown regressor kind {kind!r}; expected one of {REGRESSOR_KINDS}"
    )
