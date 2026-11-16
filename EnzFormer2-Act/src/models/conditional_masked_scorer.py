"""ESM-C masked-site scorer with LoRA adaptation for EnzFormer2-Act."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .esmc_backbone import ESMCBackboneWrapper
from .lora import inject_lora_adapters


def _iter_parameters(module: Any) -> Iterator[nn.Parameter]:
    if isinstance(module, nn.Module):
        yield from module.parameters()
        return
    model = getattr(module, "model", None)
    if isinstance(model, nn.Module):
        yield from model.parameters()


def _backbone_module(backbone: Any) -> nn.Module:
    model = getattr(backbone, "model", None)
    if not isinstance(model, nn.Module):
        raise ValueError("The ESM-C backbone must expose its torch module as .model.")
    return model


class ESMCMaskedConditionalScorer(nn.Module):
    """Run masked ESM-C forwards with trainable LoRA adapters only."""

    def __init__(
        self,
        *,
        backbone_model_name: str = "esmc_600m",
        freeze_backbone: bool = True,
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("layernorm_qkv", "out_proj"),
        backbone: Any | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone or ESMCBackboneWrapper.from_pretrained(backbone_model_name)
        self.backbone_model = _backbone_module(self.backbone)
        self.backbone_model_name = str(backbone_model_name)
        self.freeze_backbone = bool(freeze_backbone)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = tuple(str(value) for value in lora_target_modules if str(value))
        if self.freeze_backbone:
            for parameter in _iter_parameters(self.backbone):
                parameter.requires_grad_(False)
        injected = inject_lora_adapters(
            _backbone_module(self.backbone),
            target_patterns=self.lora_target_modules,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        self._injected_lora_modules = tuple(injected)
        if self.freeze_backbone:
            for name, parameter in self.named_parameters():
                if ".lora_A" not in name and ".lora_B" not in name:
                    parameter.requires_grad_(False)

    def injected_lora_module_names(self) -> tuple[str, ...]:
        return self._injected_lora_modules

    def _resolve_wt_token_ids(self, batch: Mapping[str, Any]) -> Tensor | None:
        wt_token_ids = batch.get("wt_token_ids")
        has_wt_token_ids = batch.get("has_wt_token_ids")
        if not isinstance(wt_token_ids, Tensor):
            return None
        if isinstance(has_wt_token_ids, Tensor) and not bool(has_wt_token_ids.all().item()):
            return None
        return wt_token_ids

    def forward_from_backbone_outputs(
        self,
        backbone_outputs: Mapping[str, Tensor | None],
        *,
        target_positions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Expose base ESM-C log probabilities under the Act output contract."""

        logits = backbone_outputs.get("target_logits")
        log_probs = backbone_outputs.get("target_log_probs")
        wt_ids = backbone_outputs.get("wt_token_ids")
        if logits is None or log_probs is None or wt_ids is None:
            raise ValueError(
                "Masked backbone output lacks logits, log probabilities, or WT token ids."
            )
        if target_positions is None:
            target_positions = backbone_outputs.get("target_positions")
        if not isinstance(target_positions, Tensor):
            raise ValueError("Masked backbone output lacks target positions.")
        logits = logits.float()
        log_probs = log_probs.float()
        wt_ids = wt_ids.to(dtype=torch.long)
        result: dict[str, Tensor] = {
            "target_positions": target_positions,
            "base_target_logits": logits,
            "base_target_log_probs": log_probs,
            "target_wt_token_ids": wt_ids,
            "common_target_logits": logits,
            "common_target_log_probs": log_probs,
            "target_logits": logits,
            "target_log_probs": log_probs,
        }
        for key in ("target_valid_mask", "total_masked_positions_per_seq", "total_context_mask"):
            value = backbone_outputs.get(key)
            if isinstance(value, Tensor):
                result[key] = value
        return result

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        required = {"sequence", "valid_mask", "target_positions"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"forward is missing required batch keys: {missing}")
        valid_mask = batch["valid_mask"]
        target_positions = batch["target_positions"]
        if not isinstance(valid_mask, Tensor) or not isinstance(target_positions, Tensor):
            raise TypeError("valid_mask and target_positions must be tensors.")
        outputs = self.backbone.forward_masked_positions(
            sequences=batch["sequence"],
            target_positions=target_positions,
            valid_mask=valid_mask,
            wt_token_ids=self._resolve_wt_token_ids(batch),
            target_valid_mask=batch.get("target_valid_mask"),
            context_mask_policy=str(batch.get("context_mask_policy", "target_only")),
            context_mask_radius=int(batch.get("context_mask_radius", 0)),
            extra_random_mask_rate=float(batch.get("extra_random_mask_rate", 0.0)),
            masking_generator=(
                batch.get("masking_generator")
                if isinstance(batch.get("masking_generator"), torch.Generator)
                else None
            ),
            share_context_across_targets=bool(batch.get("share_context_across_targets", False)),
            return_log_probs=True,
            max_masked_rows_per_call=(
                None
                if batch.get("backbone_max_masked_rows_per_call") is None
                else int(batch["backbone_max_masked_rows_per_call"])
            ),
        )
        return self.forward_from_backbone_outputs(outputs, target_positions=target_positions)


__all__ = ["ESMCMaskedConditionalScorer"]
