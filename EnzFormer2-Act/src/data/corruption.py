"""Build context masks for Act training."""

from __future__ import annotations

import torch
from torch import Tensor


def build_context_mask(
    valid_mask: Tensor,
    target_positions: Tensor,
    *,
    target_valid_mask: Tensor | None = None,
    context_mask_policy: str = "target_only",
    context_mask_radius: int = 0,
    extra_random_mask_rate: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Build per-sequence training-time masking augmentation masks."""

    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [B, L], got {tuple(valid_mask.shape)}.")
    if target_positions.ndim != 2:
        raise ValueError(
            f"target_positions must have shape [B, M], got {tuple(target_positions.shape)}."
        )
    if valid_mask.shape[0] != target_positions.shape[0]:
        raise ValueError("valid_mask and target_positions must have matching batch sizes.")
    if target_valid_mask is None:
        target_valid_mask = torch.ones_like(
            target_positions, dtype=torch.bool, device=target_positions.device
        )
    elif target_valid_mask.shape != target_positions.shape:
        raise ValueError("target_valid_mask must have the same shape as target_positions.")
    else:
        target_valid_mask = target_valid_mask.to(dtype=torch.bool, device=target_positions.device)
    if context_mask_policy not in {"target_only", "radius", "radius_plus_random"}:
        raise ValueError("context_mask_policy must be target_only, radius, or radius_plus_random.")
    if context_mask_radius < 0:
        raise ValueError("context_mask_radius must be non-negative.")
    if not 0.0 <= extra_random_mask_rate <= 1.0:
        raise ValueError("extra_random_mask_rate must be between 0.0 and 1.0.")

    batch_size, seq_len = valid_mask.shape
    if ((target_positions < 0) | (target_positions >= seq_len)).any():
        raise ValueError("target_positions contains an out-of-range index.")
    if not valid_mask.gather(1, target_positions.to(device=valid_mask.device)).all():
        raise ValueError("target_positions must refer only to valid residues.")

    positions = torch.arange(seq_len, device=target_positions.device).view(1, 1, seq_len)
    expanded_target_valid_mask = target_valid_mask.unsqueeze(-1)
    distances = (positions - target_positions.unsqueeze(-1)).abs()

    target_mask = (distances.eq(0) & expanded_target_valid_mask).any(dim=1) & valid_mask
    radius_mask = torch.zeros_like(target_mask)
    if context_mask_policy in {"radius", "radius_plus_random"} and context_mask_radius > 0:
        radius_mask = (distances.le(context_mask_radius) & expanded_target_valid_mask).any(
            dim=1
        ) & valid_mask
    total_mask = target_mask | radius_mask

    extra_random_mask = torch.zeros_like(total_mask)
    if context_mask_policy == "radius_plus_random" and extra_random_mask_rate > 0.0:
        for batch_index in range(batch_size):
            eligible = torch.nonzero(
                valid_mask[batch_index] & ~target_mask[batch_index] & ~total_mask[batch_index],
                as_tuple=False,
            ).squeeze(-1)
            if eligible.numel() == 0:
                continue
            extra_count = int(round(float(eligible.numel()) * extra_random_mask_rate))
            extra_count = min(max(extra_count, 0), int(eligible.numel()))
            if extra_count == 0:
                continue
            permutation = torch.randperm(
                int(eligible.numel()), generator=generator, device=eligible.device
            )
            selected = eligible[permutation[:extra_count]]
            extra_random_mask[batch_index, selected] = True
        total_mask = total_mask | extra_random_mask

    total_mask = total_mask & valid_mask
    total_masked_positions_per_seq = total_mask.sum(dim=1)
    return {
        "target_mask": target_mask,
        "radius_mask": radius_mask & ~target_mask,
        "extra_random_mask": extra_random_mask,
        "total_mask": total_mask,
        "total_masked_positions_per_seq": total_masked_positions_per_seq,
    }
