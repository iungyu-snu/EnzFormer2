"""Target-position sampling for raw-sequence residue masking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(slots=True)
class SamplingStats:
    """Statistics for one target-sampling call."""

    avg_valid_sites: float
    avg_sampled_targets_per_seq: float = 0.0


@dataclass(slots=True)
class SampledTargetBatch:
    """Sampled target positions plus validity metadata for variable-count batches."""

    positions: Tensor
    target_valid_mask: Tensor
    effective_targets_per_seq: Tensor
    stats: SamplingStats


def sample_uniform_positions(
    valid_positions: Tensor,
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
    replacement: bool | None = None,
) -> Tensor:
    """Uniformly sample positions from a 1D `valid_positions` tensor."""

    if valid_positions.ndim != 1:
        raise ValueError("valid_positions must be 1D.")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if valid_positions.numel() == 0 and num_samples > 0:
        raise ValueError("valid_positions must be non-empty when requesting samples.")
    if num_samples == 0:
        return valid_positions.new_empty((0,), dtype=torch.long)

    if replacement is None:
        replacement = valid_positions.numel() < num_samples

    if replacement:
        indices = torch.randint(
            low=0,
            high=valid_positions.numel(),
            size=(num_samples,),
            generator=generator,
        )
        return valid_positions[indices]

    permutation = torch.randperm(valid_positions.numel(), generator=generator)
    return valid_positions[permutation[:num_samples]]


class TargetPositionSampler:
    """Sample fixed-count target residues from the valid raw residue axis."""

    def __init__(
        self,
        num_targets_per_seq: int | None,
        *,
        target_count_mode: str = "fixed_count",
        target_fraction: float | None = None,
        max_targets_per_seq: int | None = None,
        training: bool,
        base_seed: int = 0,
        deterministic_site_sampling: bool = False,
        prioritize_known_mutant_position: bool = False,
        known_mutant_position_boost: float = 4.0,
    ) -> None:
        if target_count_mode not in {"fixed_count", "fraction"}:
            raise ValueError("target_count_mode must be 'fixed_count' or 'fraction'.")
        if target_count_mode == "fixed_count":
            if num_targets_per_seq is None or num_targets_per_seq <= 0:
                raise ValueError("num_targets_per_seq must be positive in fixed_count mode.")
        else:
            if target_fraction is None:
                raise ValueError("target_fraction is required in fraction mode.")
            if not 0.0 < float(target_fraction) <= 1.0:
                raise ValueError("target_fraction must be in (0, 1].")
            if max_targets_per_seq is not None and int(max_targets_per_seq) <= 0:
                raise ValueError("max_targets_per_seq must be positive when provided.")
        if known_mutant_position_boost < 0.0:
            raise ValueError("known_mutant_position_boost must be non-negative.")

        self.num_targets_per_seq = (
            int(num_targets_per_seq)
            if num_targets_per_seq is not None
            else (int(max_targets_per_seq) if max_targets_per_seq is not None else 1)
        )
        self.target_count_mode = str(target_count_mode)
        self.target_fraction = None if target_fraction is None else float(target_fraction)
        self.max_targets_per_seq = None if max_targets_per_seq is None else int(max_targets_per_seq)
        self.training = training
        self.base_seed = base_seed
        self.deterministic_site_sampling = deterministic_site_sampling
        self.prioritize_known_mutant_position = prioritize_known_mutant_position
        self.known_mutant_position_boost = known_mutant_position_boost
        self.last_stats: SamplingStats | None = None

    def resolve_effective_target_count(self, *, seq_len: int) -> int:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if self.target_count_mode == "fixed_count":
            return int(self.num_targets_per_seq)

        assert self.target_fraction is not None
        effective_targets = round(seq_len * self.target_fraction)
        effective_targets = max(1, int(effective_targets))
        if self.max_targets_per_seq is not None:
            effective_targets = min(effective_targets, int(self.max_targets_per_seq))
        return effective_targets

    def sample_with_metadata(
        self,
        valid_mask: Tensor,
        *,
        seq_ids: Sequence[str] | None = None,
        priority_positions: Tensor | None = None,
        has_priority_positions: Tensor | None = None,
        deterministic: bool | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> SampledTargetBatch:
        """Return sampled targets plus validity metadata for variable-count batches."""

        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must have shape `[B, L]`, got {tuple(valid_mask.shape)}.")
        if priority_positions is not None and tuple(priority_positions.shape) != (
            valid_mask.shape[0],
        ):
            raise ValueError("priority_positions must have shape `[B]`.")
        if has_priority_positions is not None and tuple(has_priority_positions.shape) != (
            valid_mask.shape[0],
        ):
            raise ValueError("has_priority_positions must have shape `[B]`.")
        if seq_ids is not None and len(seq_ids) != valid_mask.shape[0]:
            raise ValueError("seq_ids length must match batch size.")

        effective_deterministic = self._resolve_deterministic(deterministic)
        effective_seed = self.base_seed if seed is None else int(seed)

        batch_size, _ = valid_mask.shape
        effective_counts: list[int] = []
        valid_positions_by_batch: list[Tensor] = []
        priority_positions_by_batch: list[int | None] = []
        total_valid_sites = 0.0

        for batch_index in range(batch_size):
            valid_positions = torch.nonzero(valid_mask[batch_index], as_tuple=False).squeeze(-1)
            if valid_positions.numel() == 0:
                raise ValueError(f"Sequence at batch index {batch_index} has no valid residues.")
            valid_positions_by_batch.append(valid_positions)
            effective_counts.append(
                self.resolve_effective_target_count(seq_len=int(valid_positions.numel()))
            )
            total_valid_sites += float(valid_positions.numel())

            priority_position: int | None = None
            if priority_positions is not None:
                has_priority = True
                if has_priority_positions is not None:
                    has_priority = bool(has_priority_positions[batch_index].item())
                if has_priority:
                    candidate_position = int(priority_positions[batch_index].item())
                    if 0 <= candidate_position < valid_mask.shape[1] and bool(
                        valid_mask[batch_index, candidate_position].item()
                    ):
                        priority_position = candidate_position
            priority_positions_by_batch.append(priority_position)

        max_effective_targets = max(effective_counts)
        sampled_positions = torch.empty(
            (batch_size, max_effective_targets), dtype=torch.long, device=valid_mask.device
        )
        target_valid_mask = torch.zeros(
            (batch_size, max_effective_targets), dtype=torch.bool, device=valid_mask.device
        )
        effective_targets_per_seq = torch.tensor(
            effective_counts, dtype=torch.long, device=valid_mask.device
        )

        for batch_index in range(batch_size):
            valid_positions = valid_positions_by_batch[batch_index]
            sample_generator = self._resolve_generator(
                batch_index=batch_index,
                seq_ids=seq_ids,
                deterministic=effective_deterministic,
                seed=effective_seed,
                generator=generator,
            )
            effective_count = effective_counts[batch_index]
            sampled_row = self._sample_sequence(
                valid_positions=valid_positions.cpu(),
                priority_position=priority_positions_by_batch[batch_index],
                generator=sample_generator,
                num_targets=effective_count,
            )
            padded_row = torch.full(
                (max_effective_targets,),
                int(sampled_row[0].item()),
                dtype=torch.long,
                device=valid_mask.device,
            )
            padded_row[:effective_count] = sampled_row.to(valid_mask.device)
            sampled_positions[batch_index] = padded_row
            target_valid_mask[batch_index, :effective_count] = True

        stats = SamplingStats(
            avg_valid_sites=total_valid_sites / batch_size,
            avg_sampled_targets_per_seq=float(sum(effective_counts)) / float(batch_size),
        )
        self.last_stats = stats
        return SampledTargetBatch(
            positions=sampled_positions,
            target_valid_mask=target_valid_mask,
            effective_targets_per_seq=effective_targets_per_seq,
            stats=stats,
        )

    def sample(
        self,
        valid_mask: Tensor,
        *,
        seq_ids: Sequence[str] | None = None,
        priority_positions: Tensor | None = None,
        has_priority_positions: Tensor | None = None,
        deterministic: bool | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SamplingStats]:
        """Return target positions with shape `[B, M]`."""

        sampled = self.sample_with_metadata(
            valid_mask,
            seq_ids=seq_ids,
            priority_positions=priority_positions,
            has_priority_positions=has_priority_positions,
            deterministic=deterministic,
            seed=seed,
            generator=generator,
        )
        if return_stats:
            return sampled.positions, sampled.stats
        return sampled.positions

    def _sample_sequence(
        self,
        *,
        valid_positions: Tensor,
        priority_position: int | None,
        generator: torch.Generator | None,
        num_targets: int,
    ) -> Tensor:
        if not self.prioritize_known_mutant_position or priority_position is None:
            return sample_uniform_positions(
                valid_positions,
                num_targets,
                generator=generator,
                replacement=valid_positions.numel() < num_targets,
            )

        priority_mask = valid_positions.eq(int(priority_position))
        if not bool(priority_mask.any().item()):
            return sample_uniform_positions(
                valid_positions,
                num_targets,
                generator=generator,
                replacement=valid_positions.numel() < num_targets,
            )

        priority_positions = valid_positions[priority_mask][:1]
        remaining_count = max(0, num_targets - int(priority_positions.numel()))
        if remaining_count == 0:
            return priority_positions

        remaining_positions = valid_positions[~priority_mask]
        if remaining_positions.numel() == 0:
            uniform_positions = sample_uniform_positions(
                valid_positions,
                remaining_count,
                generator=generator,
                replacement=True,
            )
        else:
            uniform_positions = sample_uniform_positions(
                remaining_positions,
                remaining_count,
                generator=generator,
                replacement=remaining_positions.numel() < remaining_count,
            )
        combined = torch.cat((priority_positions, uniform_positions), dim=0)
        if combined.numel() <= 1:
            return combined
        shuffle_order = torch.randperm(combined.numel(), generator=generator)
        return combined[shuffle_order]

    def _resolve_deterministic(self, deterministic: bool | None) -> bool:
        if deterministic is not None:
            return deterministic
        if self.deterministic_site_sampling:
            return True
        return not self.training

    def _resolve_generator(
        self,
        *,
        batch_index: int,
        seq_ids: Sequence[str] | None,
        deterministic: bool,
        seed: int,
        generator: torch.Generator | None,
    ) -> torch.Generator | None:
        if not deterministic:
            return generator

        seq_id = str(batch_index) if seq_ids is None else str(seq_ids[batch_index])
        deterministic_generator = torch.Generator(device="cpu")
        deterministic_generator.manual_seed(self._seed_for(seq_id, seed))
        return deterministic_generator

    def _seed_for(self, seq_id: str, seed: int) -> int:
        payload = f"{seed}:{seq_id}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % (2**63 - 1)

    def __call__(
        self,
        valid_mask: Tensor,
        *,
        seq_ids: Sequence[str] | None = None,
        priority_positions: Tensor | None = None,
        has_priority_positions: Tensor | None = None,
        deterministic: bool | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SamplingStats]:
        """Alias for :meth:`sample`."""

        return self.sample(
            valid_mask,
            seq_ids=seq_ids,
            priority_positions=priority_positions,
            has_priority_positions=has_priority_positions,
            deterministic=deterministic,
            seed=seed,
            generator=generator,
            return_stats=return_stats,
        )
