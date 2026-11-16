"""Utilities for raw-sequence ESMC tokenization and masked forwards."""

from __future__ import annotations

import inspect
import os
from typing import Any, Sequence

import torch
from torch import Tensor

from enzformer2.data.corruption import build_context_mask

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_esmc_class() -> type[Any]:
    """Load ESMC from the pinned external ``esm`` package."""

    try:
        from esm.models.esmc import ESMC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the pinned ESMC dependency. Install it in the "
            "active environment with: python -m pip install -e ."
        ) from exc

    return ESMC


def token_to_id(tokenizer: Any, token: str) -> int:
    """Convert one native tokenizer token into its integer id."""

    token_id = tokenizer.convert_tokens_to_ids(str(token).upper())
    if not isinstance(token_id, int):
        raise TypeError(f"Expected integer token id for {token!r}, got {type(token_id)!r}.")
    return token_id


def encode_residue_token_ids(sequence: str, tokenizer: Any) -> Tensor:
    """Encode a raw sequence into native tokenizer ids without special tokens."""

    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError("A non-empty amino-acid sequence is required.")
    return torch.tensor(
        [token_to_id(tokenizer, residue) for residue in normalized], dtype=torch.long
    )


def _as_batched_positions(target_positions: Tensor) -> tuple[Tensor, bool]:
    if target_positions.ndim == 1:
        return target_positions.view(1, -1).to(dtype=torch.long), True
    if target_positions.ndim == 2:
        return target_positions.to(dtype=torch.long), False
    raise ValueError(
        f"target_positions must have shape [M] or [B, M], got {tuple(target_positions.shape)}."
    )


def _as_batched_sequence_tokens(sequence_tokens: Tensor) -> tuple[Tensor, bool]:
    if sequence_tokens.ndim == 1:
        return sequence_tokens.view(1, -1).to(dtype=torch.long), True
    if sequence_tokens.ndim == 2:
        return sequence_tokens.to(dtype=torch.long), False
    raise ValueError(
        f"sequence_tokens must have shape [L+2] or [B, L+2], got {tuple(sequence_tokens.shape)}."
    )


def _as_batched_valid_mask(valid_mask: Tensor) -> tuple[Tensor, bool]:
    if valid_mask.ndim == 1:
        return valid_mask.view(1, -1).to(dtype=torch.bool), True
    if valid_mask.ndim == 2:
        return valid_mask.to(dtype=torch.bool), False
    raise ValueError(f"valid_mask must have shape [L] or [B, L], got {tuple(valid_mask.shape)}.")


def _as_batched_target_valid_mask(target_valid_mask: Tensor) -> tuple[Tensor, bool]:
    if target_valid_mask.ndim == 1:
        return target_valid_mask.view(1, -1).to(dtype=torch.bool), True
    if target_valid_mask.ndim == 2:
        return target_valid_mask.to(dtype=torch.bool), False
    raise ValueError(
        f"target_valid_mask must have shape [M] or [B, M], got {tuple(target_valid_mask.shape)}."
    )


def raw_positions_to_token_positions(
    target_positions: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Convert raw 0-based residue positions to ESMC token positions via `+1`."""

    positions_2d, squeezed = _as_batched_positions(target_positions)
    if bool(positions_2d.lt(0).any()):
        raise ValueError("target_positions must be non-negative raw residue indices.")

    if valid_mask is not None:
        valid_mask_2d, _ = _as_batched_valid_mask(valid_mask)
        if valid_mask_2d.shape[0] != positions_2d.shape[0]:
            raise ValueError("Batch size mismatch between target_positions and valid_mask.")
        if valid_mask_2d.shape[1] == 0:
            raise ValueError("valid_mask must include at least one residue position.")
        if int(positions_2d.max().item()) >= valid_mask_2d.shape[1]:
            raise ValueError("target_positions contains padded or invalid residue indices.")
        gathered_valid = valid_mask_2d.gather(1, positions_2d.to(device=valid_mask_2d.device))
        if not bool(gathered_valid.all()):
            raise ValueError("target_positions contains padded or invalid residue indices.")

    token_positions = positions_2d + 1
    return token_positions.view(-1) if squeezed else token_positions


def build_masked_token_batch(
    sequence_tokens: Tensor,
    target_positions: Tensor,
    mask_token_id: int,
    *,
    valid_mask: Tensor | None = None,
    target_valid_mask: Tensor | None = None,
    context_mask_policy: str = "target_only",
    context_mask_radius: int = 0,
    extra_random_mask_rate: float = 0.0,
    masking_generator: torch.Generator | None = None,
    share_context_across_targets: bool = False,
) -> dict[str, Tensor]:
    """Build one masked variant per requested raw residue position."""

    sequence_tokens_2d, _ = _as_batched_sequence_tokens(sequence_tokens)
    positions_2d, _ = _as_batched_positions(target_positions)
    if sequence_tokens_2d.shape[0] != positions_2d.shape[0]:
        raise ValueError("Batch size mismatch between sequence_tokens and target_positions.")

    if valid_mask is not None:
        valid_mask_2d, _ = _as_batched_valid_mask(valid_mask)
        if valid_mask_2d.shape[0] != sequence_tokens_2d.shape[0]:
            raise ValueError("Batch size mismatch between sequence_tokens and valid_mask.")
        expected_token_width = valid_mask_2d.shape[1] + 2
        if sequence_tokens_2d.shape[1] < expected_token_width:
            raise ValueError(
                f"sequence_tokens width {sequence_tokens_2d.shape[1]} is too small for residue width {valid_mask_2d.shape[1]}."
            )
    else:
        valid_mask_2d = None
    if target_valid_mask is None:
        target_valid_mask_2d = torch.ones_like(positions_2d, dtype=torch.bool)
    else:
        target_valid_mask_2d, _ = _as_batched_target_valid_mask(target_valid_mask)
        if target_valid_mask_2d.shape != positions_2d.shape:
            raise ValueError("target_valid_mask must have the same shape as target_positions.")

    token_positions = raw_positions_to_token_positions(positions_2d, valid_mask=valid_mask_2d)
    batch_size, num_targets = positions_2d.shape
    masked_tokens = (
        sequence_tokens_2d.unsqueeze(1)
        .expand(-1, num_targets, -1)
        .reshape(batch_size * num_targets, -1)
        .clone()
    )
    flat_token_positions = token_positions.reshape(-1).to(device=masked_tokens.device)
    flat_batch_indices = (
        torch.arange(batch_size, device=masked_tokens.device)
        .view(-1, 1)
        .expand(-1, num_targets)
        .reshape(-1)
    )
    flat_target_positions = positions_2d.reshape(-1).to(device=masked_tokens.device)

    if valid_mask_2d is None:
        raw_width = sequence_tokens_2d.shape[1] - 2
        valid_mask_2d = torch.ones(
            (batch_size, raw_width),
            dtype=torch.bool,
            device=positions_2d.device,
        )
    else:
        raw_width = valid_mask_2d.shape[1]

    if share_context_across_targets:
        context_masks = build_context_mask(
            valid_mask_2d.to(device=positions_2d.device),
            positions_2d.to(device=positions_2d.device),
            target_valid_mask=target_valid_mask_2d.to(device=positions_2d.device),
            context_mask_policy=context_mask_policy,
            context_mask_radius=context_mask_radius,
            extra_random_mask_rate=extra_random_mask_rate,
            generator=masking_generator,
        )
        total_context_mask = context_masks["total_mask"].to(device=masked_tokens.device)
        flat_context_mask = (
            total_context_mask.unsqueeze(1)
            .expand(-1, num_targets, -1)
            .reshape(batch_size * num_targets, raw_width)
        )
        total_masked_positions_per_seq = context_masks["total_masked_positions_per_seq"]
    else:
        expanded_valid_mask = (
            valid_mask_2d.unsqueeze(1)
            .expand(-1, num_targets, -1)
            .reshape(batch_size * num_targets, raw_width)
        )
        flat_positions = positions_2d.reshape(-1, 1).to(device=positions_2d.device)
        flat_target_valid_mask = target_valid_mask_2d.reshape(-1, 1).to(device=positions_2d.device)
        flat_context_masks = build_context_mask(
            expanded_valid_mask.to(device=positions_2d.device),
            flat_positions,
            target_valid_mask=flat_target_valid_mask,
            context_mask_policy=context_mask_policy,
            context_mask_radius=context_mask_radius,
            extra_random_mask_rate=extra_random_mask_rate,
            generator=masking_generator,
        )
        flat_context_mask = flat_context_masks["total_mask"].to(device=masked_tokens.device)
        total_context_mask = flat_context_mask.view(batch_size, num_targets, raw_width).any(dim=1)
        per_target_mask_counts = (
            flat_context_masks["total_masked_positions_per_seq"]
            .to(
                dtype=torch.float32,
                device=masked_tokens.device,
            )
            .view(batch_size, num_targets)
        )
        valid_target_weights = target_valid_mask_2d.to(
            dtype=torch.float32, device=masked_tokens.device
        )
        total_masked_positions_per_seq = (per_target_mask_counts * valid_target_weights).sum(
            dim=1
        ) / valid_target_weights.sum(dim=1).clamp_min(1.0)
    masked_tokens[:, 1 : 1 + raw_width] = torch.where(
        flat_context_mask,
        torch.full_like(masked_tokens[:, 1 : 1 + raw_width], int(mask_token_id)),
        masked_tokens[:, 1 : 1 + raw_width],
    )
    return {
        "masked_tokens": masked_tokens,
        "token_positions": token_positions,
        "flat_token_positions": flat_token_positions,
        "flat_batch_indices": flat_batch_indices,
        "flat_target_positions": flat_target_positions,
        "target_valid_mask": target_valid_mask_2d,
        "total_context_mask": total_context_mask,
        "total_masked_positions_per_seq": total_masked_positions_per_seq,
    }


def _build_valid_mask_from_sequences(sequences: Sequence[str]) -> Tensor:
    if not sequences:
        raise ValueError("At least one sequence is required.")
    max_length = max(len(sequence) for sequence in sequences)
    valid_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for batch_index, sequence in enumerate(sequences):
        valid_mask[batch_index, : len(sequence)] = True
    return valid_mask


def _resolve_num_hidden_layers(model: Any) -> int | None:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        return None
    with torch.no_grad():
        try:
            return int(len(blocks))
        except TypeError:
            return None


def _normalize_hidden_state_indices(
    hidden_state_indices: Sequence[int],
    *,
    num_hidden_layers: int,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for raw_index in hidden_state_indices:
        layer_index = int(raw_index)
        if layer_index < 0:
            layer_index += num_hidden_layers
        if layer_index < 0 or layer_index >= num_hidden_layers:
            raise ValueError(
                f"hidden_state_indices contains invalid layer {raw_index!r} for {num_hidden_layers} layers."
            )
        if layer_index not in normalized:
            normalized.append(layer_index)
    return tuple(normalized)


def _model_supports_hidden_state_indices(model: Any) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    return "hidden_state_indices" in signature.parameters


class ESMCBackboneWrapper:
    """Thin wrapper around ESMC tokenization and masked forwards."""

    def __init__(self, model: Any, *, model_name: str | None = None) -> None:
        self.model = model
        self.tokenizer = model.tokenizer
        self.model_name = None if model_name is None else str(model_name)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "esmc_600m",
        *,
        device: torch.device | None = None,
    ) -> "ESMCBackboneWrapper":
        esmc_class = load_esmc_class()
        model = esmc_class.from_pretrained(model_name, device=device)
        return cls(model, model_name=model_name)

    @property
    def device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device):
            return device
        return next(self.model.parameters()).device

    @property
    def mask_token_id(self) -> int:
        return int(self.tokenizer.mask_token_id)

    def token_id(self, token: str) -> int:
        return token_to_id(self.tokenizer, token)

    def tokenize(self, sequences: Sequence[str]) -> Tensor:
        """Tokenize one or more raw sequences with native ESMC specials."""

        if not sequences:
            raise ValueError("At least one sequence is required.")
        return self.model._tokenize([str(sequence) for sequence in sequences])

    def encode_single_residue_token_ids(self, sequence: str) -> Tensor:
        """Encode one raw sequence into native token ids without special tokens."""

        return encode_residue_token_ids(sequence, self.tokenizer)

    def encode_residue_token_ids(self, sequences: Sequence[str]) -> list[Tensor]:
        """Encode raw sequences into native token ids without special tokens."""

        return [self.encode_single_residue_token_ids(sequence) for sequence in sequences]

    def forward_masked_positions(
        self,
        *,
        target_positions: Tensor,
        sequences: Sequence[str] | None = None,
        sequence_tokens: Tensor | None = None,
        valid_mask: Tensor | None = None,
        wt_token_ids: Tensor | None = None,
        target_valid_mask: Tensor | None = None,
        context_mask_policy: str = "target_only",
        context_mask_radius: int = 0,
        extra_random_mask_rate: float = 0.0,
        masking_generator: torch.Generator | None = None,
        share_context_across_targets: bool = False,
        return_log_probs: bool = True,
        return_embeddings: bool = False,
        return_hidden_states: bool = False,
        hidden_state_indices: Sequence[int] | None = None,
        max_masked_rows_per_call: int | None = None,
    ) -> dict[str, Tensor | None]:
        """Mask requested raw positions, run ESMC, and gather masked-site outputs."""

        if sequence_tokens is None:
            if sequences is None:
                raise ValueError("Provide either sequences or pre-tokenized sequence_tokens.")
            sequence_tokens = self.tokenize(sequences)
        else:
            sequence_tokens, _ = _as_batched_sequence_tokens(sequence_tokens)

        positions_2d, _ = _as_batched_positions(target_positions)
        if sequence_tokens.shape[0] != positions_2d.shape[0]:
            raise ValueError("Batch size mismatch between sequence_tokens and target_positions.")

        if valid_mask is None:
            if sequences is None:
                raise ValueError("valid_mask is required when sequences are not provided.")
            valid_mask = _build_valid_mask_from_sequences(sequences)
        valid_mask_2d, _ = _as_batched_valid_mask(valid_mask)

        if valid_mask_2d.shape[0] != sequence_tokens.shape[0]:
            raise ValueError("Batch size mismatch between sequence_tokens and valid_mask.")

        masked_batch = build_masked_token_batch(
            sequence_tokens,
            positions_2d,
            self.mask_token_id,
            valid_mask=valid_mask_2d,
            target_valid_mask=target_valid_mask,
            context_mask_policy=context_mask_policy,
            context_mask_radius=context_mask_radius,
            extra_random_mask_rate=extra_random_mask_rate,
            masking_generator=masking_generator,
            share_context_across_targets=share_context_across_targets,
        )
        masked_tokens = masked_batch["masked_tokens"].to(device=self.device)
        flat_token_positions = masked_batch["flat_token_positions"].to(device=self.device)

        resolved_num_hidden_layers = _resolve_num_hidden_layers(self.model)
        resolved_hidden_state_indices: tuple[int, ...] | None = None
        model_kwargs: dict[str, Any] = {}
        used_hidden_state_subset = False
        if return_hidden_states and hidden_state_indices is not None:
            if resolved_num_hidden_layers is None:
                seen_indices: list[int] = []
                for index in hidden_state_indices:
                    resolved_index = int(index)
                    if resolved_index not in seen_indices:
                        seen_indices.append(resolved_index)
                resolved_hidden_state_indices = tuple(seen_indices)
            else:
                resolved_hidden_state_indices = _normalize_hidden_state_indices(
                    hidden_state_indices,
                    num_hidden_layers=resolved_num_hidden_layers,
                )
            if _model_supports_hidden_state_indices(self.model):
                model_kwargs["hidden_state_indices"] = resolved_hidden_state_indices
                used_hidden_state_subset = True

        resolved_chunk_size = (
            None if max_masked_rows_per_call is None else int(max_masked_rows_per_call)
        )
        if resolved_chunk_size is not None and resolved_chunk_size <= 0:
            resolved_chunk_size = None

        flat_target_logits_chunks: list[Tensor] = []
        flat_target_embeddings_chunks: list[Tensor] = []
        gathered_hidden_state_chunks: list[Tensor] = []
        num_masked_rows = int(masked_tokens.shape[0])
        if resolved_chunk_size is None or num_masked_rows <= resolved_chunk_size:
            chunk_ranges = ((0, num_masked_rows),)
        else:
            chunk_ranges = tuple(
                (start, min(start + resolved_chunk_size, num_masked_rows))
                for start in range(0, num_masked_rows, resolved_chunk_size)
            )

        for start, end in chunk_ranges:
            outputs = self.model(sequence_tokens=masked_tokens[start:end], **model_kwargs)
            sequence_logits = outputs.sequence_logits
            chunk_token_positions = flat_token_positions[start:end].to(
                device=sequence_logits.device
            )
            chunk_row_indices = torch.arange(
                sequence_logits.shape[0], device=sequence_logits.device
            )
            flat_target_logits_chunks.append(
                sequence_logits[chunk_row_indices, chunk_token_positions]
            )

            if return_embeddings:
                embeddings = getattr(outputs, "embeddings", None)
                if embeddings is not None:
                    flat_target_embeddings_chunks.append(
                        embeddings[chunk_row_indices, chunk_token_positions]
                    )

            if return_hidden_states:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    if resolved_num_hidden_layers is None:
                        resolved_num_hidden_layers = int(hidden_states.shape[0])
                    gathered_hidden_state_chunks.append(
                        hidden_states[:, chunk_row_indices, chunk_token_positions].permute(1, 0, 2)
                    )

        flat_target_logits = (
            flat_target_logits_chunks[0]
            if len(flat_target_logits_chunks) == 1
            else torch.cat(flat_target_logits_chunks, dim=0)
        )

        batch_size, num_targets = positions_2d.shape
        target_logits = flat_target_logits.reshape(batch_size, num_targets, -1)

        target_log_probs = None
        if return_log_probs:
            target_log_probs = torch.log_softmax(flat_target_logits.float(), dim=-1).reshape(
                batch_size, num_targets, -1
            )

        target_embeddings = None
        if return_embeddings and flat_target_embeddings_chunks:
            gathered_embeddings = (
                flat_target_embeddings_chunks[0]
                if len(flat_target_embeddings_chunks) == 1
                else torch.cat(flat_target_embeddings_chunks, dim=0)
            )
            target_embeddings = gathered_embeddings.reshape(batch_size, num_targets, -1)

        target_hidden_states = None
        if return_hidden_states and gathered_hidden_state_chunks:
            gathered_hidden_states = (
                gathered_hidden_state_chunks[0]
                if len(gathered_hidden_state_chunks) == 1
                else torch.cat(gathered_hidden_state_chunks, dim=0)
            )
            target_hidden_states = gathered_hidden_states.reshape(
                batch_size,
                num_targets,
                *gathered_hidden_states.shape[1:],
            )

        if wt_token_ids is not None:
            wt_token_ids_2d, _ = _as_batched_positions(wt_token_ids)
            if wt_token_ids_2d.shape[0] != batch_size:
                raise ValueError("Batch size mismatch between wt_token_ids and target_positions.")
            gathered_wt_token_ids = wt_token_ids_2d.to(device=sequence_logits.device).gather(
                1,
                positions_2d.to(device=sequence_logits.device),
            )
        else:
            token_positions = masked_batch["token_positions"].to(device=sequence_tokens.device)
            gathered_wt_token_ids = sequence_tokens.gather(1, token_positions)
        target_valid_mask_2d = masked_batch["target_valid_mask"].to(
            device=gathered_wt_token_ids.device
        )
        gathered_wt_token_ids = torch.where(
            target_valid_mask_2d,
            gathered_wt_token_ids.to(dtype=torch.long),
            torch.full_like(gathered_wt_token_ids, -100, dtype=torch.long),
        )

        return {
            "masked_tokens": masked_tokens,
            "token_positions": masked_batch["token_positions"],
            "target_positions": positions_2d,
            "target_logits": target_logits,
            "target_log_probs": target_log_probs,
            "target_embeddings": target_embeddings,
            "target_hidden_states": target_hidden_states,
            "hidden_state_indices": resolved_hidden_state_indices
            if used_hidden_state_subset
            else None,
            "num_hidden_layers": resolved_num_hidden_layers,
            "wt_token_ids": gathered_wt_token_ids.to(dtype=torch.long),
            "target_valid_mask": target_valid_mask_2d,
            "total_context_mask": masked_batch["total_context_mask"],
            "total_masked_positions_per_seq": masked_batch["total_masked_positions_per_seq"],
        }
