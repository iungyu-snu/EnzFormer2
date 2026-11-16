"""Batch collation for raw amino-acid sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


DEFAULT_AA_PAD_VALUE = 0


def collate_raw_sequence_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad sequences and preserve identifiers and optional sample weights."""

    if not samples:
        raise ValueError("Cannot collate an empty sequence batch.")
    sequences = [str(sample["sequence"]).strip().upper() for sample in samples]
    if any(not sequence for sequence in sequences):
        raise ValueError("A sequence batch cannot contain an empty sequence.")
    max_length = max(map(len, sequences))
    valid_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        valid_mask[index, : len(sequence)] = True
    return {
        "sequence": sequences,
        "seq_id": [str(sample["seq_id"]) for sample in samples],
        "valid_mask": valid_mask,
        "sample_weights": torch.tensor(
            [float(sample.get("sample_weight", 1.0)) for sample in samples],
            dtype=torch.float32,
        ),
    }


__all__ = ["DEFAULT_AA_PAD_VALUE", "collate_raw_sequence_batch"]
