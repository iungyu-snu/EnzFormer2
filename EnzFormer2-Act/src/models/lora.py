"""Minimal in-repo LoRA adapter for ``nn.Linear`` layers.

This module is self-contained and has no ``peft`` dependency. The ESM-C wrapper
exposes the linear modules needed by the Act configuration. The API provides:

- :class:`LoRALinear` wraps an existing :class:`torch.nn.Linear` with a
  zero-initialized low-rank delta ``B @ A`` scaled by ``alpha / rank``.
- :func:`inject_lora_adapters` walks a module tree, swaps every matching
  ``nn.Linear`` in place, and returns the list of injected fully-qualified names.
- Helpers :func:`iter_lora_parameters` and :func:`is_lora_parameter_name`
  identify the trainable LoRA weights.

Pattern matching is substring-based. ``target_patterns`` must contain at least
one pattern that matches at least one linear; otherwise injection raises.
``exclude_patterns`` is OR'd against the same name and suppresses matches.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


LORA_PARAMETER_SUFFIXES = ("lora_A", "lora_B")


class LoRALinear(nn.Module):
    """Low-rank adapter around an existing :class:`nn.Linear`.

    ``y = base(x) + (alpha / rank) * dropout(x) @ A^T @ B^T``.

    Kaiming initialization on ``A`` and zero initialization on ``B`` make the
    injected adapter an identity transformation at initialization.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)!r}.")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def forward(self, inputs: Tensor) -> Tensor:
        base_out = self.base(inputs)
        dropped = self.lora_dropout(inputs)
        # Cast LoRA arithmetic to the base output's dtype so bf16/fp16 call
        # sites don't silently upcast.
        lora_A = self.lora_A.to(dtype=base_out.dtype)
        lora_B = self.lora_B.to(dtype=base_out.dtype)
        delta = F.linear(F.linear(dropped.to(dtype=base_out.dtype), lora_A), lora_B)
        return base_out + self.scaling * delta


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(pattern and pattern in name for pattern in patterns)


def _resolve_parent_module(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_adapters(
    module: nn.Module,
    *,
    target_patterns: Sequence[str],
    exclude_patterns: Sequence[str] = (),
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every matching :class:`nn.Linear` in ``module`` with a :class:`LoRALinear`.

    Matching is substring-based against the module's dotted name. A linear is
    wrapped iff its name contains at least one ``target_patterns`` entry and
    none of the ``exclude_patterns`` entries. Raises :class:`ValueError` if the
    traversal ends with zero injections.
    """

    if not target_patterns:
        raise ValueError("inject_lora_adapters requires at least one target pattern.")
    target_patterns = [str(pattern) for pattern in target_patterns if pattern]
    exclude_patterns = [str(pattern) for pattern in exclude_patterns if pattern]
    if not target_patterns:
        raise ValueError("inject_lora_adapters requires at least one non-empty target pattern.")

    candidates: list[tuple[str, nn.Linear]] = []
    for name, submodule in module.named_modules():
        if not isinstance(submodule, nn.Linear) or isinstance(submodule, LoRALinear):
            continue
        if not _matches_any(name, target_patterns):
            continue
        if exclude_patterns and _matches_any(name, exclude_patterns):
            continue
        candidates.append((name, submodule))

    if not candidates:
        raise ValueError(
            "inject_lora_adapters matched zero nn.Linear modules. "
            f"target_patterns={list(target_patterns)}, exclude_patterns={list(exclude_patterns)}."
        )

    injected: list[str] = []
    for name, linear in candidates:
        parent, attr = _resolve_parent_module(module, name)
        wrapped = LoRALinear(
            linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, attr, wrapped)
        injected.append(name)
    return injected


def is_lora_parameter_name(name: str) -> bool:
    """Return True if ``name`` refers to a LoRA parameter (``.lora_A`` / ``.lora_B``)."""

    return any(
        name.endswith(suffix) or f".{suffix}." in name or name.endswith(f".{suffix}")
        for suffix in LORA_PARAMETER_SUFFIXES
    )


def iter_lora_parameters(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield ``(qualified_name, parameter)`` for every LoRA parameter in ``module``."""

    for name, parameter in module.named_parameters():
        if is_lora_parameter_name(name):
            yield name, parameter


__all__ = [
    "LoRALinear",
    "LORA_PARAMETER_SUFFIXES",
    "inject_lora_adapters",
    "is_lora_parameter_name",
    "iter_lora_parameters",
]
