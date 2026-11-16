"""Custom LoRA injection for arbitrary `nn.Linear` modules.

Target modules are matched by qualified-name substring and each matched
`nn.Linear` is wrapped with LoRA. Only the A/B matrices are trainable;
original weights and bias remain frozen and unchanged.

ESMC 600M layer names:
  transformer.blocks.{i}.attn.layernorm_qkv.1   (1152 -> 3456, fused QKV)
  transformer.blocks.{i}.attn.out_proj          (1152 -> 1152)
  transformer.blocks.{i}.ffn.1                  (1152 -> 6144)
  transformer.blocks.{i}.ffn.3                  (3072 -> 1152)

The default LoRA target patterns therefore include "layernorm_qkv" and
"out_proj"; FFN modules are opt-in.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_module_patterns: tuple[str, ...] = field(default_factory=tuple)
    init: str = "kaiming"   # "kaiming" or "zeros" for A; B is always zeros
    fail_if_no_match: bool = True

    def scaling(self) -> float:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        return self.alpha / self.rank


@dataclass
class InjectionReport:
    wrapped: list[str]
    skipped_total_linear: int
    available_sample: list[str]


@dataclass
class TrainableParamReport:
    total: int
    trainable: int
    lora_modules: int

    @property
    def trainable_ratio(self) -> float:
        return self.trainable / max(1, self.total)


def _match_any(qualified_name: str, patterns: Iterable[str]) -> bool:
    return any(p in qualified_name for p in patterns)


def find_target_linear_names(model, patterns: Iterable[str]) -> list[str]:
    """Return qualified names of `nn.Linear` modules whose name matches a pattern."""
    import torch.nn as nn  # local import: keep module importable without torch

    out: list[str] = []
    patterns = tuple(patterns)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match_any(name, patterns):
            out.append(name)
    return out


def _all_linear_names(model) -> list[str]:
    import torch.nn as nn
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def make_lora_linear(base, cfg: LoRAConfig):
    """Wrap a frozen `nn.Linear` in a LoRA-augmented module.

    Forward: `base(x) + B(A(dropout(x))) * scaling`. Base weights and bias
    (if any) are frozen.
    """
    import torch
    import torch.nn as nn

    if not isinstance(base, nn.Linear):
        raise TypeError(f"make_lora_linear expects nn.Linear, got {type(base).__name__}")

    in_features = base.in_features
    out_features = base.out_features

    class LoRALinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False

            self.rank = cfg.rank
            self.scaling = cfg.scaling()
            self.lora_disabled = False
            self.lora_A = nn.Linear(in_features, cfg.rank, bias=False)
            self.lora_B = nn.Linear(cfg.rank, out_features, bias=False)
            self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

            # Place LoRA params on the same device as the base, but keep them
            # in fp32 even when the base is bf16/fp16. Mixed-precision autocast
            # casts inputs automatically, while fp32 master copies preserve
            # gradient precision.
            base_w = self.base.weight
            self.lora_A.to(device=base_w.device, dtype=torch.float32)
            self.lora_B.to(device=base_w.device, dtype=torch.float32)

            if cfg.init == "kaiming":
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            elif cfg.init == "zeros":
                nn.init.zeros_(self.lora_A.weight)
            else:
                raise ValueError(f"unknown LoRA init: {cfg.init!r}")
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            base_out = self.base(x)
            if self.lora_disabled:
                return base_out
            # Run the LoRA path in the LoRA params' dtype (typically fp32) and
            # cast back to the base output dtype so mixed-precision callers
            # don't have to wrap the model in autocast.
            lora_dtype = self.lora_A.weight.dtype
            x_lora = self.dropout(x).to(lora_dtype)
            lora_out = self.lora_B(self.lora_A(x_lora)) * self.scaling
            return base_out + lora_out.to(base_out.dtype)

    return LoRALinear()


def inject_lora(model, cfg: LoRAConfig) -> InjectionReport:
    """Replace each matching `nn.Linear` in `model` with a LoRA-augmented module.

    Returns an InjectionReport listing the wrapped module names, the count of
    Linear modules in the model, and a small sample of available names. Raises
    if `cfg.fail_if_no_match` is set and no patterns matched.
    """
    was_training = model.training
    targets = find_target_linear_names(model, cfg.target_module_patterns)
    if not targets and cfg.fail_if_no_match:
        all_names = _all_linear_names(model)
        sample = all_names[:30]
        raise RuntimeError(
            "LoRA injection found no matching nn.Linear modules.\n"
            f"  patterns: {list(cfg.target_module_patterns)}\n"
            f"  available linear modules ({len(all_names)} total, first 30): {sample}"
        )

    for qual in targets:
        parent_path, _, attr = qual.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        base = getattr(parent, attr)
        wrapped = make_lora_linear(base, cfg)
        setattr(parent, attr, wrapped)

    # Newly-created LoRA wrappers and Dropout modules default to train mode even
    # when the parent model was already in eval mode. Preserve the parent state
    # so inference/smoke paths do not accidentally enable LoRA dropout.
    if not was_training:
        model.eval()

    return InjectionReport(
        wrapped=targets,
        skipped_total_linear=len(_all_linear_names(model)),
        available_sample=targets[:5],
    )


def freeze_non_lora(model) -> None:
    """Set requires_grad=False on every parameter except LoRA A/B matrices."""
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def lora_state_dict(model) -> dict:
    """Return only the LoRA tensors so we never checkpoint backbone weights."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if ("lora_A" in name) or ("lora_B" in name)
    }


def trainable_param_report(model) -> TrainableParamReport:
    """Counts of total / trainable parameters and # of LoRA-wrapped modules."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_modules = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            lora_modules += 1
    return TrainableParamReport(total=total, trainable=trainable, lora_modules=lora_modules)


@contextmanager
def lora_disabled(model):
    """Temporarily run all LoRA-wrapped linear layers as their frozen base layers."""
    modules = [mod for _, mod in model.named_modules() if hasattr(mod, "lora_disabled")]
    previous = [bool(mod.lora_disabled) for mod in modules]
    try:
        for mod in modules:
            mod.lora_disabled = True
        yield
    finally:
        for mod, value in zip(modules, previous, strict=True):
            mod.lora_disabled = value


def save_lora_state_dict(model, path: str | Path) -> None:
    """Persist LoRA-only parameters to `path`."""
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_lora_state_dict(model, path: str | Path, *, strict: bool = True) -> list[str]:
    """Load a LoRA-only state dict into a model that has had LoRA injected.

    Returns the list of LoRA parameter names that were successfully assigned.
    """
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=True)
    target_params = {n: p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n}
    missing = sorted(set(target_params) - set(sd))
    extra = sorted(set(sd) - set(target_params))
    if strict and (missing or extra):
        raise RuntimeError(
            f"LoRA state dict mismatch:\n  missing in file: {missing[:5]}...\n  unexpected in file: {extra[:5]}..."
        )
    assigned: list[str] = []
    with torch.no_grad():
        for name, val in sd.items():
            if name in target_params:
                target_params[name].copy_(val.to(target_params[name]))
                assigned.append(name)
    return assigned
