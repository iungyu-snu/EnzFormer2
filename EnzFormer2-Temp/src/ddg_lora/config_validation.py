"""Model-free training schema checks; paths retain caller-relative semantics."""
from __future__ import annotations

import math
from pathlib import Path

import yaml


def validate_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping")
    required = {
        "experiment": ("seed", "output_dir"),
        "data": ("train_path", "validation_path"),
        "model": ("name",),
        "lora": ("r", "alpha"),
        "regressor": ("kind",),
        "training": ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
                     "weight_decay", "max_grad_norm", "log_every", "eval_every", "save_every"),
    }
    for section, fields in required.items():
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
        for field in fields:
            if field not in cfg[section] or cfg[section][field] is None:
                raise ValueError(f"missing {section}.{field}")
    if "test_path" in cfg["data"]:
        raise ValueError("data.test_path must not appear in a training configuration")

    def finite(value, name):
        if isinstance(value, dict):
            for key, item in value.items():
                finite(item, f"{name}.{key}")
        elif isinstance(value, list):
            for item in value:
                finite(item, name)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    finite(cfg, "config")

    def number(section, key, *, minimum=0, strict=False, integer=False):
        value = cfg[section].get(key)
        if value is None:
            return
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or (integer and not isinstance(value, int))
                or value < minimum or (strict and value == minimum)):
            raise ValueError(f"invalid numeric setting {section}.{key}: {value!r}")

    for section, keys in {
        "training": ("epochs", "batch_size", "gradient_accumulation_steps", "log_every"),
        "lora": ("r",),
        "data": ("max_train_examples", "max_validation_examples", "max_sequence_length"),
    }.items():
        for key in keys:
            number(section, key, strict=True, integer=True)
    for key in ("eval_every", "save_every"):
        number("training", key, integer=True)
    number("experiment", "seed", integer=True)
    number("regressor", "context_window_k", integer=True)
    for section, keys in {"training": ("learning_rate", "smooth_l1_beta"),
                          "model": ("cuda_memory_limit_gb",), "lora": ("alpha",)}.items():
        for key in keys:
            number(section, key, strict=True)
    for key in ("weight_decay", "max_grad_norm"):
        number("training", key)
    for section, key in (("lora", "dropout"), ("regressor", "head_dropout")):
        number(section, key)
        if cfg[section].get(key, 0) > 1:
            raise ValueError(f"{section}.{key} must be in [0, 1]")
    for section, key in (("experiment", "output_dir"), ("data", "train_path"),
                         ("data", "validation_path")):
        if not isinstance(cfg[section][key], str) or not cfg[section][key].strip():
            raise ValueError(f"{section}.{key} must be a nonempty path string")
    choices = {
        ("model", "name"): ("esmc_300m", "esmc_600m"),
        ("model", "precision"): ("fp32", "fp16", "bf16"),
        ("regressor", "kind"): ("siamese_delta",),
        ("training", "loss"): ("smooth_l1", "mse"),
        ("training", "target_transform"): ("standardize_train", "none"),
    }
    for (section, key), values in choices.items():
        if key in cfg[section] and cfg[section][key] not in values:
            raise ValueError(f"unsupported {section}.{key}: {cfg[section][key]!r}")
    device = cfg["model"].get("device", "auto")
    if not isinstance(device, str) or not (
        device in {"auto", "cpu", "cuda"} or
        (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("unsupported model.device")
    patterns = cfg["lora"].get("target_patterns") or cfg["lora"].get("target_module_patterns")
    if not isinstance(patterns, list) or not patterns or not all(
        isinstance(item, str) and item for item in patterns
    ):
        raise ValueError("lora target patterns must be a nonempty list of strings")
    dims = cfg["regressor"].get("head_hidden_dims", [512, 128])
    if not isinstance(dims, list) or not all(type(x) is int and x > 0 for x in dims):
        raise ValueError("regressor.head_hidden_dims must contain positive integers")
    for section, key in (("data", "include_neutral"), ("lora", "fail_if_no_match"),
                         ("regressor", "use_context")):
        if key in cfg[section] and not isinstance(cfg[section][key], bool):
            raise ValueError(f"{section}.{key} must be boolean")
    prior = cfg["regressor"].get("masked_logodds")
    if prior is None:
        prior = {}
    if not isinstance(prior, dict):
        raise ValueError("regressor.masked_logodds must be a mapping")
    if not isinstance(prior.get("enabled", False), bool):
        raise ValueError("masked_logodds.enabled must be boolean")
    for key, values in {
        "source": ("base_esmc_frozen", "lora_detached"),
        "mode": ("wt_context", "symmetric", "both"),
        "transform": ("raw", "none", "train_zscore", "train_zscore_clip5"),
        "fusion": ("concat", "residual_prior"),
    }.items():
        if key in prior and prior[key] not in values:
            raise ValueError(f"unsupported masked_logodds.{key}")
    if prior.get("fusion") == "residual_prior" and prior.get("mode") == "both":
        raise ValueError("residual_prior requires a 1-D mode")
    for key in ("feature_dropout", "residual_gamma_init"):
        value = prior.get(key, 0)
        if type(value) not in (int, float):
            raise ValueError(f"masked_logodds.{key} must be numeric")
    if not 0 <= prior.get("feature_dropout", 0) <= 1:
        raise ValueError("masked_logodds.feature_dropout must be in [0, 1]")
    dim = 2 if prior.get("mode") == "both" else 1
    for key, default in (("mean", 0.0), ("std", 1.0)):
        values = prior.get(key, [default] * dim)
        if (not isinstance(values, list) or len(values) != dim
                or any(type(x) not in (int, float) for x in values)
                or (key == "std" and any(x <= 0 for x in values))):
            raise ValueError(f"masked_logodds.{key} must have {dim} valid numeric entries")
    inputs = {key: {"path": cfg["data"][key],
                    "available": Path(cfg["data"][key]).is_file()}
              for key in ("train_path", "validation_path")}
    return {"schema_valid": True, "inputs": inputs,
            "all_inputs_available": all(item["available"] for item in inputs.values())}
