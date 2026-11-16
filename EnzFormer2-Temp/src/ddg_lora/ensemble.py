"""Final Temp bundles and sequential, decoded-ddG ensemble prediction."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

MODEL_NAME = "EnzFormer2-Temp"
DEFAULT_CHECKPOINT = Path("checkpoints/enzformer2_temp")
FINAL_SEEDS = (42, 43, 44)
BUNDLE_FILES = ("train_config.yaml", "target_transform.json", "best_adapter.pt",
                "best_head.pt", "best_regressor.pt", "masked_logodds.json")
TARGET_COLUMNS = ["protein_id", "mutation", "position_0idx", "wt_aa", "mut_aa",
                  "ddg", "label3", "label_binary"]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_member(path: Path, *, require_final: bool = True) -> dict:
    if not (path / "train_config.yaml").is_file():
        raise FileNotFoundError(f"EnzFormer2-Temp checkpoint bundle is incomplete: {path}/train_config.yaml")
    try:
        cfg = yaml.safe_load((path / "train_config.yaml").read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}/train_config.yaml: malformed YAML") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}/train_config.yaml must contain a mapping")
    masked = cfg.get("regressor", {}).get("masked_logodds") or {}
    enabled = masked.get("enabled", False)
    required = BUNDLE_FILES if require_final or enabled else BUNDLE_FILES[:4]
    missing = [str(path / name) for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError("EnzFormer2-Temp checkpoint bundle is incomplete; missing:\n  "
                                + "\n  ".join(missing))
    if not require_final:
        return cfg
    expected = {
        "model": {"name": "esmc_600m"},
        "lora": {"r": 16, "alpha": 64, "dropout": .05,
                 "target_patterns": ["attn.layernorm_qkv.1", "attn.out_proj"]},
        "regressor": {"kind": "siamese_delta", "use_context": True, "context_window_k": 2,
                      "head_hidden_dims": [256, 64], "head_dropout": .1},
        "training": {"loss": "smooth_l1", "smooth_l1_beta": 2.,
                     "target_transform": "standardize_train"},
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if cfg.get(section, {}).get(key) != value:
                raise ValueError(f"Not the final Temp configuration: {path}: {section}.{key}")
    prior = cfg["regressor"].get("masked_logodds", {})
    for key, value in {"enabled": True, "source": "base_esmc_frozen", "mode": "wt_context",
                       "transform": "train_zscore_clip5", "fusion": "residual_prior",
                       "feature_dropout": .05, "residual_gamma_init": .2}.items():
        if prior.get(key) != value:
            raise ValueError(f"Not the final Temp prior: {path}: {key}")
    if cfg.get("ablation", {}).get("freeze_prior_gamma", False):
        raise ValueError("Final Temp requires a learned prior gamma")
    return cfg


def bundle_members(checkpoint_dir: str | Path, *, require_final: bool = True) -> list[Path]:
    root = Path(checkpoint_dir)
    manifest = root / "ensemble.json"
    if not manifest.exists():
        validate_member(root, require_final=require_final)
        return [root]
    record = json.loads(manifest.read_text())
    if (record.get("version") != 1 or record.get("model_name") != MODEL_NAME
            or record.get("aggregation") != "mean_decoded_ddg"):
        raise ValueError("Unsupported Temp ensemble manifest")
    entries = record.get("members", [])
    if tuple(m.get("seed") for m in entries) != FINAL_SEEDS:
        raise ValueError(f"Final ensemble requires exactly seeds {FINAL_SEEDS} in order")
    members = []
    reference = None
    for item in entries:
        relative = Path(item["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()) or path in members:
            raise ValueError("Invalid or duplicate ensemble member path")
        cfg = validate_member(path)
        if cfg["experiment"]["seed"] != item["seed"]:
            raise ValueError("Ensemble seed disagrees with checkpoint")
        settings = {k: cfg[k] for k in ("data", "model", "lora", "regressor", "training")}
        if reference is not None and settings != reference:
            raise ValueError("Ensemble members have different scientific settings")
        reference = settings
        if set(item.get("sha256", {})) != set(BUNDLE_FILES):
            raise ValueError("Ensemble member checksum manifest is incomplete")
        for name in BUNDLE_FILES:
            if file_hash(path / name) != item["sha256"][name]:
                raise ValueError(f"Checkpoint checksum mismatch: {path / name}")
        members.append(path)
    return members


def provenance(checkpoint_dir: str | Path, members: list[Path]) -> dict:
    return {"model_name": MODEL_NAME, "checkpoint_dir": str(checkpoint_dir),
            "inference_mode": "ensemble_mean" if len(members) > 1 else "single_seed",
            "aggregation": "mean_decoded_ddg", "members": [
                {"path": str(p), "seed": validate_member(p, require_final=False)["experiment"]["seed"],
                 "sha256": {n: file_hash(p / n) for n in BUNDLE_FILES if (p / n).is_file()}}
                for p in members]}


def predict_members(members: list[Path], predict: Callable) -> list:
    """Run one full member at a time, releasing its GPU references before the next."""
    import torch
    from ddg_lora.evaluation.regression_eval import load_regression_checkpoint

    results = []
    for member in members:
        loaded = load_regression_checkpoint(member)
        try:
            results.append(predict(loaded))
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def mean_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Align mutation identities and labels before averaging original-unit ddG."""
    if not frames:
        raise ValueError("No member predictions")
    reference = frames[0]
    keys = ["protein_id", "mutation"]
    index = pd.MultiIndex.from_frame(reference[keys])
    values = []
    for frame in frames:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Missing/duplicate prediction identities")
        if len(frame) != len(reference) or set(map(tuple, frame[keys].to_numpy())) != set(index):
            raise ValueError("Ensemble members cover different mutations")
        aligned = frame.set_index(keys).loc[index].reset_index()
        pd.testing.assert_frame_equal(reference[TARGET_COLUMNS].reset_index(drop=True),
                                      aligned[TARGET_COLUMNS], check_exact=True, check_dtype=False)
        if not np.isfinite(aligned[["ddg_hat", "score"]].to_numpy()).all():
            raise ValueError("Nonfinite member prediction")
        np.testing.assert_array_equal(aligned.score.to_numpy(), -aligned.ddg_hat.to_numpy())
        values.append(aligned.ddg_hat.to_numpy())
    out = reference.copy().reset_index(drop=True)
    out["ddg_hat"] = np.mean(np.stack(values), axis=0)
    out["score"] = -out.ddg_hat
    return out


def predict_datasets(members: list[Path], datasets: dict, *, batch_size: int = 16) -> dict:
    from ddg_lora.evaluation.regression_eval import predict_dataset

    results = predict_members(members, lambda loaded: {
        name: predict_dataset(loaded, ds, batch_size=batch_size) for name, ds in datasets.items()})
    return {name: mean_predictions([result[name] for result in results]) for name in datasets}


def predict_example_list(members: list[Path], examples: list, *, batch_size: int = 16) -> list[float]:
    from ddg_lora.evaluation.regression_eval import predict_examples

    values = predict_members(members, lambda loaded: predict_examples(
        loaded.model, loaded.backbone_module, examples, target_transform=loaded.target_transform,
        precision=loaded.precision, batch_size=batch_size))
    array = np.asarray(values, dtype=float)
    if array.shape != (len(members), len(examples)) or not np.isfinite(array).all():
        raise ValueError("Invalid member predictions")
    return array.mean(axis=0).tolist()
