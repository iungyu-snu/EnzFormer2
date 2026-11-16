#!/usr/bin/env python
"""Predict EnzFormer2-Temp scores for every single mutant from one sequence.

The final model uses attention-only LoRA, context k=2, normalized residual prior,
and SmoothL1 beta=2. By default, decoded ddG predictions from seeds 42/43/44
are averaged. Frozen-base prior features are computed once per WT sequence.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ddg_lora.constants import AMINO_ACIDS_20, AA_SET
from ddg_lora.data.regression_dataset import RegressionExample, _apply_single_mutation
from ddg_lora.evaluation.regression_eval import predict_examples
from ddg_lora.ensemble import (
    DEFAULT_CHECKPOINT, bundle_members, predict_members, provenance, validate_member,
)
from ddg_lora.models.lora import lora_disabled

_AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS_20)}
_FIXED_OUTPUT_FILENAMES = (
    "predictions.csv",
    "predictions.parquet",
    "top_mutations.csv",
    "candidates_with_logodds.parquet",
    "prediction_report.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="Raw canonical amino-acid sequence.")
    source.add_argument("--fasta", help="FASTA containing exactly one nonempty record.")
    p.add_argument("--sequence-id", help="Optional output sequence identifier.")
    p.add_argument("--output-dir", required=True, help="Directory for the five prediction outputs.")
    p.add_argument("--checkpoint-dir", help="Checkpoint bundle directory.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace only this command's five fixed prediction files.",
    )
    return p.parse_args(argv)


def normalize_sequence(raw: str, *, source: str) -> str:
    sequence = "".join(raw.split()).upper()
    if not sequence:
        raise ValueError(f"{source} is empty; provide one or more canonical amino-acid residues")
    noncanonical = sorted(set(sequence) - set(AA_SET))
    if noncanonical:
        raise ValueError(f"{source} contains non-canonical residues: {', '.join(noncanonical)}")
    return sequence


def read_single_fasta(path: Path) -> tuple[str, str]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = text[1:].strip().split()[0] or path.stem
            chunks = []
        else:
            chunks.append(text)
    if header is not None:
        records.append((header, "".join(chunks)))
    if len(records) != 1:
        raise ValueError(f"{path} must contain exactly one FASTA record; found {len(records)}")
    sequence_id, sequence = records[0]
    return sequence_id, normalize_sequence(sequence, source=f"FASTA record in {path}")


def resolve_input(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.sequence is not None:
        sequence = normalize_sequence(args.sequence, source="--sequence")
        return args.sequence_id or "query", sequence, "--sequence"
    fasta_path = Path(args.fasta)
    sequence_id, sequence = read_single_fasta(fasta_path)
    return args.sequence_id or sequence_id, sequence, str(fasta_path)


def resolve_checkpoint(args: argparse.Namespace, *, default_root: Path | None = None) -> tuple[Path, str, str]:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir), "final", "--checkpoint-dir"
    environment = os.environ.get("ENZFORMER2_TEMP_CHECKPOINT_DIR")
    if environment:
        return Path(environment), "final", "ENZFORMER2_TEMP_CHECKPOINT_DIR"
    return (default_root or Path.cwd()) / DEFAULT_CHECKPOINT, "final", "default final bundle"


def validate_checkpoint_bundle(checkpoint_dir: Path, version: str | None = None) -> None:
    bundle_members(checkpoint_dir)


def validate_runtime_arguments(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")


def preflight_output_paths(output_dir: Path, *, allow_overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir must be a directory, not {output_dir}")
    paths = {name: output_dir / name for name in _FIXED_OUTPUT_FILENAMES}
    unsafe = [
        str(path)
        for path in paths.values()
        if path.is_symlink() or (path.exists() and not path.is_file())
    ]
    if unsafe:
        raise FileExistsError(
            "refusing unsafe output target(s); fixed outputs must be regular files:\n  "
            + "\n  ".join(unsafe)
        )
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions and not allow_overwrite:
        raise FileExistsError(
            "refusing to overwrite existing EnzFormer2_temp outputs; pass --allow-overwrite:\n  "
            + "\n  ".join(collisions)
        )
    return paths


def materialize_all_single_mutants(
    *,
    protein_id: str,
    sequence: str,
    source_fasta: Path,
) -> pd.DataFrame:
    noncanonical = sorted(set(sequence) - set(AA_SET))
    if noncanonical:
        raise ValueError(f"{source_fasta} contains non-canonical residues: {noncanonical}")
    rows: list[dict] = []
    for position_0idx, wt_aa in enumerate(sequence):
        for mut_aa in AMINO_ACIDS_20:
            if mut_aa == wt_aa:
                continue
            rows.append(
                {
                    "protein_id": protein_id,
                    "wt_sequence": sequence,
                    "position_0idx": position_0idx,
                    "position_1idx": position_0idx + 1,
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "mutation": f"{wt_aa}{position_0idx + 1}{mut_aa}",
                    "ddg": float("nan"),
                    "label3": "unknown",
                    "label_binary": None,
                    "source_fasta": str(source_fasta),
                }
            )
    return pd.DataFrame(rows)


def _inner_siamese_model(model: object) -> object:
    return getattr(model, "inner", model)


def add_wt_context_masked_logodds(df: pd.DataFrame, model: object) -> pd.DataFrame:
    inner = _inner_siamese_model(model)
    masked_cfg = getattr(inner, "masked_logodds", None)
    if masked_cfg is None or not bool(getattr(masked_cfg, "enabled", False)):
        out = df.copy()
        out["masked_logodds_forward"] = None
        out["masked_logodds_reverse"] = None
        return out
    mode = str(getattr(masked_cfg, "mode", ""))
    if mode != "wt_context":
        raise ValueError(
            f"this FASTA scorer only supports masked_logodds.mode='wt_context'; got {mode!r}"
        )
    backbone = getattr(inner, "backbone", None)
    if backbone is None or not hasattr(backbone, "masked_log_probs_at_position"):
        raise RuntimeError("loaded model does not expose an ESMC masked-logodds backbone")
    backbone_model = getattr(backbone, "model", None)
    disable_ctx = (
        lora_disabled(backbone_model)
        if str(getattr(masked_cfg, "source", "")) == "base_esmc_frozen" and backbone_model is not None
        else nullcontext()
    )

    out = df.copy()
    forward_by_index: dict[int, float] = {}
    grouped = out.groupby(["wt_sequence", "position_0idx", "wt_aa"], sort=False)
    with torch.inference_mode(), disable_ctx:
        if backbone_model is not None:
            backbone_model.eval()
        for (wt_sequence, position_0idx, wt_aa), group in tqdm(
            grouped,
            total=grouped.ngroups,
            desc="wt-context log-odds",
            unit="pos",
        ):
            log_probs = backbone.masked_log_probs_at_position(str(wt_sequence), int(position_0idx))
            wt_id = backbone.amino_acid_to_token_id(str(wt_aa))
            for row_index, mut_aa in zip(group.index, group["mut_aa"].astype(str), strict=True):
                mut_id = backbone.amino_acid_to_token_id(mut_aa)
                forward_by_index[int(row_index)] = float((log_probs[mut_id] - log_probs[wt_id]).item())
    out["masked_logodds_forward"] = [forward_by_index[int(idx)] for idx in out.index]
    # The final Temp model uses wt_context mode, so reverse is intentionally a finite
    # placeholder to avoid runtime recomputation in the generic regressor path.
    out["masked_logodds_reverse"] = 0.0
    return out


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def rows_to_examples(df: pd.DataFrame) -> list[RegressionExample]:
    examples: list[RegressionExample] = []
    for _, row in df.iterrows():
        wt_sequence = str(row["wt_sequence"])
        position_0idx = int(row["position_0idx"])
        wt_aa = str(row["wt_aa"])
        mut_aa = str(row["mut_aa"])
        mut_sequence = _apply_single_mutation(wt_sequence, position_0idx, mut_aa)
        examples.append(
            RegressionExample(
                protein_id=str(row["protein_id"]),
                wt_sequence=wt_sequence,
                mut_sequence=mut_sequence,
                position_0idx=position_0idx,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                wt_token_index=_AA_INDEX[wt_aa],
                mut_token_index=_AA_INDEX[mut_aa],
                ddg=float("nan"),
                label3="unknown",
                label_binary=None,
                mutation=str(row["mutation"]),
                masked_logodds_forward=_optional_float(row["masked_logodds_forward"]),
                masked_logodds_reverse=_optional_float(row["masked_logodds_reverse"]),
            )
        )
    return examples


def predict_examples_cached_wt(
    *,
    loaded: object,
    wt_sequence: str,
    examples: list[RegressionExample],
    batch_size: int,
) -> list[float]:
    """Predict same-WT single mutants while reusing the WT hidden states once."""
    inner = _inner_siamese_model(loaded.model)
    if getattr(loaded, "regressor_kind", None) != "siamese_delta" or inner is not loaded.model:
        return predict_examples(
            loaded.model,
            loaded.backbone_module,
            examples,
            target_transform=loaded.target_transform,
            precision=loaded.precision,
            batch_size=batch_size,
        )
    backbone = getattr(inner, "backbone", None)
    if backbone is None or not hasattr(backbone, "forward_residue_hidden_states_batch"):
        return predict_examples(
            loaded.model,
            loaded.backbone_module,
            examples,
            target_transform=loaded.target_transform,
            precision=loaded.precision,
            batch_size=batch_size,
        )

    device_type = next(loaded.backbone_module.parameters()).device.type
    autocast_dtype = torch.bfloat16 if loaded.precision == "bf16" else torch.float16
    use_autocast = device_type == "cuda" and loaded.precision in {"bf16", "fp16"}
    decode = getattr(loaded.target_transform, "decode")
    out: list[float] = []

    loaded.model.eval()
    loaded.backbone_module.eval()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype,
            enabled=use_autocast,
        ):
            h_wt_single = backbone.forward_residue_hidden_states(wt_sequence)
        for start in tqdm(
            range(0, len(examples), max(1, batch_size)),
            desc="EnzFormer2_temp prediction",
            unit="batch",
        ):
            batch = examples[start : start + max(1, batch_size)]
            mut_sequences = [ex.mut_sequence for ex in batch]
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                h_mut, lengths = backbone.forward_residue_hidden_states_batch(mut_sequences)
                batch_n, max_len, _ = h_mut.shape
                h_wt = h_wt_single[:max_len].unsqueeze(0).expand(batch_n, max_len, -1)
                positions = torch.tensor(
                    [ex.position_0idx for ex in batch], device=h_mut.device, dtype=torch.long
                )
                wt_idx = torch.tensor(
                    [ex.wt_token_index for ex in batch], device=h_mut.device, dtype=torch.long
                )
                mut_idx = torch.tensor(
                    [ex.mut_token_index for ex in batch], device=h_mut.device, dtype=torch.long
                )
                raw_logodds = None
                if bool(getattr(inner.masked_logodds, "enabled", False)):
                    raw_logodds = torch.stack(
                        [inner._raw_masked_logodds_from_example(ex) for ex in batch],
                        dim=0,
                    )
                feat, logodds_feat = inner._build_feature_batch(
                    h_wt,
                    h_mut,
                    lengths,
                    positions,
                    wt_idx,
                    mut_idx,
                    raw_logodds,
                )
                pred_std = inner._apply_masked_logodds_residual(inner.head(feat), logodds_feat)
            values = pred_std.detach().float().cpu().numpy().reshape(-1)
            out.extend(float(decode(float(v))) for v in values)
    return out


def predict_one_record(
    *,
    protein_id: str,
    sequence: str,
    source_fasta: Path,
    loaded: object,
    output_paths: dict[str, Path],
    batch_size: int,
    top_k: int,
    checkpoint_dir: Path,
    model_version: str,
    checkpoint_provenance: str,
    candidates: pd.DataFrame | None = None,
    member_predictions: dict | None = None,
) -> dict:
    if candidates is None:
        candidates = materialize_all_single_mutants(
            protein_id=protein_id, sequence=sequence, source_fasta=source_fasta)
        candidates = add_wt_context_masked_logodds(candidates, loaded.model)
    if member_predictions is None:
        preds = predict_examples_cached_wt(
            loaded=loaded, wt_sequence=sequence, examples=rows_to_examples(candidates),
            batch_size=batch_size)
    else:
        values = np.asarray(list(member_predictions.values()), dtype=float)
        if values.shape != (len(member_predictions), len(candidates)) or not np.isfinite(values).all():
            raise ValueError("Invalid ensemble member predictions")
        preds = values.mean(axis=0)
    out = candidates[
        [
            "protein_id",
            "mutation",
            "position_0idx",
            "position_1idx",
            "wt_aa",
            "mut_aa",
            "source_fasta",
            "masked_logodds_forward",
            "masked_logodds_reverse",
        ]
    ].copy()
    out = out.rename(columns={"protein_id": "sequence_id"})
    out["position"] = out["position_1idx"]
    out["wt"] = out["wt_aa"]
    out["mutant"] = out["mut_aa"]
    out["predicted_ddg"] = [float(v) for v in preds]
    out["score_stabilizing"] = -out["predicted_ddg"]
    for seed, values in (member_predictions or {}).items():
        out[f"predicted_ddg_seed{seed}"] = values
    out = out.sort_values(["score_stabilizing", "mutation"], ascending=[False, True], kind="stable").reset_index(drop=True)
    out["rank"] = out.index + 1
    candidates.to_parquet(output_paths["candidates_with_logodds.parquet"], index=False)
    out.to_csv(output_paths["predictions.csv"], index=False)
    out.to_parquet(output_paths["predictions.parquet"], index=False)
    out.head(min(max(0, int(top_k)), len(out))).to_csv(output_paths["top_mutations.csv"], index=False)
    return {
        "sequence_id": protein_id,
        "input_source": str(source_fasta),
        "sequence_length": len(sequence),
        "n_single_mutants": int(len(out)),
        "model_name": "EnzFormer2-Temp",
        "model_version": model_version,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_provenance": checkpoint_provenance,
    }


def main(argv: list[str] | None = None, *, default_root: Path | None = None) -> int:
    args = parse_args(argv)
    try:
        protein_id, sequence, source = resolve_input(args)
        validate_runtime_arguments(args)
        checkpoint_dir, model_version, checkpoint_provenance = resolve_checkpoint(args, default_root=default_root)
        members = bundle_members(checkpoint_dir)
        output_dir = Path(args.output_dir)
        output_paths = preflight_output_paths(output_dir, allow_overwrite=bool(args.allow_overwrite))
    except (OSError, ValueError, FileExistsError) as exc:
        args_error = argparse.ArgumentTypeError(str(exc))
        raise SystemExit(f"error: {args_error}") from exc

    # Validate every member and every fixed output before loading any weights.
    batch_size = args.batch_size
    candidates = materialize_all_single_mutants(
        protein_id=protein_id, sequence=sequence, source_fasta=Path(source))
    examples = None

    def predict(loaded):
        nonlocal candidates, examples
        if examples is None:
            candidates = add_wt_context_masked_logodds(candidates, loaded.model)
            examples = rows_to_examples(candidates)
        return predict_examples_cached_wt(
            loaded=loaded, wt_sequence=sequence, examples=examples, batch_size=batch_size)

    values = predict_members(members, predict)
    seeds = [validate_member(member)["experiment"]["seed"] for member in members]
    output_dir.mkdir(parents=True, exist_ok=True)
    record = predict_one_record(
        protein_id=protein_id, sequence=sequence, source_fasta=Path(source), loaded=None,
        output_paths=output_paths, batch_size=batch_size, top_k=int(args.top_k),
        checkpoint_dir=checkpoint_dir, model_version=model_version,
        checkpoint_provenance=checkpoint_provenance, candidates=candidates,
        member_predictions=dict(zip(seeds, values, strict=True)),
    )
    report = {
        **record, **provenance(checkpoint_dir, members),
        "regressor_kind": "siamese_delta",
        "precision": validate_member(members[0])["model"]["precision"],
        "batch_size": batch_size,
        "score_ddg_convention": "predicted_ddg is mean decoded kcal/mol; score_stabilizing = -predicted_ddg; higher ranks first",
        "output_paths": {name: str(path) for name, path in output_paths.items()},
    }
    output_paths["prediction_report.json"].write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
