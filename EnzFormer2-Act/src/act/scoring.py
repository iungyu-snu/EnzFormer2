"""Rank all canonical single substitutions with an Act checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import Tensor

from enzformer2.config import act_score_config_to_namespace, load_act_score_config
from enzformer2.data.manifest_utils import DEFAULT_AA_ALPHABET, DEFAULT_SEQ_ID_COLUMNS, _read_table
from enzformer2.data.raw_sequence_dataset import RawSequenceDataset
from enzformer2.models.conditional_masked_scorer import ESMCMaskedConditionalScorer
from enzformer2.models.esmc_backbone import ESMCBackboneWrapper
from enzformer2.act.training_common import (
    load_model_state_into_scorer,
    load_trusted_checkpoint,
    resolve_device,
)


CANONICAL_AA_ALPHABET = tuple(DEFAULT_AA_ALPHABET[:20])
CANONICAL_AA_SET = frozenset(CANONICAL_AA_ALPHABET)
REQUIRED_SCAN_COLUMNS = (
    "seq_id",
    "pos",
    "wt_aa",
    "mut_aa",
    "score",
    "rank",
)


@dataclass(slots=True)
class ScanSpec:
    """Resolved Act scoring model."""

    model: ESMCMaskedConditionalScorer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enzformer2 act score",
        description="Score all 19 non-wild-type substitutions at every canonical residue.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--sequence-root", type=Path, default=None)
    parser.add_argument("--seq-id", default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def resolve_score_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    if cli_args.config is not None:
        args = act_score_config_to_namespace(load_act_score_config(cli_args.config))
    else:
        args = argparse.Namespace(
            checkpoint=None,
            manifest_path=None,
            sequence_root=None,
            sequence_column=None,
            seq_id_column=None,
            sequence_lookup_path=None,
            seq_id=None,
            input_manifest=None,
            output=None,
            chunk_size=64,
            device=None,
        )
    for name in (
        "checkpoint",
        "manifest",
        "sequence_root",
        "seq_id",
        "input_manifest",
        "output",
        "chunk_size",
        "device",
    ):
        value = getattr(cli_args, name, None)
        if value is not None:
            setattr(args, "manifest_path" if name == "manifest" else name, value)
    for name in ("checkpoint", "manifest_path", "sequence_root", "input_manifest", "output"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    if args.checkpoint is None:
        raise ValueError(
            "Act scoring requires --checkpoint or a scoring config with checkpoint.checkpoint."
        )
    args.fasta = cli_args.fasta
    if args.fasta is not None:
        if args.manifest_path is not None or args.input_manifest is not None:
            raise ValueError("Use --fasta or a sequence manifest, not both.")
        args.fasta = args.fasta.expanduser().resolve()
    elif args.manifest_path is None:
        raise ValueError("Act scoring requires --fasta, --manifest or data.manifest_path.")
    return args


def _canonical_token_ids(backbone: ESMCBackboneWrapper) -> Tensor:
    return torch.tensor(
        [backbone.token_id(residue) for residue in CANONICAL_AA_ALPHABET], dtype=torch.long
    )


def load_act_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[ESMCMaskedConditionalScorer, dict[str, Any]]:
    """Instantiate the adapter described by a local checkpoint."""

    checkpoint = load_trusted_checkpoint(checkpoint_path, device=device)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Act checkpoint lacks a model_config mapping.")
    backbone_name = str(model_config.get("backbone_model_name", "esmc_600m"))
    backbone = ESMCBackboneWrapper.from_pretrained(backbone_name, device=device)
    model = ESMCMaskedConditionalScorer(
        backbone_model_name=backbone_name,
        freeze_backbone=bool(model_config.get("freeze_backbone", True)),
        lora_rank=int(model_config.get("lora_rank", 32)),
        lora_alpha=float(model_config.get("lora_alpha", 32)),
        lora_dropout=float(model_config.get("lora_dropout", 0.05)),
        lora_target_modules=tuple(
            model_config.get("lora_target_modules", ("layernorm_qkv", "out_proj"))
        ),
        backbone=backbone,
    ).to(device)
    load_model_state_into_scorer(
        model,
        checkpoint.get("model_state", {}),
        state_dict_mode=str(
            checkpoint.get("extra_state", {}).get("state_dict_mode", "trainable_only")
        ),
    )
    model.eval()
    return model, checkpoint


def build_scan_spec_from_checkpoints(
    *,
    checkpoint_single: str | Path,
    device: torch.device,
) -> tuple[ScanSpec, dict[str, Any]]:
    model, checkpoint = load_act_model(checkpoint_single, device=device)
    return ScanSpec(model), checkpoint


def _score_chunk(
    model: ESMCMaskedConditionalScorer, sequence: str, positions: Tensor
) -> pd.DataFrame:
    with torch.inference_mode():
        output = model.backbone.forward_masked_positions(
            sequences=[sequence],
            target_positions=positions.view(1, -1),
            return_log_probs=True,
            share_context_across_targets=False,
        )
    log_probs = output["target_log_probs"]
    wt_ids = output["wt_token_ids"]
    if log_probs is None or wt_ids is None:
        raise RuntimeError("ESM-C did not return target log probabilities and WT token ids.")
    log_probs = log_probs.squeeze(0).float()
    wt_ids = wt_ids.squeeze(0).to(dtype=torch.long)
    token_ids = _canonical_token_ids(model.backbone).to(device=log_probs.device)
    candidate_scores = log_probs.index_select(-1, token_ids)
    wt_scores = log_probs.gather(-1, wt_ids.view(-1, 1))
    deltas = candidate_scores - wt_scores
    rows: list[dict[str, Any]] = []
    for row_index, position in enumerate(positions.tolist()):
        wt = (
            CANONICAL_AA_ALPHABET[int((token_ids == wt_ids[row_index]).nonzero()[0].item())]
            if bool((token_ids == wt_ids[row_index]).any())
            else "X"
        )
        for aa_index, mutant in enumerate(CANONICAL_AA_ALPHABET):
            if mutant == wt:
                continue
            score = float(deltas[row_index, aa_index].item())
            rows.append(
                {
                    "seq_id": "",
                    "pos": int(position) + 1,
                    "wt_aa": wt,
                    "mut_aa": mutant,
                    "score": score,
                    "masked_log_odds": score,
                    "delta_base": score,
                    "delta_lora": score,
                    "delta_common": score,
                    "final_score_common": score,
                }
            )
    return pd.DataFrame(rows)


def scan_single_mutants(
    model_or_spec: ESMCMaskedConditionalScorer | ScanSpec,
    seq_id: str,
    sequence: str,
    *,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """Return descending masked-log-odds scores for all valid substitutions."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    sequence = str(sequence).strip().upper()
    positions = torch.tensor(
        [index for index, residue in enumerate(sequence) if residue in CANONICAL_AA_SET],
        dtype=torch.long,
    )
    if positions.numel() == 0:
        raise ValueError("The query sequence contains no canonical amino-acid residues.")
    model = model_or_spec.model if isinstance(model_or_spec, ScanSpec) else model_or_spec
    frames: list[pd.DataFrame] = []
    for chunk in positions.split(int(chunk_size)):
        frame = _score_chunk(model, sequence, chunk)
        frame["seq_id"] = str(seq_id)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(
        ["score", "pos", "mut_aa"], ascending=[False, True, True]
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result[
        [
            "seq_id",
            "pos",
            "wt_aa",
            "mut_aa",
            "score",
            "masked_log_odds",
            "delta_base",
            "delta_lora",
            "delta_common",
            "final_score_common",
            "rank",
        ]
    ]


def export_scan_results(results: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        path = path.with_suffix(".csv")
    if path.suffix.lower() == ".tsv":
        results.to_csv(path, sep="\t", index=False)
    else:
        results.to_csv(path, index=False)
    return path


def _seq_ids_from_manifest(path: Path) -> list[str]:
    frame = _read_table(path)
    column = next((name for name in DEFAULT_SEQ_ID_COLUMNS if name in frame.columns), None)
    if column is None:
        raise ValueError(f"Input manifest needs one of {DEFAULT_SEQ_ID_COLUMNS}.")
    return [str(value) for value in frame[column].tolist()]


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_score_args(parse_args(argv))
    if args.fasta is not None:
        from enzformer2.profile.fasta_utils import parse_fasta_records, sanitize_seq

        records = parse_fasta_records(args.fasta)
        if len(records) != 1:
            raise ValueError("--fasta must contain exactly one sequence.")
        header, sequence = records[0]
        sequence = sanitize_seq(sequence)
        device = resolve_device(args.device)
        spec, _ = build_scan_spec_from_checkpoints(checkpoint_single=args.checkpoint, device=device)
        scores = scan_single_mutants(
            spec, args.seq_id or header or args.fasta.stem, sequence, chunk_size=args.chunk_size
        )
        export_scan_results(scores, args.output or Path("act_scores.csv"))
        return
    device = resolve_device(args.device)
    spec, checkpoint = build_scan_spec_from_checkpoints(
        checkpoint_single=args.checkpoint, device=device
    )
    data_config = checkpoint.get("data_config", {})
    sequence_root = args.sequence_root or data_config.get("sequence_root")
    dataset = RawSequenceDataset(
        args.manifest_path,
        sequence_root=sequence_root,
        sequence_column=getattr(args, "sequence_column", None),
        seq_id_column=getattr(args, "seq_id_column", None),
        sequence_lookup_path=getattr(args, "sequence_lookup_path", None),
    )
    if args.seq_id is not None:
        wanted = [str(args.seq_id)]
    elif args.input_manifest is not None:
        wanted = _seq_ids_from_manifest(args.input_manifest)
    else:
        wanted = [str(record["seq_id"]) for record in dataset.records]
    by_id = {str(record["seq_id"]): record for record in dataset.records}
    frames: list[pd.DataFrame] = []
    for seq_id in wanted:
        if seq_id not in by_id:
            raise KeyError(f"Sequence identifier not found in the manifest: {seq_id}")
        frames.append(
            scan_single_mutants(spec, seq_id, by_id[seq_id]["sequence"], chunk_size=args.chunk_size)
        )
    if not frames:
        raise ValueError("No sequences were selected for scoring.")
    output = args.output or Path("act_scores.csv")
    export_scan_results(pd.concat(frames, ignore_index=True), output)


if __name__ == "__main__":  # pragma: no cover
    main()
