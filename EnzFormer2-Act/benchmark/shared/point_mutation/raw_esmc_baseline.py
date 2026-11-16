#!/usr/bin/env python3
"""Materialize raw-ESMC masked-marginal predictions in a subprocess."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from enzformer2.models.esmc_backbone import ESMCBackboneWrapper
from enzformer2.act.training_common import resolve_device

from benchmark.shared.point_mutation.evaluate import (
    _read_fasta_sequence,
    prepare_truth_rows,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw-ESMC masked-marginal scoring for one query."
    )
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="esmc_600m")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wt-mismatch-policy", type=str, default="error")
    return parser.parse_args(argv)


def _token_id(model: ESMCBackboneWrapper, aa: str) -> int:
    return int(model.token_id(aa))


def _compute_masked_position_log_probs(
    model: ESMCBackboneWrapper,
    sequence: str,
    *,
    batch_size: int,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("raw_esmc batch_size must be positive.")
    raw_length = len(sequence)
    raw_positions = torch.arange(raw_length, dtype=torch.long)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for chunk_positions in raw_positions.split(batch_size):
            outputs = model.forward_masked_positions(
                sequences=[sequence],
                target_positions=chunk_positions.view(1, -1),
                return_log_probs=True,
            )
            target_log_probs = outputs["target_log_probs"]
            if target_log_probs is None:
                raise RuntimeError("Masked ESMC forward did not return target log-probabilities.")
            flat_log_probs = target_log_probs.squeeze(0)
            for local_index, raw_position in enumerate(chunk_positions.tolist()):
                rows.append(
                    {
                        "seq_position": raw_position + 1,
                        "log_probs": flat_log_probs[local_index].detach().cpu(),
                    }
                )
    return pd.DataFrame(rows).sort_values("seq_position", ignore_index=True)


def _score_masked_marginals(
    truth_rows: list[dict[str, object]],
    log_prob_table: pd.DataFrame,
    *,
    model: ESMCBackboneWrapper,
) -> pd.DataFrame:
    log_probs_by_pos = {
        int(row["seq_position"]): row["log_probs"]
        for row in log_prob_table.to_dict(orient="records")
    }
    rows: list[dict[str, object]] = []
    for truth_row in truth_rows:
        seq_position = int(truth_row["seq_position"])
        wt_aa = str(truth_row["seq_wt_aa"])
        mut_aa = str(truth_row["mutant_aa"])
        log_probs = log_probs_by_pos[seq_position]
        wt_token_id = _token_id(model, wt_aa)
        mut_token_id = _token_id(model, mut_aa)
        wt_logp = float(log_probs[wt_token_id].item())
        mut_logp = float(log_probs[mut_token_id].item())
        rows.append(
            {
                "id": str(truth_row["mutant_basename"]),
                "mut_id": str(truth_row["seq_mut_id"]),
                "ambler_mut_id": str(truth_row["mut_id"]),
                "seq_position": seq_position,
                "ambler_position": int(truth_row["ambler_position"]),
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "wt_logp": wt_logp,
                "mut_logp": mut_logp,
                "score": mut_logp - wt_logp,
            }
        )
    scored = (
        pd.DataFrame(rows)
        .sort_values(
            ["score", "seq_position", "mut_aa"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _, truth_rows = prepare_truth_rows(
        truth_csv=args.truth_csv,
        wt_fasta=args.wt_fasta,
        wt_mismatch_policy=args.wt_mismatch_policy,
        query_name=args.query_name or args.wt_fasta.stem,
    )
    if not truth_rows:
        raise ValueError("No mapped truth rows are available for raw_esmc scoring.")
    sequence = _read_fasta_sequence(args.wt_fasta)
    device = resolve_device(args.device)
    model = ESMCBackboneWrapper.from_pretrained(args.model_name, device=device)
    try:
        log_prob_table = _compute_masked_position_log_probs(
            model, sequence, batch_size=args.batch_size
        )
        score_df = _score_masked_marginals(truth_rows, log_prob_table, model=model)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(args.output, index=False)
    print(f"[raw-esmc] predictions_csv={args.output}", flush=True)


if __name__ == "__main__":
    main()
