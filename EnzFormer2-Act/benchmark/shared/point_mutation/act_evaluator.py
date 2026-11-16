"""Evaluation helpers for EnzFormer2-Act mutation-score tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from enzformer2.act.scoring import load_act_model, scan_single_mutants
from enzformer2.act.training_common import compute_benchmark_mean, resolve_device

from benchmark.shared.point_mutation import evaluate as metric_evaluator


def _read_fasta_sequence(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sequence = "".join(line for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError(f"FASTA contains no sequence: {path}")
    return sequence.upper()


def scan_checkpoint(
    checkpoint_path: str | Path,
    *,
    seq_id: str,
    sequence: str,
    chunk_size: int = 64,
    device: str | None = None,
) -> pd.DataFrame:
    model, _ = load_act_model(checkpoint_path, device=resolve_device(device))
    return scan_single_mutants(model, seq_id, sequence, chunk_size=chunk_size)


def _mutation_id(row: Mapping[str, Any]) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("seq_id", "query"))).strip("_") or "query"
    return f"{prefix}_{int(row['pos'])}{row['wt_aa']}to{row['mut_aa']}"


def scanner_output_to_predictions(
    scan_df: pd.DataFrame, *, score_column: str = "score"
) -> pd.DataFrame:
    if score_column not in scan_df.columns:
        raise KeyError(f"Scanner output has no score column {score_column!r}.")
    required = {"seq_id", "pos", "wt_aa", "mut_aa", score_column}
    missing = sorted(required.difference(scan_df.columns))
    if missing:
        raise KeyError(f"Scanner output is missing columns: {missing}")
    result = scan_df.copy()
    result["id"] = result.apply(_mutation_id, axis=1)
    result["mut_id"] = result.apply(
        lambda row: f"{row['wt_aa']}{int(row['pos'])}{row['mut_aa']}", axis=1
    )
    result["final_score_common"] = result[score_column].astype(float)
    return result.sort_values(
        [score_column, "pos", "mut_aa"], ascending=[False, True, True]
    ).reset_index(drop=True)


def benchmark_scan_dataframe(
    scan_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    model_name: str,
    truth_csv: str | Path,
    wt_fasta: str | Path,
    reference_index_csv: str | Path | None = None,
    score_column: str = "score",
) -> dict[str, Any]:
    del reference_index_csv
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions = scanner_output_to_predictions(scan_df, score_column=score_column)
    predictions_path = output_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    query_name = str(scan_df["seq_id"].iloc[0]) if not scan_df.empty else None
    summary, merged = metric_evaluator.evaluate(
        predictions_csv=predictions_path,
        truth_csv=Path(truth_csv),
        wt_fasta=Path(wt_fasta),
        id_column="id",
        score_column=score_column,
        query_name=query_name,
    )
    summary["model_name"] = str(model_name)
    summary["benchmark_mean"] = compute_benchmark_mean(
        spearman_rho=summary.get("Spearman"), ndcg_mean=summary.get("NDCG@10")
    )
    pd.DataFrame(merged).to_csv(output_path / "merged_predictions.csv", index=False)
    (output_path / "metrics.json").write_text(
        __import__("json").dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wt-fasta", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-id", default="query")
    parser.add_argument("--model-name", default="EnzFormer2-Act")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sequence = _read_fasta_sequence(args.wt_fasta)
    scan = scan_checkpoint(
        args.checkpoint,
        seq_id=args.seq_id,
        sequence=sequence,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    benchmark_scan_dataframe(
        scan,
        output_dir=args.output_dir,
        model_name=args.model_name,
        truth_csv=args.truth_csv,
        wt_fasta=args.wt_fasta,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
