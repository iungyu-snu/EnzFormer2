#!/usr/bin/env python
"""Precompute frozen-ESMC masked log-odds columns for regression parquets.

For a WT sequence with mutation wt->mut at position i, this writes:

  * masked_logodds_forward = log p(mut | mask(WT, i)) - log p(wt | mask(WT, i))
  * masked_logodds_reverse = log p(wt | mask(Mut, i)) - log p(mut | mask(Mut, i))

The output parquets preserve all original columns and add the two scalar
features expected by the masked-logodds Siamese-delta sweep configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ddg_lora.constants import AA_SET
from ddg_lora.data.regression_dataset import _apply_single_mutation
from ddg_lora.models.esmc_wrapper import ESMCMaskedLM
from ddg_lora.utils.logging import get_logger

log = get_logger("precompute_esmc_masked_logodds")


REQUIRED_COLUMNS = ("wt_sequence", "position_0idx", "wt_aa", "mut_aa")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default="data/processed/megascale_thermompnn",
        help="Directory containing train/validation/test parquet files.",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed/megascale_thermompnn_esmc_logodds",
        help="Directory where enriched parquets will be written.",
    )
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--model-name", default="esmc_600m")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-rows", type=int, default=None, help="Debug cap per split.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _validate_row(row: pd.Series, *, row_index: int) -> tuple[str, str, int, str, str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in row.index]
    if missing:
        raise ValueError(f"input parquet missing required columns: {missing}")

    wt_sequence = str(row["wt_sequence"])
    position_0idx = int(row["position_0idx"])
    wt_aa = str(row["wt_aa"]).upper()
    mut_aa = str(row["mut_aa"]).upper()

    if not (0 <= position_0idx < len(wt_sequence)):
        raise ValueError(
            f"row {row_index}: position_0idx={position_0idx} out of range "
            f"for sequence length {len(wt_sequence)}"
        )
    if wt_sequence[position_0idx].upper() != wt_aa:
        raise ValueError(
            f"row {row_index}: wt_sequence[{position_0idx}]="
            f"{wt_sequence[position_0idx]!r}, expected {wt_aa!r}"
        )
    if wt_aa not in AA_SET or mut_aa not in AA_SET:
        raise ValueError(f"row {row_index}: non-canonical mutation {wt_aa}->{mut_aa}")
    if wt_aa == mut_aa:
        raise ValueError(f"row {row_index}: wt_aa equals mut_aa ({wt_aa})")

    mut_sequence = _apply_single_mutation(wt_sequence, position_0idx, mut_aa)
    return wt_sequence, mut_sequence, position_0idx, wt_aa, mut_aa


def _precompute_split(
    *,
    wrapper: ESMCMaskedLM,
    input_path: Path,
    output_path: Path,
    max_rows: int | None,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        log.info("skipping existing output: %s", output_path)
        return
    if not input_path.exists():
        raise FileNotFoundError(f"missing input parquet: {input_path}")

    df = pd.read_parquet(input_path)
    if max_rows is not None:
        df = df.head(int(max_rows)).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{input_path} missing required columns: {missing}")

    cache: dict[tuple[str, int, str, str], tuple[float, float]] = {}
    forward_values: list[float] = []
    reverse_values: list[float] = []
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc=input_path.stem,
        unit="mut",
    )
    for row_index, row in iterator:
        wt_sequence, mut_sequence, position_0idx, wt_aa, mut_aa = _validate_row(
            row,
            row_index=int(row_index),
        )
        key = (wt_sequence, position_0idx, wt_aa, mut_aa)
        if key not in cache:
            cache[key] = wrapper.masked_logodds_pair(
                wt_sequence,
                mut_sequence,
                position_0idx,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
            )
        forward, reverse = cache[key]
        forward_values.append(float(forward))
        reverse_values.append(float(reverse))

    out_df = df.copy()
    out_df["masked_logodds_forward"] = forward_values
    out_df["masked_logodds_reverse"] = reverse_values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    out_df.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    log.info(
        "wrote %s rows=%d unique_masked_contexts=%d",
        output_path,
        len(out_df),
        len(cache),
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    wrapper = ESMCMaskedLM.load(model_name=args.model_name, device=args.device)
    wrapper.model.eval()
    for split in args.splits:
        _precompute_split(
            wrapper=wrapper,
            input_path=input_dir / f"{split}.parquet",
            output_path=out_dir / f"{split}.parquet",
            max_rows=args.max_rows,
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
