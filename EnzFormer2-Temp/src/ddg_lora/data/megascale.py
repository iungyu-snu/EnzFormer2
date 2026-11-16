"""MegaScale loader + normalizer.

Builds normalized per-split parquets at
`output.processed_dir`. The source schema is configured in
`configs/data/megascale_thermompnn.yaml`.

Output schema (per row):
    split            : "train" | "validation" | "test"
    protein_id       : str (from WT_name)
    wt_sequence      : str (from aa_seq)
    position_0idx    : int
    position_1idx    : int
    wt_aa            : str (1-letter)
    mut_aa           : str (1-letter)
    mutation         : str (canonical "<wt><1idx><mut>")
    ddg              : float (kcal/mol, positive = destabilizing)
    label3           : "stabilizing" | "neutral" | "destabilizing"
    label_binary     : 1 (stab) | 0 (destab) | None (neutral)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ddg_lora.constants import AA_SET
from ddg_lora.data.mutation import Mutation, parse_mutation
from ddg_lora.utils.io import load_yaml
from ddg_lora.utils.logging import get_logger

log = get_logger("megascale")

# Robust mutation parser that tolerates separators: "A23V", "A_23_V", "A23_V",
# "A 23 V". The strict canonical parser in ddg_lora.data.mutation.parse_mutation
# is intentionally narrow; this one is lenient for source data idiosyncrasies.
_LENIENT_MUT_RE = re.compile(
    r"^\s*([A-Za-z])[\s_\-]*?(\d+)[\s_\-]*?([A-Za-z])\s*$"
)


def parse_mutation_lenient(s: str) -> Mutation:
    """Parse a one-letter mutation string with optional separators.

    Falls back to `ddg_lora.data.mutation.parse_mutation` for canonical strings.
    Raises ValueError on malformed input.
    """
    if not isinstance(s, str):
        raise TypeError(f"mutation must be a string, got {type(s).__name__}")
    s = s.strip()
    if not s:
        raise ValueError("empty mutation string")
    m = _LENIENT_MUT_RE.match(s)
    if m:
        wt = m.group(1).upper()
        pos = int(m.group(2))
        mut = m.group(3).upper()
        return Mutation(wt=wt, position=pos, mut=mut)
    # Fall back to the canonical parser for "p.Ala123Val" etc.
    return parse_mutation(s)


@dataclass(frozen=True)
class LabelPolicy3:
    stabilizing_threshold: float = -0.5
    destabilizing_threshold: float = 0.5

    def label3(self, ddg: float) -> str:
        if ddg <= self.stabilizing_threshold:
            return "stabilizing"
        if ddg >= self.destabilizing_threshold:
            return "destabilizing"
        return "neutral"

    def label_binary(self, ddg: float) -> int | None:
        l = self.label3(ddg)
        if l == "stabilizing":
            return 1
        if l == "destabilizing":
            return 0
        return None


def _normalize_ddg_sign(ddg: float, convention: str) -> float:
    """Return ddg in the canonical convention (positive = destabilizing)."""
    if convention == "thermompnn_spurs_negative_is_stabilizing":
        return float(ddg)
    if convention == "rosetta_legacy_positive_is_stabilizing":
        return -float(ddg)
    raise ValueError(f"unknown ddg_sign_convention: {convention!r}")


def _coerce_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return str(v)


@dataclass
class _RowResult:
    ok: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def _normalize_one_row(
    row: dict[str, Any],
    cfg_schema: dict[str, Any],
    canonical_split: str,
    policy: LabelPolicy3,
) -> _RowResult:
    """Normalize a single raw row. Returns either an `ok` payload or `error`.

    IMPORTANT: in MegaScale dataset3_single, the `sequence_col` (`aa_seq`) is
    the **mutant** sequence — already carrying the mutation. We reconstruct
    the wild-type sequence by reverting the single-point mutation:

        wt_sequence = mut_sequence[:pos0] + mut.wt + mut_sequence[pos0+1:]

    and assert that `mut_sequence[pos0] == mut.mut` before reverting. Rows
    that fail this assertion are written to errors.csv, never silently dropped.

    Set `cfg_schema['sequence_is_mutant']: false` if a downstream variant of
    this dataset stores wild-type sequences directly; the assertion then
    reverts to checking `seq[pos0] == mut.wt`.
    """
    protein_id = _coerce_str(row.get(cfg_schema["protein_id_col"]))
    seq_in = _coerce_str(row.get(cfg_schema["sequence_col"]))
    mut_str = _coerce_str(row.get(cfg_schema["mutation_col"]))
    ddg_raw = row.get(cfg_schema["ddg_col"])

    if not seq_in:
        return _RowResult(error={**row, "_error": "missing_sequence"})
    if not mut_str:
        return _RowResult(error={**row, "_error": "missing_mutation"})
    if ddg_raw is None or (isinstance(ddg_raw, float) and pd.isna(ddg_raw)):
        return _RowResult(error={**row, "_error": "missing_ddg"})

    try:
        mut = parse_mutation_lenient(mut_str)
    except (TypeError, ValueError) as e:
        return _RowResult(error={**row, "_error": f"unparseable_mutation: {e}"})

    pos0 = mut.index0()
    if pos0 >= len(seq_in):
        return _RowResult(
            error={**row, "_error": f"position_out_of_range: pos={mut.position} len={len(seq_in)}"}
        )

    if mut.wt not in AA_SET or mut.mut not in AA_SET:
        return _RowResult(error={**row, "_error": "non_standard_amino_acid"})

    sequence_is_mutant = bool(cfg_schema.get("sequence_is_mutant", True))
    seq_aa = seq_in[pos0].upper()
    if sequence_is_mutant:
        # Source sequence is the mutant; the column at pos0 should be mut.mut.
        if seq_aa != mut.mut:
            return _RowResult(
                error={
                    **row,
                    "_error": f"mut_aa_mismatch: seq[{mut.position}]={seq_aa!r} != mut.mut={mut.mut!r}",
                }
            )
        wt_sequence = seq_in[:pos0] + mut.wt + seq_in[pos0 + 1:]
    else:
        # Source sequence is wild-type; column at pos0 should be mut.wt.
        if seq_aa != mut.wt:
            return _RowResult(
                error={
                    **row,
                    "_error": f"wt_aa_mismatch: seq[{mut.position}]={seq_aa!r} != mut.wt={mut.wt!r}",
                }
            )
        wt_sequence = seq_in

    try:
        ddg_norm = _normalize_ddg_sign(float(ddg_raw), cfg_schema["ddg_sign_convention"])
    except (TypeError, ValueError) as e:
        return _RowResult(error={**row, "_error": f"ddg_cast_failed: {e}"})

    label3 = policy.label3(ddg_norm)
    label_binary = policy.label_binary(ddg_norm)

    return _RowResult(ok={
        "split": canonical_split,
        "protein_id": protein_id,
        "wt_sequence": wt_sequence,
        "position_0idx": pos0,
        "position_1idx": mut.position,
        "wt_aa": mut.wt,
        "mut_aa": mut.mut,
        "mutation": mut.to_token(),
        "ddg": ddg_norm,
        "label3": label3,
        "label_binary": label_binary,
    })


def _hf_split_to_dataframe(ds_split, schema: dict[str, Any], primary_split_name: str) -> pd.DataFrame:
    """Filter HF split rows down to the primary split assignment."""
    split_name_col = schema["split_name_col"]
    cols_needed = [
        c for c in {
            schema["protein_id_col"],
            schema["sequence_col"],
            schema["mutation_col"],
            schema["ddg_col"],
            split_name_col,
        } if c is not None
    ]
    cols_in_ds = set(ds_split.column_names)
    missing = [c for c in cols_needed if c not in cols_in_ds]
    if missing:
        raise KeyError(f"required columns missing from split: {missing}")

    df = ds_split.to_pandas()
    df = df[df[split_name_col] == primary_split_name].reset_index(drop=True)
    return df


def normalize_split(
    ds_split,
    cfg: dict[str, Any],
    canonical_split: str,
    primary_split_name: str,
    *,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ok_df, errors_df) for one HF split mapped to a canonical split."""
    schema = cfg["schema"]
    policy = LabelPolicy3(
        stabilizing_threshold=cfg["labels"]["stabilizing_threshold"],
        destabilizing_threshold=cfg["labels"]["destabilizing_threshold"],
    )
    df_raw = _hf_split_to_dataframe(ds_split, schema, primary_split_name)
    if limit is not None:
        df_raw = df_raw.head(limit)

    log.info("[%s] %d rows after filtering split_name=='%s'",
             canonical_split, len(df_raw), primary_split_name)

    oks: list[dict[str, Any]] = []
    errs: list[dict[str, Any]] = []
    for rec in df_raw.to_dict(orient="records"):
        res = _normalize_one_row(rec, schema, canonical_split, policy)
        if res.ok is not None:
            oks.append(res.ok)
        elif res.error is not None:
            errs.append(res.error)

    ok_df = pd.DataFrame(oks)
    err_df = pd.DataFrame(errs)
    return ok_df, err_df


def load_data_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    for key in ("dataset", "schema", "labels", "output"):
        if key not in cfg:
            raise KeyError(f"data config missing required section: {key!r}")
    return cfg


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Per-split summary stats stored in summary.json."""
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "label3_counts": df["label3"].value_counts().to_dict(),
        "label_binary_counts": (
            df["label_binary"].dropna().astype(int).value_counts().to_dict()
        ),
        "ddg_stats": {
            "mean": float(df["ddg"].mean()),
            "std": float(df["ddg"].std()),
            "min": float(df["ddg"].min()),
            "max": float(df["ddg"].max()),
        },
        "sequence_length": {
            "min": int(df["wt_sequence"].str.len().min()),
            "max": int(df["wt_sequence"].str.len().max()),
            "median": float(df["wt_sequence"].str.len().median()),
        },
        "protein_id_unique": int(df["protein_id"].nunique()),
    }
