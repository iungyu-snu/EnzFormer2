"""PyTorch-style dataset for ddG regression on the normalized MegaScale parquet.

Each example carries both the wild-type sequence and the mutant sequence
(generated from the wild-type by replacing the residue at ``position_0idx``
with ``mut_aa``). Both sequences are required because the regression models
defined in :mod:`ddg_lora.models.regressors` consume them directly.

Schema (per :class:`RegressionExample`):

    protein_id        str | None
    wt_sequence       str
    mut_sequence      str
    position_0idx     int
    wt_aa             str   (1-letter)
    mut_aa            str   (1-letter)
    wt_token_index    int   (index into AMINO_ACIDS_20, 0..19)
    mut_token_index   int   (index into AMINO_ACIDS_20, 0..19)
    ddg               float (kcal/mol, positive = destabilizing)
    label3            str   ("stabilizing" | "neutral" | "destabilizing")
    label_binary      int | None  (1 stab, 0 destab, None for neutral)
    mutation          str
    masked_logodds_forward  float | None
    masked_logodds_reverse  float | None

Filtering rules:

  - ``include_neutral=True`` (default) retains neutral rows; ``False`` excludes them.
    Both paper training configurations exclude neutral rows; the
    ``include_neutral`` ablation retains them.
  - WT integrity (``wt_sequence[pos] == wt_aa``) is validated.
  - ``wt_aa != mut_aa`` is validated.
  - Sequences longer than ``max_sequence_length`` (if set) are rejected with
    an explanatory error rather than silently truncated.

Each dataset item is a :class:`RegressionExample`. The trainer groups examples
into microbatches and uses batch prediction when supported by the backbone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ddg_lora.constants import AA_SET, AMINO_ACIDS_20
from ddg_lora.utils.logging import get_logger

log = get_logger("regression_dataset")


_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS_20)}


@dataclass
class RegressionExample:
    protein_id: str | None
    wt_sequence: str
    mut_sequence: str
    position_0idx: int
    wt_aa: str
    mut_aa: str
    wt_token_index: int
    mut_token_index: int
    ddg: float
    label3: str
    label_binary: int | None
    mutation: str
    masked_logodds_forward: float | None = None
    masked_logodds_reverse: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _apply_single_mutation(wt_sequence: str, position_0idx: int, mut_aa: str) -> str:
    """Return ``wt_sequence`` with the residue at ``position_0idx`` replaced by ``mut_aa``."""
    return wt_sequence[:position_0idx] + mut_aa + wt_sequence[position_0idx + 1 :]


class RegressionDataset:
    """A torch-style dataset over normalized MegaScale rows for ddG regression.

    Implemented without inheriting from ``torch.utils.data.Dataset`` so the
    pure-Python tests do not need to import torch. ``__len__`` and
    ``__getitem__`` still work with PyTorch ``DataLoader`` if needed.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        include_neutral: bool = True,
        max_examples: int | None = None,
        max_sequence_length: int | None = None,
        verify_wt: bool = True,
    ):
        self.parquet_path = Path(parquet_path)
        self.include_neutral = bool(include_neutral)
        self.max_sequence_length = (
            int(max_sequence_length) if max_sequence_length is not None else None
        )

        df = pd.read_parquet(self.parquet_path)
        n_total = len(df)

        if not include_neutral:
            df = df[df["label3"] != "neutral"].reset_index(drop=True)
        log.info(
            "[%s] %d/%d rows after include_neutral=%s",
            self.parquet_path.name, len(df), n_total, include_neutral,
        )

        if verify_wt:
            self._validate_integrity_or_raise(df)

        if self.max_sequence_length is not None:
            too_long = df[df["wt_sequence"].str.len() > self.max_sequence_length]
            if len(too_long):
                raise ValueError(
                    f"{len(too_long)} rows exceed max_sequence_length="
                    f"{self.max_sequence_length}. Refusing to truncate; "
                    "raise the config or filter upstream."
                )

        if max_examples is not None:
            df = df.head(int(max_examples)).reset_index(drop=True)
        self._df = df

    @staticmethod
    def _validate_integrity_or_raise(df: pd.DataFrame) -> None:
        if df.empty:
            return
        errors: list[dict[str, object]] = []
        for i, row in df.iterrows():
            seq = row["wt_sequence"]
            pos = int(row["position_0idx"])
            wt = row["wt_aa"]
            mut = row["mut_aa"]
            if not (0 <= pos < len(seq)):
                reason = "position_0idx_out_of_range"
            elif seq[pos].upper() != str(wt).upper():
                reason = "wt_sequence_position_mismatch"
            elif wt not in AA_SET or mut not in AA_SET:
                reason = "noncanonical_amino_acid"
            elif wt == mut:
                reason = "wt_aa_equals_mut_aa"
            else:
                continue
            errors.append(
                {
                    "row_index": int(i),
                    "reason": reason,
                    "protein_id": row.get("protein_id"),
                    "mutation": row.get("mutation"),
                    "position_0idx": pos,
                    "wt_aa": wt,
                    "mut_aa": mut,
                }
            )
        if errors:
            preview = errors[:5]
            raise ValueError(
                f"{len(errors)}/{len(df)} rows failed regression dataset integrity checks; "
                f"first_errors={preview}"
            )

    # ----- python protocol ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> RegressionExample:
        row = self._df.iloc[idx]
        wt_seq: str = row["wt_sequence"]
        pos0: int = int(row["position_0idx"])
        wt_aa: str = row["wt_aa"]
        mut_aa: str = row["mut_aa"]

        # Defensive: the upstream filter already enforces this, but if a caller
        # bypasses verification we still want a clear error before tokenization.
        if wt_seq[pos0].upper() != wt_aa.upper():
            raise ValueError(
                f"WT integrity violated at idx={idx}: "
                f"sequence[{pos0}]={wt_seq[pos0]!r}, expected {wt_aa!r}"
            )
        if wt_aa == mut_aa:
            raise ValueError(
                f"wt_aa == mut_aa at idx={idx}; not a mutation: {wt_aa!r}"
            )
        if mut_aa not in _AA_INDEX or wt_aa not in _AA_INDEX:
            raise ValueError(
                f"non-canonical amino acid at idx={idx}: wt={wt_aa!r}, mut={mut_aa!r}"
            )

        mut_seq = _apply_single_mutation(wt_seq, pos0, mut_aa)
        forward = row.get("masked_logodds_forward", None)
        reverse = row.get("masked_logodds_reverse", None)

        return RegressionExample(
            protein_id=(None if pd.isna(row.get("protein_id")) else str(row["protein_id"])),
            wt_sequence=wt_seq,
            mut_sequence=mut_seq,
            position_0idx=pos0,
            wt_aa=wt_aa,
            mut_aa=mut_aa,
            wt_token_index=_AA_INDEX[wt_aa],
            mut_token_index=_AA_INDEX[mut_aa],
            ddg=float(row["ddg"]),
            label3=str(row["label3"]),
            label_binary=(None if pd.isna(row["label_binary"]) else int(row["label_binary"])),
            mutation=str(row["mutation"]),
            masked_logodds_forward=(
                None if forward is None or pd.isna(forward) else float(forward)
            ),
            masked_logodds_reverse=(
                None if reverse is None or pd.isna(reverse) else float(reverse)
            ),
        )

    def iter_examples(self) -> Iterable[RegressionExample]:
        for i in range(len(self)):
            yield self[i]

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def label3_counts(self) -> dict[str, int]:
        return self._df["label3"].value_counts().to_dict() if not self._df.empty else {}
