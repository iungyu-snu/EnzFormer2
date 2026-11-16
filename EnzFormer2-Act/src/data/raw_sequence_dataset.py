"""Validated sequence-manifest dataset for EnzFormer2-Act."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset


ID_COLUMNS = ("seq_id", "basename", "accession", "id")
SEQUENCE_COLUMNS = ("sequence", "aa_sequence", "seq")
PATH_COLUMNS = ("sequence_path", "fasta_path", "file", "path")
FASTA_SUFFIXES = (".fasta", ".fa", ".faa")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported sequence manifest format: {path}")


def _read_first_fasta(path: Path) -> str:
    fragments: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if fragments:
                    break
                continue
            fragments.append(line)
    sequence = "".join(fragments).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"FASTA contains no sequence: {path}")
    return sequence


def _resolve_path(value: Any, *, anchor: Path, sequence_root: Path | None, seq_id: str) -> Path:
    if not _missing(value):
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = anchor / candidate
        return candidate.resolve()
    if sequence_root is None:
        raise FileNotFoundError(
            f"No sequence column or sequence file was supplied for sequence {seq_id!r}."
        )
    for suffix in FASTA_SUFFIXES:
        candidate = sequence_root / f"{seq_id}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No FASTA file for sequence {seq_id!r} under {sequence_root}. "
        f"Expected one of: {', '.join(seq_id + suffix for suffix in FASTA_SUFFIXES)}"
    )


class RawSequenceDataset(Dataset[dict[str, Any]]):
    """Load non-empty amino-acid sequences from a validated manifest.

    The manifest contains an identifier and either an inline sequence or a
    reference to a FASTA file. No activity or ecological annotations are read.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sequence_root: str | Path | None = None,
        sequence_column: str | None = None,
        seq_id_column: str | None = None,
        sequence_lookup_path: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Sequence manifest not found: {self.manifest_path}")
        frame = _read_table(self.manifest_path)
        if frame.empty:
            raise ValueError(f"Sequence manifest is empty: {self.manifest_path}")

        self.sequence_root = (
            None if sequence_root is None else Path(sequence_root).expanduser().resolve()
        )
        self.seq_id_column = seq_id_column or next(
            (column for column in ID_COLUMNS if column in frame.columns), None
        )
        if self.seq_id_column is None:
            raise ValueError(f"Manifest needs one of the identifier columns: {ID_COLUMNS}")
        ids = frame[self.seq_id_column]
        if ids.map(_missing).any():
            raise ValueError(f"Manifest contains a missing identifier in {self.seq_id_column!r}.")
        normalized_ids = ids.astype(str).str.strip()
        if (normalized_ids == "").any():
            raise ValueError("Manifest contains an empty sequence identifier.")
        duplicate_ids = sorted(set(normalized_ids[normalized_ids.duplicated()].tolist()))
        if duplicate_ids:
            raise ValueError(
                f"Manifest contains duplicate sequence identifiers: {duplicate_ids[:5]}"
            )

        self.sequence_column = sequence_column or next(
            (column for column in SEQUENCE_COLUMNS if column in frame.columns), None
        )
        lookup: dict[str, Any] = {}
        if sequence_lookup_path is not None:
            lookup_path = Path(sequence_lookup_path).expanduser().resolve()
            if not lookup_path.is_file():
                raise FileNotFoundError(f"Sequence lookup table not found: {lookup_path}")
            lookup_frame = _read_table(lookup_path)
            lookup_id_column = next(
                (column for column in ID_COLUMNS if column in lookup_frame.columns), None
            )
            lookup_sequence_column = next(
                (column for column in SEQUENCE_COLUMNS if column in lookup_frame.columns), None
            )
            if lookup_id_column is None or lookup_sequence_column is None:
                raise ValueError("Sequence lookup table needs an identifier and a sequence column.")
            lookup_ids: set[str] = set()
            for row in lookup_frame.to_dict(orient="records"):
                lookup_id = str(row[lookup_id_column]).strip()
                if not lookup_id:
                    raise ValueError("Sequence lookup table contains an empty identifier.")
                if lookup_id in lookup_ids:
                    raise ValueError(
                        f"Sequence lookup table contains duplicate identifier: {lookup_id}"
                    )
                lookup_ids.add(lookup_id)
                lookup[lookup_id] = row[lookup_sequence_column]

        records: list[dict[str, Any]] = []
        for row, seq_id in zip(
            frame.to_dict(orient="records"), normalized_ids.tolist(), strict=True
        ):
            raw_sequence = (
                row.get(self.sequence_column) if self.sequence_column else lookup.get(seq_id)
            )
            if not _missing(raw_sequence):
                sequence = str(raw_sequence).replace(" ", "").replace("\n", "").upper()
                if not sequence:
                    raise ValueError(f"Sequence {seq_id!r} is empty.")
                sequence_path: Path | None = None
            else:
                sequence_path = _resolve_path(
                    next((row.get(column) for column in PATH_COLUMNS if column in row), None),
                    anchor=self.manifest_path.parent,
                    sequence_root=self.sequence_root,
                    seq_id=seq_id,
                )
                if not sequence_path.is_file():
                    raise FileNotFoundError(
                        f"Sequence file not found for {seq_id!r}: {sequence_path}"
                    )
                sequence = _read_first_fasta(sequence_path)
            sample_weight = row.get("sample_weight", 1.0)
            if _missing(sample_weight):
                sample_weight = 1.0
            try:
                sample_weight = float(sample_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"sample_weight for {seq_id!r} is not numeric.") from exc
            if sample_weight <= 0.0 or not math.isfinite(sample_weight):
                raise ValueError(f"sample_weight for {seq_id!r} must be positive and finite.")
            records.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "sequence_path": None if sequence_path is None else str(sequence_path),
                    "sample_weight": sample_weight,
                }
            )
        if not records:
            raise ValueError("Sequence manifest yielded no sequences.")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return dict(record)


__all__ = ["RawSequenceDataset"]
