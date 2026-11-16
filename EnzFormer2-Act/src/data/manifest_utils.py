"""Small helpers shared by Act manifests and benchmark readers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
DEFAULT_SEQ_ID_COLUMNS = ("seq_id", "basename", "accession", "id")


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(table_path, lines=True)
    if suffix == ".json":
        return pd.read_json(table_path)
    raise ValueError(f"Unsupported table format: {table_path}")


__all__ = [
    "DEFAULT_AA_ALPHABET",
    "DEFAULT_SEQ_ID_COLUMNS",
    "_read_table",
]
