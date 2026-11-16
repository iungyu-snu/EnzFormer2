"""Source-specific materialization for external single-point ddG benchmarks.

This module bridges raw third-party benchmark archives (FireProt(HF) on Zenodo,
SPURS data bundle, GeoStab repo, ProtDDG-Bench repo, ThermoMPNN repo, S669
Zenodo) and the staged CSV contract consumed by
:mod:`ddg_lora.data.external_benchmarks`.

It does **not** infer ddG sign from data. Each source-specific adapter must be
told its multiplier explicitly, and that multiplier is recorded in the staged
CSV provenance columns. The generic normalizer in
:mod:`ddg_lora.data.external_benchmarks` is reused for label assignment so the
threshold/label semantics never drift between modules.

Public surface
--------------

* dataclasses: :class:`RawMutation`, :class:`StructureChain`,
  :class:`MaterializeResult`.
* helpers: :func:`normalize_pdb_id`, :func:`normalize_chain_id`,
  :func:`parse_raw_mutation_token`, :func:`load_structure_chain`,
  :func:`map_pdb_residue`, :func:`map_structure_index`.
* adapters: :func:`materialize_fireprot_hf`, :func:`materialize_ssym`,
  :func:`materialize_s669`, :func:`materialize_geostab_ddg`.

Callers select an adapter for each benchmark and write the staged CSV and
manifest. A source-materialization CLI is not included in this distribution.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from ddg_lora.constants import AA_SET
from ddg_lora.data.external_benchmarks import (
    _PDB_EXTENSIONS,
    find_structure_file,
    labels_for_ddg,
    mutation_overlap_mask,
    parse_structure_chain,
)


STAGED_COLUMNS: tuple[str, ...] = (
    "benchmark_name",
    "source_dataset",
    "protein_id",
    "wt_sequence",
    "position_0idx",
    "position_1idx",
    "wt_aa",
    "mut_aa",
    "mutation",
    "ddg_project",
    "label3",
    "label_binary",
    "raw_ddg",
    "raw_ddg_multiplier_to_project",
    "raw_mutation",
    "raw_position",
    "raw_row_id",
    "raw_source_path",
    "pdb_id",
    "chain_id",
    "normalization_notes",
)


@dataclass(frozen=True)
class RawMutation:
    """A mutation token where the residue id is preserved as a string.

    ``raw_residue_id`` keeps insertion codes intact (e.g. ``"123A"``). It must
    not be cast to ``int`` until the structure mapping resolves it.
    """

    wt_aa: str
    raw_residue_id: str
    mut_aa: str

    def to_token(self) -> str:
        return f"{self.wt_aa}{self.raw_residue_id}{self.mut_aa}"


@dataclass(frozen=True)
class StructureChain:
    """Sequence of canonical 20 amino acids and PDB residue id -> seq index map."""

    sequence: str
    residue_to_index: Mapping[str, int]
    chain_id: str
    source_path: str


@dataclass
class MaterializeResult:
    """Per-benchmark materialization output.

    Attributes
    ----------
    rows:
        Successfully materialized staged rows (each a dict with keys equal to
        :data:`STAGED_COLUMNS`).
    skipped:
        Rows that failed integrity or mapping. Each dict contains
        ``benchmark_name``, ``raw_row_id``, ``reason``, ``detail``,
        ``raw_record_json`` (a JSON-serialisable string of the source row), and
        ``raw_source_path``.
    info:
        Adapter-level metadata: source paths, sign provenance, raw row count.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class MaterializationNeedsReview(RuntimeError):
    """Raised when a source is found but should not be auto-materialized."""


# ---------------------------------------------------------------------------
# Mutation token parsing
# ---------------------------------------------------------------------------

_PDB_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_ONE_LETTER_RE = re.compile(r"^([A-Z])(\d+[A-Za-z]?)([A-Z])$")
_THREE_LETTER_RE = re.compile(
    r"^p?\.?([A-Za-z]{3})(\d+[A-Za-z]?)([A-Za-z]{3})$"
)
_SEPARATED_ONE_LETTER_RE = re.compile(
    r"^\s*([A-Za-z])[\s_\-]+(\d+[A-Za-z]?)[\s_\-]+([A-Za-z])\s*$"
)


def _strip_pdb_chain_prefix(token: str) -> tuple[str, str | None]:
    """Strip a leading ``CHAIN:`` or trailing ``:CHAIN`` from a mutation token."""
    chain: str | None = None
    if ":" in token:
        head, _, tail = token.partition(":")
        if len(head) == 1 and head.isalpha():
            return tail, head.upper()
        if len(tail) == 1 and tail.isalpha():
            return head, tail.upper()
    return token, chain


def parse_raw_mutation_token(token: str) -> RawMutation:
    """Parse mutation tokens preserving the source residue id.

    Accepted forms include::

        A123V
        A 123 V
        A_123_V
        ALA123VAL
        p.Ala123Val
        A:A123V        (chain prefix stripped, returned mutation only)
        A123V:A        (chain suffix stripped, returned mutation only)

    Insertion codes such as ``A123AV`` are accepted: residue id ``"123A"`` and
    mut ``"V"``.
    """
    if not isinstance(token, str):
        raise TypeError(f"mutation token must be a string, got {type(token).__name__}")
    raw = token.strip()
    if not raw:
        raise ValueError("empty mutation token")

    cleaned, _ = _strip_pdb_chain_prefix(raw)
    cleaned = cleaned.strip()

    one = _ONE_LETTER_RE.match(cleaned.upper())
    if one is not None:
        wt = one.group(1)
        residue_id = one.group(2)
        mut = one.group(3)
        return RawMutation(wt_aa=wt, raw_residue_id=residue_id, mut_aa=mut)

    three = _THREE_LETTER_RE.match(cleaned)
    if three is not None:
        wt3 = three.group(1).upper()
        residue_id = three.group(2)
        mut3 = three.group(3).upper()
        if wt3 not in _PDB_THREE_TO_ONE or mut3 not in _PDB_THREE_TO_ONE:
            raise ValueError(f"unknown 3-letter amino acid in {token!r}")
        return RawMutation(
            wt_aa=_PDB_THREE_TO_ONE[wt3],
            raw_residue_id=residue_id,
            mut_aa=_PDB_THREE_TO_ONE[mut3],
        )

    sep = _SEPARATED_ONE_LETTER_RE.match(cleaned)
    if sep is not None:
        return RawMutation(
            wt_aa=sep.group(1).upper(),
            raw_residue_id=sep.group(2),
            mut_aa=sep.group(3).upper(),
        )

    raise ValueError(f"could not parse mutation token: {token!r}")


# ---------------------------------------------------------------------------
# Source-id helpers
# ---------------------------------------------------------------------------

_PDB_ID_RE = re.compile(r"^([0-9A-Za-z]{4})([_\.\-]([A-Za-z]))?$")


def normalize_pdb_id(value: Any) -> str | None:
    """Return an upper-case PDB id from a noisy source value.

    Accepts forms like ``"1ABC"``, ``"1abc_A"``, ``"1ABC.pdb"``, ``"1ABCA"``.
    Returns ``None`` on missing values.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split("/")[-1]
    text = text.split("\\")[-1]
    for ext in _PDB_EXTENSIONS:
        if text.lower().endswith(ext):
            text = text[: -len(ext)]
            break
    text = text.replace(".", "_")
    if "_" in text:
        head = text.split("_", 1)[0]
        if len(head) >= 4:
            text = head
    if len(text) >= 4 and len(text) <= 5 and re.match(r"^[0-9A-Za-z]{4}([A-Za-z])?$", text):
        return text[:4].upper()
    if _PDB_ID_RE.match(text):
        return text[:4].upper()
    return text.upper()


def normalize_chain_id(value: Any, fallback_from_token: str | None = None) -> str | None:
    """Return an upper-case single-letter chain id, or ``None`` if missing.

    ``fallback_from_token`` is used when the dedicated chain column is absent
    (for example a PDB id of the form ``"1ABC_A"``). It is parsed for a
    trailing chain letter.
    """
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        text = str(value).strip()
        if text:
            return text[:1].upper()
    if fallback_from_token:
        token = str(fallback_from_token).strip()
        for ext in _PDB_EXTENSIONS:
            if token.lower().endswith(ext):
                token = token[: -len(ext)]
                break
        token = token.replace(".", "_")
        if "_" in token:
            tail = token.rsplit("_", 1)[1]
            if len(tail) == 1 and tail.isalpha():
                return tail.upper()
        if len(token) == 5 and token[-1].isalpha():
            return token[-1].upper()
    return None


def load_structure_chain(path: Path, chain_id: str | None) -> StructureChain:
    """Load one chain of a PDB/mmCIF file as a :class:`StructureChain`."""
    sequence, residue_to_index, resolved = parse_structure_chain(path, chain_id)
    return StructureChain(
        sequence=sequence,
        residue_to_index=dict(residue_to_index),
        chain_id=resolved,
        source_path=str(path),
    )


def map_pdb_residue(raw_residue_id: str, structure: StructureChain) -> int:
    """Return the 0-indexed sequence position for ``raw_residue_id`` in chain.

    ``raw_residue_id`` is matched as a string (insertion codes preserved).
    Raises ``KeyError`` if not found.
    """
    if raw_residue_id not in structure.residue_to_index:
        raise KeyError(
            f"PDB residue id {raw_residue_id!r} not found in chain "
            f"{structure.chain_id!r} of {structure.source_path}"
        )
    return int(structure.residue_to_index[raw_residue_id])


def map_structure_index(position_0idx: int, structure: StructureChain) -> int:
    """Validate a 0-based structure-sequence index and return it.

    The check is done eagerly so callers can rely on ``position_0idx`` being
    inside the chain sequence.
    """
    if not (0 <= position_0idx < len(structure.sequence)):
        raise IndexError(
            f"structure index {position_0idx} out of range for chain "
            f"{structure.chain_id!r} of length {len(structure.sequence)}"
        )
    return int(position_0idx)


# ---------------------------------------------------------------------------
# Internal staged-row construction
# ---------------------------------------------------------------------------


def _label_pair(ddg_project: float, labels_cfg: Mapping[str, Any]) -> tuple[str, int | None]:
    return labels_for_ddg(float(ddg_project), labels_cfg)


def _is_canonical_aa(value: str) -> bool:
    return isinstance(value, str) and len(value) == 1 and value in AA_SET


def _record_skip(
    skipped: list[dict[str, Any]],
    *,
    benchmark_name: str,
    raw_row_id: Any,
    raw_source_path: str | Path,
    reason: str,
    detail: str,
    raw_record: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {}
    if raw_record is not None:
        for k, v in raw_record.items():
            if v is None:
                payload[str(k)] = None
            elif isinstance(v, (str, int, float, bool)):
                payload[str(k)] = v
            else:
                payload[str(k)] = str(v)
    skipped.append(
        {
            "benchmark_name": benchmark_name,
            "raw_row_id": raw_row_id,
            "raw_source_path": str(raw_source_path),
            "reason": reason,
            "detail": detail,
            "raw_record_json": _json_dumps_compact(payload),
        }
    )


def _json_dumps_compact(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _add_staged_row(
    rows: list[dict[str, Any]],
    *,
    benchmark_name: str,
    source_dataset: str,
    protein_id: str | None,
    structure: StructureChain | None,
    wt_sequence: str,
    position_0idx: int,
    wt_aa: str,
    mut_aa: str,
    raw_ddg: float,
    raw_multiplier: float,
    raw_mutation: Any,
    raw_position: Any,
    raw_row_id: Any,
    raw_source_path: str | Path,
    pdb_id: str | None,
    chain_id: str | None,
    normalization_notes: str,
    labels_cfg: Mapping[str, Any],
) -> None:
    if not _is_canonical_aa(wt_aa) or not _is_canonical_aa(mut_aa):
        raise ValueError(f"non-canonical amino acid: wt={wt_aa!r}, mut={mut_aa!r}")
    if wt_aa == mut_aa:
        raise ValueError("wt_aa equals mut_aa; not a mutation")
    if not (0 <= position_0idx < len(wt_sequence)):
        raise ValueError(
            f"position_0idx {position_0idx} out of range for sequence of length {len(wt_sequence)}"
        )
    if wt_sequence[position_0idx] != wt_aa:
        raise ValueError(
            f"wt sequence mismatch: sequence[{position_0idx}]={wt_sequence[position_0idx]!r}, expected {wt_aa!r}"
        )
    ddg_project = float(raw_ddg) * float(raw_multiplier)
    label3, label_binary = _label_pair(ddg_project, labels_cfg)
    rows.append(
        {
            "benchmark_name": benchmark_name,
            "source_dataset": source_dataset,
            "protein_id": protein_id,
            "wt_sequence": wt_sequence,
            "position_0idx": int(position_0idx),
            "position_1idx": int(position_0idx) + 1,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "mutation": f"{wt_aa}{int(position_0idx) + 1}{mut_aa}",
            "ddg_project": float(ddg_project),
            "label3": label3,
            "label_binary": label_binary,
            "raw_ddg": float(raw_ddg),
            "raw_ddg_multiplier_to_project": float(raw_multiplier),
            "raw_mutation": None if raw_mutation is None else str(raw_mutation),
            "raw_position": None if raw_position is None else str(raw_position),
            "raw_row_id": raw_row_id,
            "raw_source_path": str(raw_source_path),
            "pdb_id": pdb_id,
            "chain_id": chain_id if chain_id is None else str(chain_id).upper(),
            "normalization_notes": normalization_notes,
        }
    )


def _coerce_float(value: Any) -> float:
    if value is None:
        raise ValueError("missing ddG value")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError("missing ddG value")
    return float(value)


def _table_columns(path: Path) -> set[str]:
    return set(map(str, pd.read_csv(path, nrows=0).columns))


def _record_value(
    record: Mapping[str, Any],
    primary: str,
    aliases: Sequence[str] = (),
) -> Any:
    for key in (primary, *aliases):
        if key in record:
            return record[key]
    raise KeyError(f"missing column {primary!r}")


def _single_sequence_diff(wt_seq: str, mut_seq: str) -> tuple[int, str, str]:
    wt = str(wt_seq).strip().upper()
    mut = str(mut_seq).strip().upper()
    if len(wt) != len(mut):
        raise ValueError(f"wt_seq and mut_seq lengths differ: {len(wt)} != {len(mut)}")
    diffs = [(idx, a, b) for idx, (a, b) in enumerate(zip(wt, mut)) if a != b]
    if len(diffs) != 1:
        raise ValueError(f"expected exactly one sequence difference, found {len(diffs)}")
    return diffs[0]


# ---------------------------------------------------------------------------
# Source-specific adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireProtColumns:
    pdb_id: str = "pdb_id_corrected"
    pdb_position: str = "pdb_position"
    wild_type: str = "wild_type"
    mutation: str = "mutation"
    ddg: str = "ddG"
    chain_id: str = "chain"
    sequence: str = "sequence"


def materialize_fireprot_hf(
    csv_path: Path,
    pdb_dir: Path,
    *,
    labels_cfg: Mapping[str, Any],
    columns: FireProtColumns = FireProtColumns(),
    benchmark_name: str = "fireprot_hf",
    source_dataset: str = "fireprot_hf",
    multiplier: float = 1.0,
    sign_evidence: str = "FireProt Zenodo readme: ddG = mutant - WT",
    position_evidence: str = (
        "FireProt Zenodo: pdb_position is 0-based PDB-file residue index; "
        "the canonical 'sequence' column is preferred when present"
    ),
) -> MaterializeResult:
    """Materialize FireProt(HF) rows.

    Each row carries a canonical ``sequence`` (no X gaps) plus a 0-based
    ``pdb_position``. We use ``sequence`` directly as the wild-type sequence
    when available — this avoids dropping rows whose PDB structures have
    missing residues. PDB chain resolution is still attempted to populate the
    ``chain_id`` provenance column, but it is best-effort.
    """
    df = pd.read_csv(csv_path)
    info = {
        "source_csv": str(csv_path),
        "pdb_dir": str(pdb_dir),
        "n_raw_rows": int(len(df)),
        "raw_ddg_multiplier_to_project": float(multiplier),
        "source_evidence": {
            "sign_convention": sign_evidence,
            "position_convention": position_evidence,
        },
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_row_id, record in enumerate(df.to_dict(orient="records")):
        try:
            pdb_token = record.get(columns.pdb_id)
            pdb_id = normalize_pdb_id(pdb_token)
            if pdb_id is None:
                raise ValueError(f"missing pdb id column {columns.pdb_id!r}")
            chain_value = record.get(columns.chain_id) if columns.chain_id else None
            chain_id = normalize_chain_id(chain_value, fallback_from_token=pdb_token)
            wt_aa = str(record[columns.wild_type]).strip().upper()
            mut_aa = str(record[columns.mutation]).strip().upper()
            position_0idx = int(record[columns.pdb_position])
            ddg = _coerce_float(record[columns.ddg])

            wt_sequence: str | None = None
            note: str
            structure_error: str | None = None
            try:
                structure_path = find_structure_file(pdb_id, pdb_dir)
                structure = load_structure_chain(structure_path, chain_id)
                map_structure_index(position_0idx, structure)
                if structure.sequence[position_0idx] != wt_aa:
                    raise ValueError(
                        f"PDB chain mismatch at pos0={position_0idx}: "
                        f"{structure.sequence[position_0idx]!r} != expected {wt_aa!r}"
                    )
                wt_sequence = structure.sequence
                chain_id = structure.chain_id
                note = "fireprot_pdb_position_as_structure_index"
            except Exception as exc:
                structure_error = f"{type(exc).__name__}: {exc}"
                seq_value = record.get(columns.sequence) if columns.sequence else None
                if isinstance(seq_value, str) and seq_value.strip():
                    candidate = seq_value.strip().upper()
                    if (0 <= position_0idx < len(candidate)
                            and candidate[position_0idx] == wt_aa):
                        wt_sequence = candidate
                        note = (
                            "fireprot_sequence_column_fallback;structure_error="
                            f"{structure_error}"
                        )
                if wt_sequence is None:
                    raise ValueError(structure_error)

            _add_staged_row(
                rows,
                benchmark_name=benchmark_name,
                source_dataset=source_dataset,
                protein_id=f"{pdb_id}_{chain_id}" if chain_id else pdb_id,
                structure=None,
                wt_sequence=wt_sequence,
                position_0idx=position_0idx,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                raw_ddg=ddg,
                raw_multiplier=multiplier,
                raw_mutation=record.get(columns.mutation),
                raw_position=record.get(columns.pdb_position),
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                pdb_id=pdb_id,
                chain_id=chain_id,
                normalization_notes=note,
                labels_cfg=labels_cfg,
            )
        except Exception as exc:
            _record_skip(
                skipped,
                benchmark_name=benchmark_name,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                reason=type(exc).__name__,
                detail=str(exc),
                raw_record=record,
            )
    return MaterializeResult(rows=rows, skipped=skipped, info=info)


@dataclass(frozen=True)
class PdbResidueColumns:
    pdb_id: str = "PDB"
    mutation: str = "MUT"
    ddg: str = "DDG"
    chain_id: str | None = None


def _materialize_pdb_residue_table(
    csv_path: Path,
    pdb_dir: Path,
    *,
    labels_cfg: Mapping[str, Any],
    columns: PdbResidueColumns,
    benchmark_name: str,
    source_dataset: str,
    multiplier: float,
    sign_evidence: str,
    position_evidence: str,
    notes_tag: str,
) -> MaterializeResult:
    df = pd.read_csv(csv_path)
    info = {
        "source_csv": str(csv_path),
        "pdb_dir": str(pdb_dir),
        "n_raw_rows": int(len(df)),
        "raw_ddg_multiplier_to_project": float(multiplier),
        "source_evidence": {
            "sign_convention": sign_evidence,
            "position_convention": position_evidence,
        },
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    structure_cache: dict[tuple[str, str | None], StructureChain] = {}

    for raw_row_id, record in enumerate(df.to_dict(orient="records")):
        try:
            pdb_token = record.get(columns.pdb_id)
            pdb_id = normalize_pdb_id(pdb_token)
            if pdb_id is None:
                raise ValueError(f"missing pdb id column {columns.pdb_id!r}")
            chain_value = record.get(columns.chain_id) if columns.chain_id else None
            chain_id = normalize_chain_id(chain_value, fallback_from_token=pdb_token)
            cache_key = (pdb_id, chain_id)
            if cache_key not in structure_cache:
                structure_path = find_structure_file(pdb_id, pdb_dir)
                structure_cache[cache_key] = load_structure_chain(structure_path, chain_id)
            structure = structure_cache[cache_key]
            mutation_token = str(record[columns.mutation])
            parsed = parse_raw_mutation_token(mutation_token)
            position_0idx = map_pdb_residue(parsed.raw_residue_id, structure)
            ddg = _coerce_float(record[columns.ddg])
            _add_staged_row(
                rows,
                benchmark_name=benchmark_name,
                source_dataset=source_dataset,
                protein_id=f"{pdb_id}_{structure.chain_id}",
                structure=structure,
                wt_sequence=structure.sequence,
                position_0idx=position_0idx,
                wt_aa=parsed.wt_aa,
                mut_aa=parsed.mut_aa,
                raw_ddg=ddg,
                raw_multiplier=multiplier,
                raw_mutation=mutation_token,
                raw_position=parsed.raw_residue_id,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                pdb_id=pdb_id,
                chain_id=structure.chain_id,
                normalization_notes=notes_tag,
                labels_cfg=labels_cfg,
            )
        except Exception as exc:
            _record_skip(
                skipped,
                benchmark_name=benchmark_name,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                reason=type(exc).__name__,
                detail=str(exc),
                raw_record=record,
            )
    return MaterializeResult(rows=rows, skipped=skipped, info=info)


def materialize_ssym(
    csv_path: Path,
    pdb_dir: Path,
    *,
    labels_cfg: Mapping[str, Any],
    benchmark_name: str,
    source_dataset: str,
    multiplier: float,
    sign_evidence: str,
    columns: PdbResidueColumns = PdbResidueColumns(),
    position_evidence: str = "Mutation token references PDB residue numbering",
) -> MaterializeResult:
    return _materialize_pdb_residue_table(
        csv_path,
        pdb_dir,
        labels_cfg=labels_cfg,
        columns=columns,
        benchmark_name=benchmark_name,
        source_dataset=source_dataset,
        multiplier=multiplier,
        sign_evidence=sign_evidence,
        position_evidence=position_evidence,
        notes_tag="ssym_pdb_residue_mapping",
    )


def materialize_s669(
    csv_path: Path,
    pdb_dir: Path | None,
    *,
    labels_cfg: Mapping[str, Any],
    multiplier: float,
    sign_evidence: str,
    columns: PdbResidueColumns = PdbResidueColumns(),
    benchmark_name: str = "s669",
    source_dataset: str = "s669",
    position_evidence: str = "S669 mutation token references PDB residue numbering",
) -> MaterializeResult:
    cols = _table_columns(csv_path)
    if {"name", "wt_seq", "mut_seq"}.issubset(cols):
        return materialize_geostab_ddg(
            csv_path,
            pdb_dir,
            labels_cfg=labels_cfg,
            benchmark_name=benchmark_name,
            source_dataset=source_dataset,
            multiplier=multiplier,
            sign_evidence=sign_evidence,
            position_evidence="GeoStab-style S669 row uses wt_seq/mut_seq single-difference position",
        )
    if {"PDB", "chain", "MUT", "wt_seq", "mut_seq"}.issubset(cols):
        return _materialize_pdb_chain_seq_diff_table(
            csv_path,
            labels_cfg=labels_cfg,
            benchmark_name=benchmark_name,
            source_dataset=source_dataset,
            multiplier=multiplier,
            sign_evidence=sign_evidence,
            position_evidence="ddG_cleaned/S669-style row carries wt_seq/mut_seq alongside PDB+chain+MUT",
        )
    if pdb_dir is None:
        raise ValueError("S669 PDB-residue table requires a pdb_dir")
    return _materialize_pdb_residue_table(
        csv_path,
        pdb_dir,
        labels_cfg=labels_cfg,
        columns=columns,
        benchmark_name=benchmark_name,
        source_dataset=source_dataset,
        multiplier=multiplier,
        sign_evidence=sign_evidence,
        position_evidence=position_evidence,
        notes_tag="s669_pdb_residue_mapping",
    )


@dataclass(frozen=True)
class GeoStabSequenceColumns:
    name: str = "name"
    wt_sequence: str = "wt_seq"
    mut_sequence: str = "mut_seq"
    ddg: str = "ddG"


_GEOSTAB_NAME_RE = re.compile(
    r"^(?:rcsb_)?(?P<pdb>[0-9A-Za-z]{4})_(?P<chain>[A-Za-z0-9])_"
    r"(?P<mutation>[A-Za-z]\d+[A-Za-z]?[A-Za-z])(?:_|$)"
)


def _parse_geostab_name(value: Any) -> tuple[str, str, RawMutation]:
    text = str(value).strip()
    match = _GEOSTAB_NAME_RE.match(text)
    if match is None:
        raise ValueError(f"could not parse GeoStab name field: {value!r}")
    pdb_id = match.group("pdb").upper()
    chain_id = match.group("chain").upper()
    mutation = parse_raw_mutation_token(match.group("mutation"))
    return pdb_id, chain_id, mutation


def _materialize_geostab_sequence_table(
    csv_path: Path,
    *,
    labels_cfg: Mapping[str, Any],
    benchmark_name: str,
    source_dataset: str,
    multiplier: float,
    sign_evidence: str,
    columns: GeoStabSequenceColumns = GeoStabSequenceColumns(),
    position_evidence: str = "GeoStab row provides wt_seq/mut_seq; position is the single sequence difference",
    df: pd.DataFrame | None = None,
) -> MaterializeResult:
    df = pd.read_csv(csv_path) if df is None else df.reset_index(drop=True)
    info = {
        "source_csv": str(csv_path),
        "pdb_dir": None,
        "n_raw_rows": int(len(df)),
        "raw_ddg_multiplier_to_project": float(multiplier),
        "source_evidence": {
            "sign_convention": sign_evidence,
            "position_convention": position_evidence,
        },
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_row_id, record in enumerate(df.to_dict(orient="records")):
        try:
            pdb_id, chain_id, parsed = _parse_geostab_name(record[columns.name])
            wt_seq = str(record[columns.wt_sequence]).strip().upper()
            mut_seq = str(record[columns.mut_sequence]).strip().upper()
            position_0idx, wt_aa, mut_aa = _single_sequence_diff(wt_seq, mut_seq)
            if parsed.wt_aa != wt_aa or parsed.mut_aa != mut_aa:
                raise ValueError(
                    "GeoStab name mutation does not match sequence diff: "
                    f"name={parsed.to_token()}, seq={wt_aa}{position_0idx + 1}{mut_aa}"
                )
            if not parsed.raw_residue_id.isdigit():
                raise ValueError(
                    f"GeoStab sequence rows require numeric mutation position, got {parsed.raw_residue_id!r}"
                )
            if int(parsed.raw_residue_id) != position_0idx + 1:
                raise ValueError(
                    "GeoStab name position does not match sequence diff: "
                    f"name_pos={parsed.raw_residue_id}, seq_pos={position_0idx + 1}"
                )
            ddg = _coerce_float(_record_value(record, columns.ddg, aliases=("DDG", "ddg")))
            _add_staged_row(
                rows,
                benchmark_name=benchmark_name,
                source_dataset=source_dataset,
                protein_id=f"{pdb_id}_{chain_id}",
                structure=None,
                wt_sequence=wt_seq,
                position_0idx=position_0idx,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                raw_ddg=ddg,
                raw_multiplier=multiplier,
                raw_mutation=parsed.to_token(),
                raw_position=parsed.raw_residue_id,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                pdb_id=pdb_id,
                chain_id=chain_id,
                normalization_notes="geostab_sequence_diff_name_validated",
                labels_cfg=labels_cfg,
            )
        except Exception as exc:
            _record_skip(
                skipped,
                benchmark_name=benchmark_name,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                reason=type(exc).__name__,
                detail=str(exc),
                raw_record=record,
            )
    return MaterializeResult(rows=rows, skipped=skipped, info=info)


def _protddg_s2648_key(record: Mapping[str, Any]) -> str:
    pdb_chain = str(_record_value(record, "PDB")).strip().upper()
    if len(pdb_chain) < 5:
        raise ValueError(f"ProtDDG S2648 PDB field must include chain id: {pdb_chain!r}")
    mutation = str(_record_value(record, "MUT")).strip().upper()
    return f"{pdb_chain[:4]}_{pdb_chain[4]}_{mutation}"


def _geostab_key_from_name(value: Any) -> str:
    pdb_id, chain_id, mutation = _parse_geostab_name(value)
    return f"{pdb_id}_{chain_id}_{mutation.to_token()}"


def materialize_s2648_from_geostab_s8754(
    csv_path: Path,
    protddg_s2648_csv: Path | None,
    *,
    labels_cfg: Mapping[str, Any],
    benchmark_name: str,
    source_dataset: str,
    multiplier: float,
    sign_evidence: str,
    position_evidence: str = (
        "S2648 rows are selected by matching ProtDDG-Bench S2648 PDB+chain+mutation "
        "keys against GeoStab S8754 wt_seq/mut_seq rows"
    ),
) -> MaterializeResult:
    if protddg_s2648_csv is None or not Path(protddg_s2648_csv).is_file():
        raise MaterializationNeedsReview(
            "S2648 needs ProtDDG-Bench S2648 split CSV plus GeoStab S8754 sequence CSV"
        )
    protddg_df = pd.read_csv(protddg_s2648_csv)
    wanted_keys = {_protddg_s2648_key(record) for record in protddg_df.to_dict(orient="records")}
    geostab_df = pd.read_csv(csv_path)
    if "name" not in geostab_df.columns:
        raise MaterializationNeedsReview("GeoStab S8754 CSV missing name column")
    matched_mask = geostab_df["name"].map(lambda value: _geostab_key_from_name(value) in wanted_keys)
    matched = geostab_df[matched_mask].reset_index(drop=True)
    if matched.empty:
        raise MaterializationNeedsReview(
            "No GeoStab S8754 rows matched ProtDDG-Bench S2648 keys"
        )
    result = _materialize_geostab_sequence_table(
        csv_path,
        labels_cfg=labels_cfg,
        benchmark_name=benchmark_name,
        source_dataset=source_dataset,
        multiplier=multiplier,
        sign_evidence=sign_evidence,
        position_evidence=position_evidence,
        df=matched,
    )
    result.info["source_csv"] = str(csv_path)
    result.info["protddg_s2648_csv"] = str(protddg_s2648_csv)
    result.info["n_raw_rows"] = int(len(protddg_df))
    result.info["n_geostab_s8754_rows"] = int(len(geostab_df))
    result.info["n_matched_rows_before_validation"] = int(len(matched))
    return result


def _materialize_pdb_chain_seq_diff_table(
    csv_path: Path,
    *,
    labels_cfg: Mapping[str, Any],
    benchmark_name: str,
    source_dataset: str,
    multiplier: float,
    sign_evidence: str,
    position_evidence: str = "wt_seq/mut_seq differ at exactly one position; PDB+chain+MUT validate it",
    df: pd.DataFrame | None = None,
) -> MaterializeResult:
    """Adapt a ``PDB,chain,MUT,wt_seq,mut_seq,DDG`` style table.

    SPURS' GeoStab `ddG_cleaned/<NAME>.csv` (and the parallel S2648.csv) ship
    full wt/mut sequences, so we never need to load PDB structures: the row's
    own sequences carry the canonical mutation position. The mutation token is
    cross-validated against the single sequence diff.
    """
    df = pd.read_csv(csv_path) if df is None else df.reset_index(drop=True)
    info = {
        "source_csv": str(csv_path),
        "pdb_dir": None,
        "n_raw_rows": int(len(df)),
        "raw_ddg_multiplier_to_project": float(multiplier),
        "source_evidence": {
            "sign_convention": sign_evidence,
            "position_convention": position_evidence,
        },
    }
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_row_id, record in enumerate(df.to_dict(orient="records")):
        try:
            pdb_id = normalize_pdb_id(_record_value(record, "PDB", aliases=("pdb",)))
            if pdb_id is None:
                raise ValueError("missing PDB column value")
            chain_value = _record_value(record, "chain", aliases=("CHAIN", "Chain"))
            chain_id = normalize_chain_id(chain_value)
            mutation_token = str(_record_value(record, "MUT", aliases=("mut", "Mut", "mutation"))).strip()
            parsed = parse_raw_mutation_token(mutation_token)
            wt_seq = str(_record_value(record, "wt_seq", aliases=("WT_SEQ", "wt_sequence"))).strip().upper()
            mut_seq = str(_record_value(record, "mut_seq", aliases=("MUT_SEQ", "mut_sequence"))).strip().upper()
            position_0idx, wt_aa, mut_aa = _single_sequence_diff(wt_seq, mut_seq)
            if parsed.wt_aa != wt_aa or parsed.mut_aa != mut_aa:
                raise ValueError(
                    "mutation token does not match sequence diff: "
                    f"token={parsed.to_token()}, seq={wt_aa}{position_0idx + 1}{mut_aa}"
                )
            if parsed.raw_residue_id.isdigit() and int(parsed.raw_residue_id) != position_0idx + 1:
                # Token uses PDB residue numbering, not sequence position. Keep
                # the canonical sequence position from the diff and record the
                # raw residue id for provenance.
                pass
            ddg = _coerce_float(_record_value(record, "DDG", aliases=("ddG", "ddg")))
            _add_staged_row(
                rows,
                benchmark_name=benchmark_name,
                source_dataset=source_dataset,
                protein_id=f"{pdb_id}_{chain_id}" if chain_id else pdb_id,
                structure=None,
                wt_sequence=wt_seq,
                position_0idx=position_0idx,
                wt_aa=wt_aa,
                mut_aa=mut_aa,
                raw_ddg=ddg,
                raw_multiplier=multiplier,
                raw_mutation=mutation_token,
                raw_position=parsed.raw_residue_id,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                pdb_id=pdb_id,
                chain_id=chain_id,
                normalization_notes="pdb_chain_seq_diff_validated",
                labels_cfg=labels_cfg,
            )
        except Exception as exc:
            _record_skip(
                skipped,
                benchmark_name=benchmark_name,
                raw_row_id=raw_row_id,
                raw_source_path=csv_path,
                reason=type(exc).__name__,
                detail=str(exc),
                raw_record=record,
            )
    return MaterializeResult(rows=rows, skipped=skipped, info=info)


def materialize_geostab_ddg(
    csv_path: Path,
    pdb_dir: Path | None,
    *,
    labels_cfg: Mapping[str, Any],
    benchmark_name: str,
    source_dataset: str,
    multiplier: float,
    sign_evidence: str,
    columns: PdbResidueColumns = PdbResidueColumns(chain_id="chain"),
    position_evidence: str = "GeoStab ddG row references PDB residue numbering",
) -> MaterializeResult:
    columns_present = _table_columns(csv_path)
    if {"name", "wt_seq", "mut_seq"}.issubset(columns_present):
        return _materialize_geostab_sequence_table(
            csv_path,
            labels_cfg=labels_cfg,
            benchmark_name=benchmark_name,
            source_dataset=source_dataset,
            multiplier=multiplier,
            sign_evidence=sign_evidence,
            position_evidence=position_evidence,
        )
    if {"PDB", "chain", "MUT", "wt_seq", "mut_seq"}.issubset(columns_present):
        return _materialize_pdb_chain_seq_diff_table(
            csv_path,
            labels_cfg=labels_cfg,
            benchmark_name=benchmark_name,
            source_dataset=source_dataset,
            multiplier=multiplier,
            sign_evidence=sign_evidence,
        )
    if pdb_dir is None:
        raise ValueError("GeoStab PDB-residue table requires a pdb_dir")
    return _materialize_pdb_residue_table(
        csv_path,
        pdb_dir,
        labels_cfg=labels_cfg,
        columns=columns,
        benchmark_name=benchmark_name,
        source_dataset=source_dataset,
        multiplier=multiplier,
        sign_evidence=sign_evidence,
        position_evidence=position_evidence,
        notes_tag="geostab_pdb_residue_mapping",
    )


# ---------------------------------------------------------------------------
# Result helpers consumed by the CLI
# ---------------------------------------------------------------------------


def staged_rows_to_dataframe(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=list(STAGED_COLUMNS))
    if df.empty:
        return df
    df = df[list(STAGED_COLUMNS)]
    return df


def assert_staged_integrity(df: pd.DataFrame) -> None:
    """Sanity-check the materialized frame; raise on any violation."""
    if df.empty:
        return
    missing = [c for c in STAGED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"staged frame missing columns: {missing}")
    bad: list[str] = []
    for _, row in df.iterrows():
        seq = row["wt_sequence"]
        pos = int(row["position_0idx"])
        if not (0 <= pos < len(seq)):
            bad.append("position_out_of_range")
            break
        if seq[pos] != row["wt_aa"]:
            bad.append("wt_sequence_position_mismatch")
            break
        if row["wt_aa"] == row["mut_aa"]:
            bad.append("wt_eq_mut")
            break
    if bad:
        raise ValueError(f"staged frame failed integrity: {bad[0]}")


def drop_megascale_overlaps(
    df: pd.DataFrame,
    reference_parquets: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (kept, dropped, status) with substring-coordinate-corrected matching."""
    if df.empty:
        return df, df.iloc[0:0].copy(), {"status": "empty", "references": []}
    if not reference_parquets:
        return df, df.iloc[0:0].copy(), {"status": "not_checked", "references": []}
    mask_drop = pd.Series(False, index=df.index)
    statuses: list[dict[str, Any]] = []
    for path_like in reference_parquets:
        path = Path(path_like)
        if not path.exists():
            statuses.append({"path": str(path), "exists": False})
            continue
        ref = pd.read_parquet(path)
        if "wt_sequence" not in ref.columns or "mutation" not in ref.columns:
            statuses.append({"path": str(path), "exists": True, "schema_ok": False})
            continue
        mask_drop |= mutation_overlap_mask(df, ref)
        statuses.append({"path": str(path), "exists": True, "schema_ok": True})
    if not statuses or all(not s.get("schema_ok") for s in statuses):
        return df, df.iloc[0:0].copy(), {"status": "not_checked", "references": statuses}
    dropped = df[mask_drop].copy()
    kept = df[~mask_drop].reset_index(drop=True)
    status = "checked" if all(s.get("schema_ok") for s in statuses) else "incomplete"
    return kept, dropped, {
        "status": status, "references": statuses, "n_dropped": int(mask_drop.sum()),
        "match_method": "exact_sequence_containment_with_mutation_offset",
    }


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceCandidate:
    csv: Path
    pdb_dir: Path | None
    label: str
    needs_review_reason: str | None = None


_FIREPROT_HF_NAME_RE = re.compile(r"(homolog(?:ue)?[\W_]*free|\bhf\b)", re.IGNORECASE)


def _exists_dir(path: Path) -> bool:
    return path.exists() and path.is_dir()


def discover_fireprot_hf(roots: Sequence[Path]) -> SourceCandidate | None:
    for root in roots:
        for csv_path in sorted(root.rglob("*.csv")):
            if "fireprot" not in str(csv_path).lower():
                continue
            if not _FIREPROT_HF_NAME_RE.search(csv_path.name):
                continue
            ancestor = csv_path.parent
            for _ in range(4):
                pdb_candidate = ancestor / "pdbs"
                if _exists_dir(pdb_candidate):
                    return SourceCandidate(csv=csv_path, pdb_dir=pdb_candidate, label="fireprot_zenodo")
                pdb_candidate = ancestor / "pdb"
                if _exists_dir(pdb_candidate):
                    return SourceCandidate(csv=csv_path, pdb_dir=pdb_candidate, label="fireprot_zenodo")
                ancestor = ancestor.parent
    return None


def _discover_csv_with_sibling_pdb(
    roots: Sequence[Path],
    *,
    name_pattern: re.Pattern[str],
    sibling_dir_names: Sequence[str],
    label: str,
    prefer_substrings: Sequence[str] = (),
) -> SourceCandidate | None:
    """Find a CSV whose name matches ``name_pattern`` with a sibling structure dir.

    ``prefer_substrings`` ranks earlier matches (case-insensitive) ahead of any
    other candidate. The first CSV under the first ``roots`` entry that owns a
    sibling structure directory wins after the preference filter.
    """
    candidates: list[SourceCandidate] = []
    for root in roots:
        for csv_path in sorted(root.rglob("*.csv")):
            if not name_pattern.search(csv_path.name):
                continue
            ancestor = csv_path.parent
            pdb_match: Path | None = None
            for _ in range(4):
                for name in sibling_dir_names:
                    candidate = ancestor / name
                    if _exists_dir(candidate):
                        pdb_match = candidate
                        break
                if pdb_match is not None:
                    break
                ancestor = ancestor.parent
            if pdb_match is None:
                continue
            candidates.append(SourceCandidate(csv=csv_path, pdb_dir=pdb_match, label=label))

    def rank(candidate: SourceCandidate) -> tuple[int, int, str]:
        s = str(candidate.csv).lower()
        for idx, fragment in enumerate(prefer_substrings):
            if fragment.lower() in s:
                return (0, idx, s)
        return (1, len(prefer_substrings), s)

    candidates.sort(key=rank)
    return candidates[0] if candidates else None


def _discover_csv_by_pattern(
    roots: Sequence[Path],
    *,
    name_pattern: re.Pattern[str],
    label: str,
    needs_review_reason: str | None = None,
    prefer_substrings: Sequence[str] = (),
) -> SourceCandidate | None:
    candidates: list[Path] = []
    for root in roots:
        for csv_path in sorted(root.rglob("*.csv")):
            if name_pattern.search(csv_path.name):
                candidates.append(csv_path)

    def rank(path: Path) -> tuple[int, int, str]:
        s = str(path).lower()
        for idx, fragment in enumerate(prefer_substrings):
            if fragment.lower() in s:
                return (0, idx, s)
        return (1, len(prefer_substrings), s)

    if not candidates:
        return None
    candidates.sort(key=rank)
    return SourceCandidate(
        csv=candidates[0],
        pdb_dir=None,
        label=label,
        needs_review_reason=needs_review_reason,
    )


_SSYM_DIRECT_RE = re.compile(r"ssym.*(dir|direct)", re.IGNORECASE)
_SSYM_INVERSE_RE = re.compile(r"ssym.*(inv|inverse)", re.IGNORECASE)
_S669_RE = re.compile(r"(?<![A-Za-z0-9])s669(?![A-Za-z0-9])", re.IGNORECASE)


def discover_ssym(roots: Sequence[Path], *, kind: str) -> SourceCandidate | None:
    pattern = _SSYM_DIRECT_RE if kind == "direct" else _SSYM_INVERSE_RE
    return _discover_csv_with_sibling_pdb(
        roots,
        name_pattern=pattern,
        sibling_dir_names=("pdb", "pdbs", "structures"),
        label=f"ssym_{kind}",
    )


def discover_s669(roots: Sequence[Path]) -> SourceCandidate | None:
    candidate = _discover_csv_with_sibling_pdb(
        roots,
        name_pattern=_S669_RE,
        sibling_dir_names=("pdb", "pdbs", "structures"),
        label="s669",
        # Prefer the SPURS clean S669 split (PDB+chain combined column with
        # sibling pdb/) over the Zenodo S669 evaluation table.
        prefer_substrings=("s669_clean", "s669_dir", "ddG_cleaned"),
    )
    if candidate is not None:
        return candidate
    return _discover_csv_by_pattern(
        roots,
        name_pattern=_S669_RE,
        label="geostab_s669_sequence",
    )


def _geostab_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", re.IGNORECASE)


def discover_geostab(roots: Sequence[Path], *, name: str) -> SourceCandidate | None:
    # Prefer SPURS `geostab_data/ddG_cleaned/` CSVs (carry full wt/mut sequences)
    # over the `ddG/` table that only stores PDB+chain+MUT.
    return _discover_csv_by_pattern(
        roots,
        name_pattern=_geostab_pattern(name),
        label=f"geostab_{name.lower()}",
        prefer_substrings=("ddg_cleaned", "ddG_cleaned"),
    )


def _find_csv_by_pattern(roots: Sequence[Path], pattern: re.Pattern[str]) -> Path | None:
    for root in roots:
        for csv_path in sorted(root.rglob("*.csv")):
            if pattern.search(str(csv_path)):
                return csv_path
    return None


def _dispatch_s2648(
    csv_path: Path,
    pdb_dir: Path | None,
    **kwargs: Any,
) -> MaterializeResult:
    """Choose the right S2648 path based on the discovered CSV schema.

    SPURS' `geostab_data/ddG_cleaned/S2648.csv` already carries
    ``PDB,chain,MUT,wt_seq,mut_seq,DDG``, so we can route through the same
    sequence-diff adapter as ``s461``/``s8754``. Older discovery results
    (GeoStab S8754 + ProtDDG-Bench S2648 split keys) keep using the join
    adapter.
    """
    cols = _table_columns(csv_path)
    if {"wt_seq", "mut_seq"}.issubset(cols):
        return materialize_geostab_ddg(
            csv_path,
            None,
            benchmark_name="s2648",
            source_dataset="s2648",
            **kwargs,
        )
    return materialize_s2648_from_geostab_s8754(
        csv_path,
        pdb_dir,
        benchmark_name="s2648",
        source_dataset="s2648",
        **kwargs,
    )


def discover_s2648(roots: Sequence[Path]) -> SourceCandidate | None:
    exact = discover_geostab(roots, name="S2648")
    if exact is not None:
        return exact
    geostab_s8754 = _find_csv_by_pattern(roots, _geostab_pattern("S8754"))
    protddg_s2648 = _find_csv_by_pattern(
        roots,
        re.compile(r"S2648[/\\]s2648-10fold-split-0\.csv$", re.IGNORECASE),
    )
    if geostab_s8754 is not None and protddg_s2648 is not None:
        return SourceCandidate(
            csv=geostab_s8754,
            pdb_dir=protddg_s2648,
            label="geostab_s8754_join_protddg_s2648",
        )
    if geostab_s8754 is not None:
        return SourceCandidate(
            csv=geostab_s8754,
            pdb_dir=None,
            label="geostab_s8754_contains_s2648_needs_review",
            needs_review_reason=(
                "GeoStab README states S2648 is included in S8754.csv, but "
                "ProtDDG-Bench S2648/s2648-10fold-split-0.csv is needed to "
                "select the S2648 subset."
            ),
        )
    return None


# Mapping of benchmark -> (discovery callable, adapter callable)


@dataclass(frozen=True)
class BenchmarkBlueprint:
    name: str
    discover: Callable[[Sequence[Path]], SourceCandidate | None]
    adapter: Callable[..., MaterializeResult]
    multiplier: float
    sign_evidence: str
    requires_pdb_dir: bool = True
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def default_blueprints() -> dict[str, BenchmarkBlueprint]:
    """Return the default benchmark adapter dispatch table.

    Sign multipliers reflect the convention that the SPURS public benchmark
    loaders multiply ``row.DDG`` by ``-1`` before feeding the model
    (``ddgBench``/``ddgGeo`` loaders); recording the evidence string here keeps
    that audit trail attached to every materialization run.
    """
    return {
        "fireprot_hf": BenchmarkBlueprint(
            name="fireprot_hf",
            discover=discover_fireprot_hf,
            adapter=materialize_fireprot_hf,
            multiplier=1.0,
            sign_evidence="FireProt Zenodo readme: ddG = mutant - WT",
        ),
        "ssym_direct": BenchmarkBlueprint(
            name="ssym_direct",
            discover=lambda roots: discover_ssym(roots, kind="direct"),
            adapter=lambda csv_path, pdb_dir, **kw: materialize_ssym(
                csv_path,
                pdb_dir,
                benchmark_name="ssym_direct",
                source_dataset="ssym_direct",
                **kw,
            ),
            multiplier=-1.0,
            sign_evidence="SPURS ddgBench loader applies row.DDG * -1",
        ),
        "ssym_inverse": BenchmarkBlueprint(
            name="ssym_inverse",
            discover=lambda roots: discover_ssym(roots, kind="inverse"),
            adapter=lambda csv_path, pdb_dir, **kw: materialize_ssym(
                csv_path,
                pdb_dir,
                benchmark_name="ssym_inverse",
                source_dataset="ssym_inverse",
                **kw,
            ),
            multiplier=-1.0,
            sign_evidence="SPURS ddgBench loader applies row.DDG * -1",
        ),
        "s669": BenchmarkBlueprint(
            name="s669",
            discover=discover_s669,
            adapter=materialize_s669,
            multiplier=-1.0,
            sign_evidence="SPURS ddgBench loader applies row.DDG * -1",
            requires_pdb_dir=False,
        ),
        "s461": BenchmarkBlueprint(
            name="s461",
            discover=lambda roots: discover_geostab(roots, name="S461"),
            adapter=lambda csv_path, pdb_dir, **kw: materialize_geostab_ddg(
                csv_path,
                pdb_dir,
                benchmark_name="s461",
                source_dataset="s461",
                **kw,
            ),
            multiplier=-1.0,
            sign_evidence="SPURS ddgGeo loader applies row.DDG * -1",
            requires_pdb_dir=False,
        ),
        "s783": BenchmarkBlueprint(
            name="s783",
            discover=lambda roots: discover_geostab(roots, name="S783"),
            adapter=lambda csv_path, pdb_dir, **kw: materialize_geostab_ddg(
                csv_path,
                pdb_dir,
                benchmark_name="s783",
                source_dataset="s783",
                **kw,
            ),
            multiplier=-1.0,
            sign_evidence="SPURS ddgGeo loader applies row.DDG * -1",
            requires_pdb_dir=False,
        ),
        "s2648": BenchmarkBlueprint(
            name="s2648",
            discover=discover_s2648,
            adapter=_dispatch_s2648,
            multiplier=-1.0,
            sign_evidence="SPURS ddgGeo loader applies row.DDG * -1",
            requires_pdb_dir=False,
        ),
        "s8754": BenchmarkBlueprint(
            name="s8754",
            discover=lambda roots: discover_geostab(roots, name="S8754"),
            adapter=lambda csv_path, pdb_dir, **kw: materialize_geostab_ddg(
                csv_path,
                pdb_dir,
                benchmark_name="s8754",
                source_dataset="s8754",
                **kw,
            ),
            multiplier=-1.0,
            sign_evidence="SPURS ddgGeo loader applies row.DDG * -1",
            requires_pdb_dir=False,
        ),
    }


def build_local_benchmark_entry(
    *,
    benchmark_name: str,
    raw_path: Path,
) -> dict[str, Any]:
    """Return the local YAML benchmark entry for a successfully materialized CSV."""
    return {
        "enabled": True,
        "task": "single_point_ddg",
        "source_dataset": benchmark_name,
        "raw_path": str(raw_path),
        "pdb_dir": None,
        "sequence_kind": "wt_sequence",
        "position_convention": "position_0idx",
        "position_offset": 0,
        "ddg_multiplier_to_project": 1.0,
        "columns": {
            "protein_id": "protein_id",
            "wt_sequence": "wt_sequence",
            "position_0idx": "position_0idx",
            "wt_aa": "wt_aa",
            "mut_aa": "mut_aa",
            "mutation": "mutation",
            "ddg": "ddg_project",
            "pdb_id": "pdb_id",
            "chain_id": "chain_id",
        },
    }
