"""External single-point ddG benchmark normalization helpers.

External benchmark files are intentionally handled through a pinned YAML config:
column names, sign convention, sequence source, and position convention must be
reviewed before preparation. The output parquets use the same required columns
as :class:`ddg_lora.data.regression_dataset.RegressionDataset`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ddg_lora.constants import (
    AA_SET,
    DEFAULT_DESTABILIZING_MIN,
    DEFAULT_STABILIZING_MAX,
)
from ddg_lora.data.mutation import Mutation, parse_mutation, validate_against_sequence
from ddg_lora.utils.io import load_yaml


_SEPARATED_ONE_LETTER_RE = re.compile(
    r"^\s*([A-Za-z])[\s_\-]+(\d+)[\s_\-]+([A-Za-z])\s*$"
)
_PDB_EXTENSIONS = (".pdb", ".cif", ".mmcif")
_PDB_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class ParsedExternalMutation:
    mutation: Mutation
    pdb_id: str | None = None
    chain_id: str | None = None
    raw_position: str | None = None


@dataclass
class NormalizationResult:
    ok: pd.DataFrame
    errors: pd.DataFrame
    stats: dict[str, Any]


class ExternalBenchmarkConfigError(ValueError):
    """Raised when a benchmark config is incomplete or unsafe to prepare."""


def load_external_benchmark_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    if "benchmarks" not in cfg:
        raise ExternalBenchmarkConfigError("external config missing 'benchmarks'")
    cfg.setdefault("labels", {})
    cfg["labels"].setdefault("stabilizing_threshold", DEFAULT_STABILIZING_MAX)
    cfg["labels"].setdefault("destabilizing_threshold", DEFAULT_DESTABILIZING_MIN)
    cfg.setdefault("output", {})
    cfg["output"].setdefault("processed_dir", "data/processed/external_benchmarks")
    cfg["output"].setdefault("summary_filename", "summary.json")
    cfg["output"].setdefault("schema_used_filename", "schema_used.yaml")
    cfg["output"].setdefault("errors_filename", "errors.csv")
    return cfg


def _parse_one_letter_with_separators(value: str) -> Mutation:
    m = _SEPARATED_ONE_LETTER_RE.match(value)
    if not m:
        raise ValueError(f"could not parse separated mutation token: {value!r}")
    return Mutation(wt=m.group(1).upper(), position=int(m.group(2)), mut=m.group(3).upper())


def _try_parse_mutation_token(value: str) -> Mutation | None:
    try:
        return parse_mutation(value)
    except (TypeError, ValueError):
        pass
    try:
        return _parse_one_letter_with_separators(value)
    except ValueError:
        return None


def parse_external_mutation(value: str) -> ParsedExternalMutation:
    """Parse source mutation strings without broadening the repo's strict parser.

    Accepted examples include ``A123V``, ``A 123 V``, ``p.Ala123Val``,
    ``ALA123VAL``, ``A:A123V``, ``A123V:A``, and ``1ABC_A_A123V``.
    Chain/PDB prefixes are returned as metadata; the mutation position itself is
    left as the source-reported 1-indexed position until normalization applies
    the configured position convention.
    """
    if not isinstance(value, str):
        raise TypeError(f"mutation must be a string, got {type(value).__name__}")
    raw = value.strip()
    if not raw:
        raise ValueError("empty mutation string")

    direct = _try_parse_mutation_token(raw)
    if direct is not None:
        return ParsedExternalMutation(direct, raw_position=str(direct.position))

    parts = [p for p in re.split(r"[:_/|,;]+", raw) if p]
    for idx, part in enumerate(parts):
        mut = _try_parse_mutation_token(part.strip())
        if mut is None:
            continue
        chain_id = None
        pdb_id = None
        if idx > 0 and len(parts[idx - 1]) == 1 and parts[idx - 1].isalpha():
            chain_id = parts[idx - 1].upper()
        if idx + 1 < len(parts) and len(parts[idx + 1]) == 1 and parts[idx + 1].isalpha():
            chain_id = parts[idx + 1].upper()
        if parts and len(parts[0]) == 4 and parts[0].isalnum() and idx != 0:
            pdb_id = parts[0].upper()
        return ParsedExternalMutation(
            mutation=mut,
            pdb_id=pdb_id,
            chain_id=chain_id,
            raw_position=str(mut.position),
        )

    raise ValueError(f"could not parse external mutation string: {value!r}")


def label3_from_ddg(
    ddg: float,
    *,
    stabilizing_threshold: float = DEFAULT_STABILIZING_MAX,
    destabilizing_threshold: float = DEFAULT_DESTABILIZING_MIN,
) -> str:
    if ddg <= stabilizing_threshold:
        return "stabilizing"
    if ddg >= destabilizing_threshold:
        return "destabilizing"
    return "neutral"


def label_binary_from_label3(label3: str) -> int | None:
    if label3 == "stabilizing":
        return 1
    if label3 == "destabilizing":
        return 0
    return None


def labels_for_ddg(ddg: float, labels_cfg: Mapping[str, Any]) -> tuple[str, int | None]:
    label3 = label3_from_ddg(
        float(ddg),
        stabilizing_threshold=float(
            labels_cfg.get("stabilizing_threshold", DEFAULT_STABILIZING_MAX)
        ),
        destabilizing_threshold=float(
            labels_cfg.get("destabilizing_threshold", DEFAULT_DESTABILIZING_MIN)
        ),
    )
    return label3, label_binary_from_label3(label3)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    missing = pd.isna(value)
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _string_or_none(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _required_column(columns: Mapping[str, Any], name: str) -> str:
    col = columns.get(name)
    if not col:
        raise ExternalBenchmarkConfigError(f"columns.{name} must be set")
    return str(col)


def _optional_column(columns: Mapping[str, Any], name: str) -> str | None:
    col = columns.get(name)
    return str(col) if col else None


def _raw_value(row: Mapping[str, Any], col: str | None) -> Any:
    return row.get(col) if col else None


def _coerce_int(value: Any, *, field: str) -> int:
    if _is_missing(value):
        raise ValueError(f"missing integer value for {field}")
    return int(float(value))


def _coerce_aa(value: Any, *, field: str) -> str:
    text = _string_or_none(value)
    if text is None:
        raise ValueError(f"missing amino acid value for {field}")
    aa = text.upper()
    if len(aa) != 1 or aa not in AA_SET:
        raise ValueError(f"noncanonical amino acid for {field}: {value!r}")
    return aa


def _position_0idx(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    parsed: ParsedExternalMutation | None,
    bench_cfg: Mapping[str, Any],
) -> int:
    pos0_col = _optional_column(columns, "position_0idx")
    pos1_col = _optional_column(columns, "position_1idx")
    pos_col = _optional_column(columns, "position")
    offset = int(bench_cfg.get("position_offset", 0) or 0)

    if pos0_col:
        return _coerce_int(row.get(pos0_col), field=pos0_col)
    if pos1_col:
        return _coerce_int(row.get(pos1_col), field=pos1_col) - offset - 1
    if pos_col:
        convention = str(bench_cfg.get("position_convention", "position_1idx"))
        raw_pos = _coerce_int(row.get(pos_col), field=pos_col)
        if convention == "position_0idx":
            return raw_pos
        if convention in {"position_1idx", "mutation_1idx"}:
            return raw_pos - offset - 1
        raise ExternalBenchmarkConfigError(
            "position_convention must be one of position_0idx, position_1idx, mutation_1idx"
        )
    if parsed is None:
        raise ValueError("no mutation or position columns available")
    return parsed.mutation.position - offset - 1


def _mutation_from_row(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    bench_cfg: Mapping[str, Any],
) -> tuple[Mutation, ParsedExternalMutation | None, int]:
    mutation_col = _optional_column(columns, "mutation")
    parsed = None
    if mutation_col and not _is_missing(row.get(mutation_col)):
        parsed = parse_external_mutation(str(row[mutation_col]))
        wt = parsed.mutation.wt
        mut = parsed.mutation.mut
    else:
        wt = _coerce_aa(row.get(_required_column(columns, "wt_aa")), field="wt_aa")
        mut = _coerce_aa(row.get(_required_column(columns, "mut_aa")), field="mut_aa")

    wt_col = _optional_column(columns, "wt_aa")
    mut_col = _optional_column(columns, "mut_aa")
    if wt_col and not _is_missing(row.get(wt_col)):
        wt = _coerce_aa(row.get(wt_col), field=wt_col)
    if mut_col and not _is_missing(row.get(mut_col)):
        mut = _coerce_aa(row.get(mut_col), field=mut_col)

    pos0 = _position_0idx(row, columns, parsed, bench_cfg)
    return Mutation(wt=wt, position=pos0 + 1, mut=mut), parsed, pos0


def find_structure_file(pdb_id: str, pdb_dir: str | Path) -> Path:
    """Locate a PDB or mmCIF file for ``pdb_id`` inside ``pdb_dir``.

    Casing variants and the supported suffixes (``.pdb``, ``.cif``, ``.mmcif``)
    are tried.
    """
    root = Path(pdb_dir)
    candidates: list[Path] = []
    for stem in {pdb_id, pdb_id.lower(), pdb_id.upper()}:
        candidates.extend(root / f"{stem}{ext}" for ext in _PDB_EXTENSIONS)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no PDB/mmCIF file found for {pdb_id!r} under {root}")


def _parse_pdb_chain(path: Path, chain_id: str | None) -> tuple[str, dict[str, int], str]:
    """Small PDB fallback parser for sequence/residue-number mapping.

    It is sufficient for synthetic tests and simple benchmark PDB files. mmCIF
    users should install Biopython and pre-resolve sequence columns when needed.
    """
    chains: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        aa = _PDB_THREE_TO_ONE.get(resname)
        if aa is None:
            continue
        chain = (line[21].strip() or "_").upper()
        resseq = line[22:26].strip()
        icode = line[26].strip()
        resid = f"{resseq}{icode}".strip()
        key = (chain, resid, resname)
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain, []).append((resid, aa))

    if not chains:
        raise ValueError(f"no standard amino-acid residues found in {path}")
    if chain_id:
        chain_key = chain_id.upper()
        if chain_key not in chains:
            raise ValueError(f"chain {chain_id!r} not found in {path}")
    elif len(chains) == 1:
        chain_key = next(iter(chains))
    else:
        raise ValueError(f"ambiguous chains in {path}: {sorted(chains)}")

    residues = chains[chain_key]
    sequence = "".join(aa for _, aa in residues)
    residue_to_index = {resid: i for i, (resid, _) in enumerate(residues)}
    return sequence, residue_to_index, chain_key


def _parse_mmcif_chain(path: Path, chain_id: str | None) -> tuple[str, dict[str, int], str]:
    """Parse an mmCIF chain using Biopython when structure-only normalization is needed."""
    try:
        from Bio.PDB import MMCIFParser
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for mmCIF input. Install the optional external "
            "dependencies with `python -m pip install -e .[external]`."
        ) from exc

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models(), None)
    if model is None:
        raise ValueError(f"no model found in mmCIF file: {path}")

    chains = {chain.id.upper(): chain for chain in model.get_chains()}
    if not chains:
        raise ValueError(f"no chains found in mmCIF file: {path}")
    if chain_id:
        chain_key = chain_id.upper()
        if chain_key not in chains:
            raise ValueError(f"chain {chain_id!r} not found in {path}")
    elif len(chains) == 1:
        chain_key = next(iter(chains))
    else:
        raise ValueError(f"ambiguous chains in {path}: {sorted(chains)}")

    residues: list[tuple[str, str]] = []
    for residue in chains[chain_key].get_residues():
        resname = residue.get_resname().strip().upper()
        aa = _PDB_THREE_TO_ONE.get(resname)
        if aa is None:
            continue
        hetfield, resseq, icode = residue.id
        if str(hetfield).strip():
            continue
        resid = f"{int(resseq)}{str(icode).strip()}".strip()
        residues.append((resid, aa))

    if not residues:
        raise ValueError(f"no standard amino-acid residues found in {path}")
    sequence = "".join(aa for _, aa in residues)
    residue_to_index = {resid: i for i, (resid, _) in enumerate(residues)}
    return sequence, residue_to_index, chain_key


def parse_structure_chain(path: Path, chain_id: str | None) -> tuple[str, dict[str, int], str]:
    """Return sequence and residue-number mapping for PDB or mmCIF structure files."""
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return _parse_pdb_chain(path, chain_id)
    if suffix in {".cif", ".mmcif"}:
        return _parse_mmcif_chain(path, chain_id)
    raise ValueError(f"unsupported structure suffix: {suffix}")


def _resolve_wt_sequence(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    bench_cfg: Mapping[str, Any],
    parsed: ParsedExternalMutation | None,
    mut: Mutation,
    pos0: int,
) -> tuple[str, int, str | None, str | None, str]:
    wt_sequence_col = _optional_column(columns, "wt_sequence") or _optional_column(columns, "sequence")
    seq_kind = str(bench_cfg.get("sequence_kind", "wt_sequence"))
    notes: list[str] = []

    if wt_sequence_col and not _is_missing(row.get(wt_sequence_col)):
        seq = str(row[wt_sequence_col]).strip().upper()
        if not (0 <= pos0 < len(seq)):
            raise ValueError(f"position_0idx {pos0} out of range for sequence length {len(seq)}")
        observed = seq[pos0].upper()
        if seq_kind in {"wt", "wild_type", "wt_sequence"}:
            if observed != mut.wt:
                raise ValueError(
                    f"wt sequence mismatch at pos0={pos0}: observed {observed!r}, expected {mut.wt!r}"
                )
            return seq, pos0, None, None, "direct_wt_sequence"
        if seq_kind in {"mut", "mutant", "mut_sequence", "mutant_sequence"}:
            if observed != mut.mut:
                raise ValueError(
                    f"mutant sequence mismatch at pos0={pos0}: observed {observed!r}, expected {mut.mut!r}"
                )
            wt_sequence = seq[:pos0] + mut.wt + seq[pos0 + 1 :]
            return wt_sequence, pos0, None, None, "reverted_mutant_sequence"
        raise ExternalBenchmarkConfigError(
            "sequence_kind must be wt_sequence or mutant_sequence"
        )

    pdb_dir = bench_cfg.get("pdb_dir")
    if not pdb_dir:
        raise ValueError("no wt_sequence/sequence column and no pdb_dir configured")
    pdb_id_col = _optional_column(columns, "pdb_id")
    chain_col = _optional_column(columns, "chain_id")
    pdb_id = _string_or_none(_raw_value(row, pdb_id_col)) or (parsed.pdb_id if parsed else None)
    chain_id = _string_or_none(_raw_value(row, chain_col)) or (parsed.chain_id if parsed else None)
    if not pdb_id:
        raise ValueError("pdb_dir configured but no pdb_id column/value available")
    structure_path = find_structure_file(pdb_id, pdb_dir)
    sequence, residue_to_index, resolved_chain = parse_structure_chain(structure_path, chain_id)
    raw_position = parsed.raw_position if parsed else str(mut.position)
    if raw_position in residue_to_index:
        mapped_pos0 = residue_to_index[raw_position]
        notes.append(f"pdb_residue_mapped:{raw_position}->{mapped_pos0}")
    else:
        mapped_pos0 = pos0
        notes.append("pdb_residue_mapping_fallback_to_configured_position")
    if not (0 <= mapped_pos0 < len(sequence)):
        raise ValueError(f"mapped PDB position {mapped_pos0} out of range for sequence length {len(sequence)}")
    if sequence[mapped_pos0] != mut.wt:
        raise ValueError(
            f"PDB sequence mismatch at pos0={mapped_pos0}: observed {sequence[mapped_pos0]!r}, expected {mut.wt!r}"
        )
    if mapped_pos0 != pos0:
        notes.append(f"position_0idx_updated:{pos0}->{mapped_pos0}")
    return sequence, mapped_pos0, pdb_id.upper(), resolved_chain, ";".join(notes) or "pdb_sequence"


def normalize_external_dataframe(
    df: pd.DataFrame,
    *,
    benchmark_name: str,
    bench_cfg: Mapping[str, Any],
    labels_cfg: Mapping[str, Any],
    source_path: str | Path | None = None,
    conflict_strategy: str = "fail",
) -> NormalizationResult:
    """Normalize one pinned external benchmark dataframe."""
    if str(bench_cfg.get("task", "single_point_ddg")) != "single_point_ddg":
        raise ExternalBenchmarkConfigError(
            f"{benchmark_name}: only task=single_point_ddg is supported"
        )
    multiplier = bench_cfg.get("ddg_multiplier_to_project")
    if multiplier is None:
        raise ExternalBenchmarkConfigError(
            f"{benchmark_name}: ddg_multiplier_to_project must be pinned to +1.0 or -1.0"
        )
    multiplier = float(multiplier)
    if multiplier not in {-1.0, 1.0}:
        raise ExternalBenchmarkConfigError(
            f"{benchmark_name}: ddg_multiplier_to_project must be +1.0 or -1.0"
        )

    columns = bench_cfg.get("columns") or {}
    if not isinstance(columns, Mapping):
        raise ExternalBenchmarkConfigError(f"{benchmark_name}: columns must be a mapping")
    ddg_col = _required_column(columns, "ddg")
    protein_col = _optional_column(columns, "protein_id")
    mutation_col = _optional_column(columns, "mutation")

    ok_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(df.to_dict(orient="records")):
        try:
            mut, parsed, pos0 = _mutation_from_row(row, columns, bench_cfg)
            wt_sequence, pos0, pdb_id, chain_id, sequence_note = _resolve_wt_sequence(
                row, columns, bench_cfg, parsed, mut, pos0
            )
            if mut.position != pos0 + 1:
                mut = Mutation(wt=mut.wt, position=pos0 + 1, mut=mut.mut)
            raw_ddg = float(row[ddg_col])
            ddg = raw_ddg * multiplier
            label3, label_binary = labels_for_ddg(ddg, labels_cfg)
            canonical_mut = mut.to_token()
            ok_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "source_dataset": bench_cfg.get("source_dataset", benchmark_name),
                    "protein_id": _string_or_none(row.get(protein_col)) or pdb_id,
                    "wt_sequence": wt_sequence,
                    "position_0idx": pos0,
                    "position_1idx": pos0 + 1,
                    "wt_aa": mut.wt,
                    "mut_aa": mut.mut,
                    "mutation": canonical_mut,
                    "ddg": ddg,
                    "label3": label3,
                    "label_binary": label_binary,
                    "raw_ddg": raw_ddg,
                    "raw_ddg_multiplier_to_project": multiplier,
                    "raw_mutation": row.get(mutation_col) if mutation_col else None,
                    "raw_position": parsed.raw_position if parsed else str(pos0 + 1),
                    "raw_row_id": row_index,
                    "raw_source_path": str(source_path) if source_path is not None else None,
                    "pdb_id": pdb_id or (parsed.pdb_id if parsed else None),
                    "chain_id": chain_id or (parsed.chain_id if parsed else None),
                    "normalization_notes": sequence_note,
                }
            )
        except Exception as exc:  # row-level diagnostics are the point here
            error_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "row_index": row_index,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "raw_source_path": str(source_path) if source_path is not None else None,
                }
            )

    ok_df = pd.DataFrame(ok_rows)
    err_df = pd.DataFrame(error_rows)
    ok_df, duplicate_stats, conflict_df = resolve_duplicates_and_conflicts(
        ok_df, labels_cfg=labels_cfg, strategy=conflict_strategy
    )
    if not conflict_df.empty and conflict_strategy == "fail":
        conflict_errors = conflict_df.copy()
        if "benchmark_name" not in conflict_errors.columns:
            conflict_errors.insert(0, "benchmark_name", benchmark_name)
        conflict_errors["reason"] = "conflicting_ddg"
        err_df = pd.concat([err_df, conflict_errors], ignore_index=True)
        conflict_keys = conflict_df[["benchmark_name", "protein_id", "mutation"]].drop_duplicates()
        ok_df = ok_df.merge(
            conflict_keys.assign(_conflicting_ddg_key=True),
            on=["benchmark_name", "protein_id", "mutation"],
            how="left",
        )
        ok_df = ok_df[ok_df["_conflicting_ddg_key"].isna()].drop(
            columns=["_conflicting_ddg_key"]
        ).reset_index(drop=True)

    stats = {
        "rows_in": int(len(df)),
        "rows_ok": int(len(ok_df)),
        "rows_error": int(len(err_df)),
        **duplicate_stats,
    }
    return NormalizationResult(ok=ok_df, errors=err_df, stats=stats)


def resolve_duplicates_and_conflicts(
    df: pd.DataFrame,
    *,
    labels_cfg: Mapping[str, Any],
    strategy: str = "fail",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if df.empty:
        return df, {"n_duplicates": 0, "n_conflicts": 0}, pd.DataFrame()
    if strategy not in {"fail", "average"}:
        raise ValueError("strategy must be 'fail' or 'average'")
    key = ["benchmark_name", "protein_id", "mutation"]
    exact_key = key + ["ddg"]
    n_before = len(df)
    deduped = df.drop_duplicates(subset=exact_key, keep="first").reset_index(drop=True)
    n_duplicates = n_before - len(deduped)

    ddg_counts = deduped.groupby(key, dropna=False)["ddg"].nunique().reset_index(name="n_ddg")
    conflict_keys = ddg_counts[ddg_counts["n_ddg"] > 1]
    if conflict_keys.empty:
        return deduped, {"n_duplicates": int(n_duplicates), "n_conflicts": 0}, pd.DataFrame()

    conflict_df = deduped.merge(conflict_keys[key], on=key, how="inner")
    if strategy == "fail":
        return deduped, {
            "n_duplicates": int(n_duplicates),
            "n_conflicts": int(len(conflict_keys)),
        }, conflict_df

    rows: list[dict[str, Any]] = []
    for _, group in deduped.groupby(key, dropna=False):
        row = group.iloc[0].to_dict()
        mean_ddg = float(group["ddg"].mean())
        label3, label_binary = labels_for_ddg(mean_ddg, labels_cfg)
        row["ddg"] = mean_ddg
        row["label3"] = label3
        row["label_binary"] = label_binary
        note = str(row.get("normalization_notes") or "")
        row["normalization_notes"] = (
            f"{note};conflicting_ddg_averaged_n={len(group)}".strip(";")
        )
        rows.append(row)
    averaged = pd.DataFrame(rows).reset_index(drop=True)
    return averaged, {
        "n_duplicates": int(n_duplicates),
        "n_conflicts": int(len(conflict_keys)),
    }, conflict_df


def benchmark_summary(df: pd.DataFrame, *, source_path: str | Path | None, bench_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "source_path": str(source_path) if source_path is not None else None,
            "ddg_multiplier_to_project": bench_cfg.get("ddg_multiplier_to_project"),
            "pdb_dir": bench_cfg.get("pdb_dir"),
        }
    lengths = df["wt_sequence"].astype(str).str.len()
    return {
        "n_rows": int(len(df)),
        "n_proteins": int(df["protein_id"].nunique(dropna=True)),
        "label3_counts": df["label3"].value_counts().to_dict(),
        "ddg_min": float(df["ddg"].min()),
        "ddg_max": float(df["ddg"].max()),
        "ddg_mean": float(df["ddg"].mean()),
        "sequence_length_min": int(lengths.min()),
        "sequence_length_max": int(lengths.max()),
        "source_path": str(source_path) if source_path is not None else None,
        "ddg_multiplier_to_project": bench_cfg.get("ddg_multiplier_to_project"),
        "pdb_dir": bench_cfg.get("pdb_dir"),
    }


def mutation_overlap_mask(external_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.Series:
    """Match identical substitutions in identical or contained WT sequences.

    Coordinates are shifted by each exact substring occurrence, in either direction.
    This detects cropped domains without requiring protein IDs or ddG values to agree.
    It does not perform approximate/homology alignment.
    """
    mask = pd.Series(False, index=external_df.index)
    if external_df.empty or reference_df.empty:
        return mask
    reference: dict[str, set[Mutation]] = {}
    for sequence, token in reference_df[["wt_sequence", "mutation"]].drop_duplicates().itertuples(index=False):
        mutation = parse_mutation(token)
        validate_against_sequence(sequence, mutation)
        reference.setdefault(sequence, set()).add(mutation)

    hits: set[tuple[str, str]] = set()
    for sequence, group in external_df.groupby("wt_sequence", sort=False):
        tokens = set(group["mutation"])
        for token in tokens:
            validate_against_sequence(sequence, parse_mutation(token))
        mapped: set[str] = set()
        for ref_sequence, mutations in reference.items():
            # A reference domain embedded in an external full-length protein.
            offset = sequence.find(ref_sequence)
            while offset >= 0:
                mapped.update(f"{m.wt}{m.position + offset}{m.mut}" for m in mutations)
                offset = sequence.find(ref_sequence, offset + 1)
            # An external fragment embedded in the reference protein.
            if len(sequence) < len(ref_sequence):
                offset = ref_sequence.find(sequence)
                while offset >= 0:
                    mapped.update(
                        f"{m.wt}{m.position - offset}{m.mut}"
                        for m in mutations if offset < m.position <= offset + len(sequence)
                    )
                    offset = ref_sequence.find(sequence, offset + 1)
        hits.update((sequence, token) for token in tokens & mapped)
    return pd.Series(
        [(s, m) in hits for s, m in external_df[["wt_sequence", "mutation"]].itertuples(index=False)],
        index=external_df.index,
    )


def mutation_overlap_report(
    external_df: pd.DataFrame,
    reference_parquets: Sequence[str | Path],
) -> dict[str, Any]:
    """Report exact and substring-coordinate-corrected mutation overlap."""
    ext = external_df.copy()
    if ext.empty or not reference_parquets:
        return {"n_external_rows": int(len(ext)), "n_overlap_rows": 0, "references": []}
    ext["_overlap_key"] = ext["wt_sequence"].astype(str) + "|" + ext["mutation"].astype(str)
    refs: list[dict[str, Any]] = []
    overlap_keys: set[str] = set()
    for path_like in reference_parquets:
        path = Path(path_like)
        if not path.exists():
            refs.append({"path": str(path), "exists": False, "n_overlap_keys": 0})
            continue
        ref = pd.read_parquet(path)
        if "wt_sequence" not in ref.columns or "mutation" not in ref.columns:
            refs.append({"path": str(path), "exists": True, "schema_ok": False, "n_overlap_keys": None})
            continue
        hits = set(ext.loc[mutation_overlap_mask(ext, ref), "_overlap_key"])
        overlap_keys |= hits
        refs.append({"path": str(path), "exists": True, "schema_ok": True, "n_overlap_keys": int(len(hits))})
    return {
        "n_external_rows": int(len(ext)),
        "n_overlap_rows": int(ext["_overlap_key"].isin(overlap_keys).sum()),
        "n_overlap_keys": int(len(overlap_keys)),
        "references": refs,
        "match_method": "exact_sequence_containment_with_mutation_offset",
    }


def exact_overlap_report(
    external_df: pd.DataFrame,
    reference_parquets: Sequence[str | Path],
) -> dict[str, Any]:
    """Compatibility name; includes coordinate-corrected contained sequences."""
    return mutation_overlap_report(external_df, reference_parquets)
