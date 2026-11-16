#!/usr/bin/env python3
"""Shared point-mutation benchmark evaluator.

Used by the eCTL, IGPS, and extra-DMS benchmark manifests. It:
  - reads common truth CSV variants, including IGPS log10kcat, extra-DMS fitness,
    and eCTL Fitness / Estimated error in fitness schemas.
  - reports threshold-free metrics: spearman, NDCG@10/50/1%, first rank of the
    top 1% by measured fitness, and mean measured fitness in the top 10/50/1%.
  - aligns truth positions to the query WT FASTA via global alignment
    (so prediction IDs like ``eCTL_123PtoA``, ``TmIGPS_123PtoA``, or
    ``P123A`` can be matched).

Prediction CSV schema (same as the single-mutation scanner output):
  - id column auto-detected from {id, mutant_basename, basename, variant_id,
    mut_id, mutation, mutant_id}
  - score column auto-detected from {score, ai_score, p_target_class,
    ensemble_score, prediction, pred_score, y_score, final_score_common}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


BASENAME_RE = re.compile(r"^(?:[A-Za-z0-9_.-]+_)?(\d+)([A-Za-z])to([A-Za-z])$")
MUT_ID_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
COMMON_ID_COLUMNS = (
    "id",
    "mutant_basename",
    "basename",
    "variant_id",
    "mut_id",
    "mutation",
    "mutant_id",
)
COMMON_SCORE_COLUMNS = (
    "final_score_common",
    "ai_score",
    "score",
    "p_target_class",
    "ensemble_score",
    "prediction",
    "pred_score",
    "y_score",
)

# Truth schema aliases used across eCTL, IGPS, and extra-DMS CSVs.
POS_COLS = ("Ambler Position", "pos", "position", "ambler_position")
WT_COLS = ("WT AA", "wtaa", "wt_aa", "wild_type")
MUT_COLS = ("Mutant AA", "mutaa", "mutant_aa", "mut_aa", "mutant")
FITNESS_COLS = ("log10kcat_max", "log10_kcat_max", "fitness", "Fitness")
FITNESS_ERROR_COLS = (
    "Estimated error in fitness",
    "fitness_error",
    "error",
    "fitness_se",
    "stderr",
    "sem",
)
VALID_WT_MISMATCH_POLICIES = ("error", "drop_and_report")


def _normalize_id_prefix(value: Any) -> str:
    text = str(value or "query").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "query"


def _mutation_basename(prefix: Any, pos: int, wt: str, mut: str) -> str:
    return f"{_normalize_id_prefix(prefix)}_{int(pos)}{str(wt).upper()}to{str(mut).upper()}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a mutation-score CSV on a point-mutation truth CSV."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument(
        "--wt-fasta",
        type=Path,
        required=True,
        help="Query WT FASTA; defines the 1-indexed position system.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--query-name",
        type=str,
        default=None,
        help="Short query tag (e.g. TmIGPS); recorded in summary.",
    )
    parser.add_argument("--id-column", type=str, default="auto")
    parser.add_argument("--score-column", type=str, default="auto")
    parser.add_argument(
        "--wt-mismatch-policy",
        choices=VALID_WT_MISMATCH_POLICIES,
        default="error",
        help="How to handle truth rows whose WT residue disagrees with the query FASTA.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _pick_column(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for cand in candidates:
        if cand in row:
            return cand
    return None


def _parse_prediction_id(raw: Any) -> tuple[int, str, str] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = BASENAME_RE.match(text)
    if m is not None:
        pos, wt, mut = m.groups()
        return int(pos), wt.upper(), mut.upper()
    m = MUT_ID_RE.match(text)
    if m is not None:
        wt, pos, mut = m.groups()
        return int(pos), wt.upper(), mut.upper()
    return None


def _detect_column(
    rows: list[dict[str, str]],
    requested: str,
    candidates: tuple[str, ...],
    kind: str,
) -> str:
    if not rows:
        raise ValueError(f"Prediction CSV has no rows; cannot detect {kind} column.")
    if requested != "auto":
        if requested not in rows[0]:
            raise ValueError(f"Requested {kind} column {requested!r} not found.")
        return requested
    for candidate in candidates:
        if candidate in rows[0]:
            return candidate
    raise ValueError(f"Could not auto-detect {kind} column. Available: {sorted(rows[0].keys())}")


def _parse_truth_rows(path: Path, *, query_name: str | None = None) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    if not rows:
        return []
    query_prefix = _normalize_id_prefix(query_name or path.stem)
    pos_col = _pick_column(rows[0], POS_COLS)
    wt_col = _pick_column(rows[0], WT_COLS)
    mut_col = _pick_column(rows[0], MUT_COLS)
    fit_col = _pick_column(rows[0], FITNESS_COLS)
    fit_err_col = _pick_column(rows[0], FITNESS_ERROR_COLS)
    if pos_col is None or wt_col is None or mut_col is None or fit_col is None:
        raise ValueError(
            f"Truth CSV missing required columns (pos/wt/mut/fitness). "
            f"Found: {sorted(rows[0].keys())}"
        )
    parsed: list[dict[str, Any]] = []
    for row in rows:
        wt = str(row.get(wt_col, "")).strip().upper()
        mut = str(row.get(mut_col, "")).strip().upper()
        pos_raw = str(row.get(pos_col, "")).strip()
        if not wt or not mut or not pos_raw:
            continue
        try:
            ambler_position = int(pos_raw)
        except ValueError:
            continue
        fitness = _to_float(row.get(fit_col))
        fitness_error = _to_float(row.get(fit_err_col)) if fit_err_col is not None else None
        parsed.append(
            {
                "ambler_position": ambler_position,
                "wt_aa": wt,
                "mutant_aa": mut,
                "fitness": fitness,
                "fitness_error": fitness_error,
                "has_fitness": fitness is not None,
                "mut_id": f"{wt}{ambler_position}{mut}",
                "mutant_basename": _mutation_basename(query_prefix, ambler_position, wt, mut),
            }
        )
    return parsed


def _read_fasta_sequence(path: Path) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith(">"):
                continue
            lines.append(text.upper())
    sequence = "".join(lines)
    if not sequence:
        raise ValueError(f"FASTA has no sequence: {path}")
    return sequence


def _truth_wt_sequence(rows: list[dict[str, Any]]) -> tuple[list[int], list[str]]:
    wt_by_pos: dict[int, str] = {}
    for row in rows:
        ambler_pos = int(row["ambler_position"])
        wt = str(row["wt_aa"]).strip().upper()
        prev = wt_by_pos.get(ambler_pos)
        if prev is not None and prev != wt:
            raise ValueError(f"Inconsistent WT AA for position {ambler_pos}: {prev!r} vs {wt!r}")
        wt_by_pos[ambler_pos] = wt
    positions = sorted(wt_by_pos)
    residues = [wt_by_pos[pos] for pos in positions]
    return positions, residues


def _global_align(seq_a: str, seq_b: str) -> tuple[str, str]:
    match_score, mismatch_score, gap_score = 2, -3, -2
    n, m = len(seq_a), len(seq_b)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + gap_score
        trace[i][0] = "up"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + gap_score
        trace[0][j] = "left"
    for i in range(1, n + 1):
        a = seq_a[i - 1]
        for j in range(1, m + 1):
            b = seq_b[j - 1]
            diag = scores[i - 1][j - 1] + (match_score if a == b else mismatch_score)
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            best = max(diag, up, left)
            scores[i][j] = best
            trace[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = trace[i][j]
        if move == "diag":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            i -= 1
        elif move == "left":
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            j -= 1
    aligned_a.reverse()
    aligned_b.reverse()
    return "".join(aligned_a), "".join(aligned_b)


def _build_ambler_to_sequence_map(
    fasta_sequence: str,
    ambler_positions: list[int],
    ambler_residues: list[str],
) -> dict[int, int | None]:
    """Map truth Ambler position -> 1-indexed FASTA position.

    Two strategies in order:
      1. Direct: if every truth position's WT AA matches fasta[pos-1], use
         pos->pos identity. (The common IGPS case — positions are already in
         query-FASTA numbering.)
      2. Otherwise, global-align truth-WT against fasta and read out the
         mapping; positions that still fail WT checks are handled later via the
         configured WT-mismatch policy.
    """
    direct: dict[int, int | None] = {}
    all_match = True
    for pos, res in zip(ambler_positions, ambler_residues):
        if 1 <= pos <= len(fasta_sequence) and fasta_sequence[pos - 1] == res:
            direct[pos] = pos
        else:
            direct[pos] = pos  # leave the mapping; per-row WT check will drop mismatches
            all_match = False
    if all_match:
        return direct

    # Fallback: still keep identity mapping but try an alignment-based map for
    # any positions we couldn't validate. The row-level WT check decides
    # whether to fail or drop mismatches.
    aligned_fasta, aligned_truth = _global_align(fasta_sequence, "".join(ambler_residues))
    ambler_to_seq: dict[int, int | None] = dict(direct)
    seq_pos = 0
    truth_idx = 0
    for fasta_char, truth_char in zip(aligned_fasta, aligned_truth):
        if fasta_char != "-":
            seq_pos += 1
        if truth_char == "-":
            continue
        ambler_pos = ambler_positions[truth_idx]
        truth_residue = ambler_residues[truth_idx]
        if fasta_char == "-":
            ambler_to_seq[ambler_pos] = None
        elif fasta_char == truth_residue:
            ambler_to_seq[ambler_pos] = seq_pos
        # else: keep identity mapping; row-level check may still accept or drop.
        truth_idx += 1
    return ambler_to_seq


def _map_truth_row(
    row: dict[str, Any],
    *,
    ambler_to_seq: dict[int, int | None],
    fasta_sequence: str,
    wt_mismatch_policy: str,
    query_name: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    ambler_pos = int(row["ambler_position"])
    wt = str(row["wt_aa"]).strip().upper()
    mut = str(row["mutant_aa"]).strip().upper()
    fitness = row["fitness"]
    fitness_error = row.get("fitness_error")
    if wt == "*" or mut == "*":
        return None, "terminal_stop_row", None
    if mut == wt:
        return None, "self_substitution", None
    seq_pos = ambler_to_seq.get(ambler_pos)
    if seq_pos is None:
        return None, "unmapped_gap", None
    seq_wt = fasta_sequence[int(seq_pos) - 1]
    if seq_wt != wt:
        sample = {
            "ambler_position": ambler_pos,
            "seq_position": int(seq_pos),
            "truth_wt_aa": wt,
            "fasta_wt_aa": seq_wt,
            "mutant_aa": mut,
        }
        if wt_mismatch_policy == "error":
            raise ValueError(
                "WT mismatch between truth CSV and query FASTA: "
                f"ambler_position={ambler_pos} seq_position={int(seq_pos)} "
                f"truth_wt={wt!r} fasta_wt={seq_wt!r} mutant={mut!r}"
            )
        return None, "wt_mismatch", sample
    return (
        {
            "ambler_position": ambler_pos,
            "wt_aa": wt,
            "mutant_aa": mut,
            "fitness": fitness,
            "fitness_error": fitness_error,
            "has_fitness": fitness is not None,
            "seq_position": int(seq_pos),
            "seq_wt_aa": seq_wt,
            "mut_id": f"{wt}{ambler_pos}{mut}",
            "seq_mut_id": f"{seq_wt}{seq_pos}{mut}",
            "mutant_basename": _mutation_basename(query_name or "query", int(seq_pos), seq_wt, mut),
        },
        None,
        None,
    )


def prepare_truth_rows(
    *,
    truth_csv: Path,
    wt_fasta: Path,
    wt_mismatch_policy: str = "error",
    max_drop_samples: int = 5,
    query_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load, align, and validate truth rows against the query WT FASTA."""

    if wt_mismatch_policy not in VALID_WT_MISMATCH_POLICIES:
        raise ValueError(
            "wt_mismatch_policy must be one of "
            f"{VALID_WT_MISMATCH_POLICIES}, got {wt_mismatch_policy!r}."
        )
    if max_drop_samples < 0:
        raise ValueError("max_drop_samples must be >= 0.")

    query_prefix = _normalize_id_prefix(query_name or wt_fasta.stem)
    raw_truth_rows = _parse_truth_rows(truth_csv, query_name=query_prefix)
    fasta_sequence = _read_fasta_sequence(wt_fasta)
    ambler_positions, ambler_residues = _truth_wt_sequence(raw_truth_rows)
    ambler_to_seq = _build_ambler_to_sequence_map(fasta_sequence, ambler_positions, ambler_residues)
    truth_rows: list[dict[str, Any]] = []
    truth_drop_counts: dict[str, int] = {}
    truth_drop_samples: list[dict[str, Any]] = []

    for row in raw_truth_rows:
        mapped, reason, sample = _map_truth_row(
            row,
            ambler_to_seq=ambler_to_seq,
            fasta_sequence=fasta_sequence,
            wt_mismatch_policy=wt_mismatch_policy,
            query_name=query_prefix,
        )
        if mapped is None:
            key = reason or "unknown"
            truth_drop_counts[key] = truth_drop_counts.get(key, 0) + 1
            if sample is not None and len(truth_drop_samples) < max_drop_samples:
                truth_drop_samples.append(sample)
            continue
        truth_rows.append(mapped)

    truth_summary = {
        "truth_row_count_raw": int(len(raw_truth_rows)),
        "truth_row_count_mapped": int(len(truth_rows)),
        "truth_dropped_row_counts": truth_drop_counts,
        "truth_drop_samples": truth_drop_samples,
        "wt_mismatch_policy": wt_mismatch_policy,
    }
    return truth_summary, truth_rows


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = avg_rank
        start = end
    return ranks


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _spearman_rho(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson_correlation(_rankdata(x), _rankdata(y))


def _k_from_fraction(n_rows: int, fraction: float) -> int:
    if n_rows <= 0:
        return 0
    return max(1, int(math.ceil(float(n_rows) * float(fraction))))


def _mean_fitness_top_k(sorted_rows: list[dict[str, Any]], *, k: int) -> float | None:
    if not sorted_rows or k <= 0:
        return None
    k_eff = min(int(k), len(sorted_rows))
    return float(sum(float(r["fitness"]) for r in sorted_rows[:k_eff]) / k_eff)


def _mean_fitness_top_fraction(
    sorted_rows: list[dict[str, Any]],
    *,
    fraction: float,
) -> float | None:
    return _mean_fitness_top_k(sorted_rows, k=_k_from_fraction(len(sorted_rows), fraction))


def _rank_normalized_gains(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    if len(rows) == 1:
        return {str(rows[0]["mut_id"]): 1.0}
    fitness = [float(row["fitness"]) for row in rows]
    ranks = _rankdata(fitness)
    denom = float(len(rows) - 1)
    return {
        str(row["mut_id"]): (float(rank) - 1.0) / denom
        for row, rank in zip(rows, ranks, strict=False)
    }


def _dcg_from_gains(gains: list[float], *, k: int) -> float | None:
    if not gains or k <= 0:
        return None
    k_eff = min(int(k), len(gains))
    return float(sum(float(gains[index]) / math.log2(index + 2.0) for index in range(k_eff)))


def _ndcg_at_k(sorted_rows: list[dict[str, Any]], *, k: int) -> float | None:
    if not sorted_rows or k <= 0:
        return None
    gains_by_id = _rank_normalized_gains(sorted_rows)
    ranked_gains = [gains_by_id[str(row["mut_id"])] for row in sorted_rows]
    ideal_gains = sorted(gains_by_id.values(), reverse=True)
    dcg = _dcg_from_gains(ranked_gains, k=k)
    ideal_dcg = _dcg_from_gains(ideal_gains, k=k)
    if dcg is None or ideal_dcg is None or ideal_dcg <= 0.0:
        return None
    return float(dcg / ideal_dcg)


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _compute_benchmark_mean(
    *,
    spearman_rho: float | None,
    ndcg_mean: float | None,
) -> float | None:
    if spearman_rho is None or ndcg_mean is None:
        return None
    return float(((float(spearman_rho) + 1.0) / 2.0 + float(ndcg_mean)) / 2.0)


def _top_x_pct_precision_at_8(
    eval_rows: list[dict[str, Any]],
    *,
    fraction: float,
) -> tuple[float | None, int, int]:
    """Top-X% Precision@8.

    Positives = top ceil(N * fraction) rows by measured fitness desc, mut_id asc.
    Predictions = top min(8, N) rows by ai_score desc, mut_id asc.
    Returns (precision, top_x_k, pred_at_8_k). Returns (None, 0, 0) on empty input.
    """
    if not eval_rows:
        return None, 0, 0
    n = len(eval_rows)
    top_x_k = max(1, int(math.ceil(float(n) * float(fraction))))
    pred_at_8_k = min(8, n)
    positives_sorted = sorted(
        eval_rows, key=lambda row: (-float(row["fitness"]), str(row["mut_id"]))
    )
    positive_ids = {str(row["mut_id"]) for row in positives_sorted[:top_x_k]}
    predicted_sorted = sorted(
        eval_rows, key=lambda row: (-float(row["ai_score"]), str(row["mut_id"]))
    )
    predicted_ids = [str(row["mut_id"]) for row in predicted_sorted[:pred_at_8_k]]
    if pred_at_8_k <= 0:
        return None, top_x_k, pred_at_8_k
    hits = sum(1 for pid in predicted_ids if pid in positive_ids)
    return float(hits) / float(pred_at_8_k), top_x_k, pred_at_8_k


def activity_ranking_metrics(
    eval_rows: list[dict[str, Any]],
    *,
    score_key: str = "ai_score",
    k: int = 10,
    top_fraction: float = 0.01,
) -> dict[str, float | int | None]:
    """Compute the Act top-candidate metrics from already-matched rows.

    Model scores are sorted descending. Measured fitness is sorted descending
    for the top-fraction reference set, with ``mut_id`` as the deterministic
    tie-breaker. Fractional cutoffs use ``ceil(N * top_fraction)`` and are at
    least one row, matching the benchmark evaluator.
    """

    if k <= 0:
        raise ValueError("k must be positive.")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    rows = [
        row
        for row in eval_rows
        if _to_float(row.get("fitness")) is not None and _to_float(row.get(score_key)) is not None
    ]
    if not rows:
        return {
            "Hit@10": None,
            "Top-1%-hits@10": None,
            "Top1%@10": None,
            "MaxFit@10": None,
            "MeanFit@10": None,
            "Top1%Rank": None,
            "NDCG@10": None,
            "Spearman": None,
            "top_1pct_k": 0,
        }

    ranked = sorted(
        rows,
        key=lambda row: (-float(row[score_key]), str(row.get("mut_id", ""))),
    )
    top_k = min(int(k), len(ranked))
    top_rows = ranked[:top_k]
    top_activity_k = _k_from_fraction(len(rows), float(top_fraction))
    activity_rows = sorted(
        rows,
        key=lambda row: (-float(row["fitness"]), str(row.get("mut_id", ""))),
    )[:top_activity_k]
    activity_ids = {str(row.get("mut_id", "")) for row in activity_rows}
    hits = sum(str(row.get("mut_id", "")) in activity_ids for row in top_rows)
    first_rank = next(
        (
            rank
            for rank, row in enumerate(ranked, start=1)
            if str(row.get("mut_id", "")) in activity_ids
        ),
        None,
    )
    scores = [float(row[score_key]) for row in rows]
    fitness = [float(row["fitness"]) for row in rows]
    return {
        "Hit@10": float(hits > 0),
        "Top-1%-hits@10": int(hits),
        "Top1%@10": int(hits),
        "MaxFit@10": float(max(float(row["fitness"]) for row in top_rows)),
        "MeanFit@10": float(sum(float(row["fitness"]) for row in top_rows) / top_k),
        "Top1%Rank": first_rank,
        "NDCG@10": _ndcg_at_k(ranked, k=top_k),
        "Spearman": _spearman_rho(fitness, scores),
        "top_1pct_k": int(top_activity_k),
    }


def _evaluate_metrics(
    merged_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    correlation_rows = [row for row in merged_rows if bool(row.get("has_fitness"))]
    empty = {
        "correlation_row_count": 0,
        "Hit@10": None,
        "Top-1%-hits@10": None,
        "MaxFit@10": None,
        "MeanFit@10": None,
        "Top1%Rank": None,
        "NDCG@10": None,
        "Spearman": None,
        "spearman_rho": None,
        "ndcg_at_10": None,
        "ndcg_at_50": None,
        "ndcg_at_1pct": None,
        "ndcg_mean": None,
        "benchmark_mean": None,
        "top_10_k": 0,
        "top_50_k": 0,
        "top_1pct_k": 0,
        "mean_fitness_top10": None,
        "mean_fitness_top50": None,
        "mean_fitness_top1pct": None,
        "first_top1pct_relative_activity_rank": None,
        "top1pct_prec_at_8": None,
        "top2pct_prec_at_8": None,
        "top5pct_prec_at_8": None,
        "top1pct_k": 0,
        "top2pct_k": 0,
        "top5pct_k": 0,
        "pred_at_8_k": 0,
    }
    if not correlation_rows:
        return empty

    eval_rows: list[dict[str, Any]] = []
    for row in correlation_rows:
        eval_rows.append({**row, "fitness": float(row["fitness"])})

    act_metrics = activity_ranking_metrics(eval_rows)
    sorted_rows = sorted(eval_rows, key=lambda row: (-float(row["ai_score"]), str(row["mut_id"])))
    top_10_k = min(10, len(sorted_rows))
    top_50_k = min(50, len(sorted_rows))
    top_1pct_k = _k_from_fraction(len(sorted_rows), 0.01)
    ndcg_at_10 = act_metrics["NDCG@10"]
    ndcg_at_50 = _ndcg_at_k(sorted_rows, k=top_50_k)
    ndcg_at_1pct = _ndcg_at_k(sorted_rows, k=top_1pct_k)
    ndcg_mean = _mean_optional([ndcg_at_10, ndcg_at_50, ndcg_at_1pct])
    spearman_rho = act_metrics["Spearman"]
    mean_fit_top10 = act_metrics["MeanFit@10"]
    mean_fit_top50 = _mean_fitness_top_k(sorted_rows, k=50)
    mean_fit_top1pct = _mean_fitness_top_fraction(sorted_rows, fraction=0.01)
    first_top1pct = act_metrics["Top1%Rank"]
    top1pct_prec_at_8, top1pct_k, pred_at_8_k = _top_x_pct_precision_at_8(eval_rows, fraction=0.01)
    top2pct_prec_at_8, top2pct_k, _ = _top_x_pct_precision_at_8(eval_rows, fraction=0.02)
    top5pct_prec_at_8, top5pct_k, _ = _top_x_pct_precision_at_8(eval_rows, fraction=0.05)
    return {
        "correlation_row_count": int(len(eval_rows)),
        "Hit@10": act_metrics["Hit@10"],
        "Top-1%-hits@10": act_metrics["Top-1%-hits@10"],
        "MaxFit@10": act_metrics["MaxFit@10"],
        "MeanFit@10": act_metrics["MeanFit@10"],
        "Top1%Rank": act_metrics["Top1%Rank"],
        "NDCG@10": act_metrics["NDCG@10"],
        "Spearman": act_metrics["Spearman"],
        "spearman_rho": spearman_rho,
        "ndcg_at_10": ndcg_at_10,
        "ndcg_at_50": ndcg_at_50,
        "ndcg_at_1pct": ndcg_at_1pct,
        "ndcg_mean": ndcg_mean,
        "benchmark_mean": _compute_benchmark_mean(
            spearman_rho=spearman_rho,
            ndcg_mean=ndcg_mean,
        ),
        "top_10_k": top_10_k,
        "top_50_k": top_50_k,
        "top_1pct_k": top_1pct_k,
        "mean_fitness_top10": mean_fit_top10,
        "mean_fitness_top50": mean_fit_top50,
        "mean_fitness_top1pct": mean_fit_top1pct,
        "first_top1pct_relative_activity_rank": first_top1pct,
        "top1pct_prec_at_8": top1pct_prec_at_8,
        "top2pct_prec_at_8": top2pct_prec_at_8,
        "top5pct_prec_at_8": top5pct_prec_at_8,
        "top1pct_k": top1pct_k,
        "top2pct_k": top2pct_k,
        "top5pct_k": top5pct_k,
        "pred_at_8_k": pred_at_8_k,
    }


def evaluate(
    *,
    predictions_csv: Path,
    truth_csv: Path,
    wt_fasta: Path,
    id_column: str = "auto",
    score_column: str = "auto",
    wt_mismatch_policy: str = "error",
    query_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Library entrypoint. Returns (summary_dict, merged_rows)."""
    prediction_rows = _read_csv_rows(predictions_csv)
    id_col = _detect_column(prediction_rows, id_column, COMMON_ID_COLUMNS, "id")
    score_col = _detect_column(prediction_rows, score_column, COMMON_SCORE_COLUMNS, "score")

    predictions_by_basename: dict[str, dict[str, Any]] = {}
    predictions_by_mut_id: dict[str, dict[str, Any]] = {}
    invalid_prediction_id_count = 0
    invalid_score_count = 0
    duplicate_prediction_id_count = 0
    for row in prediction_rows:
        parsed = _parse_prediction_id(row.get(id_col))
        score = _to_float(row.get(score_col))
        if parsed is None:
            invalid_prediction_id_count += 1
            continue
        if score is None:
            invalid_score_count += 1
            continue
        pos, wt, mut = parsed
        basename = _mutation_basename(query_name or wt_fasta.stem, pos, wt, mut)
        mut_id = f"{wt}{pos}{mut}"
        normalized = dict(row)
        normalized["mutant_basename"] = basename
        normalized["mut_id"] = mut_id
        normalized["ai_score"] = float(score)
        if basename in predictions_by_basename or mut_id in predictions_by_mut_id:
            duplicate_prediction_id_count += 1
        predictions_by_basename[basename] = normalized
        predictions_by_mut_id[mut_id] = normalized

    truth_summary, truth_rows = prepare_truth_rows(
        truth_csv=truth_csv,
        wt_fasta=wt_fasta,
        wt_mismatch_policy=wt_mismatch_policy,
        query_name=query_name,
    )
    merged_rows: list[dict[str, Any]] = []
    dropped_counts = {
        "prediction_missing": 0,
        "invalid_prediction_id": invalid_prediction_id_count,
        "invalid_ai_score": invalid_score_count,
        "duplicate_prediction_id": duplicate_prediction_id_count,
    }
    for row in truth_rows:
        pred_row = (
            predictions_by_basename.get(str(row["mutant_basename"]))
            or predictions_by_mut_id.get(str(row["seq_mut_id"]))
            or predictions_by_mut_id.get(str(row["mut_id"]))
        )
        if pred_row is None:
            dropped_counts["prediction_missing"] += 1
            continue
        merged_rows.append(
            {
                **row,
                "score_column": score_col,
                "input_id_column": id_col,
                "ai_score": float(pred_row["ai_score"]),
                "prediction_input_id": pred_row.get(id_col),
            }
        )

    metrics = _evaluate_metrics(merged_rows)
    summary = {
        "predictions_csv": str(predictions_csv),
        "truth_csv": str(truth_csv),
        "wt_fasta": str(wt_fasta),
        "input_id_column": id_col,
        "score_column": score_col,
        "matched_prediction_row_count": int(len(merged_rows)),
        "rows_without_fitness": int(sum(1 for r in merged_rows if not bool(r.get("has_fitness")))),
        "dropped_row_counts": dropped_counts,
        **truth_summary,
        **metrics,
    }
    return summary, merged_rows


def main() -> None:
    args = _parse_args()
    predictions_csv = args.predictions_csv.expanduser().resolve()
    truth_csv = args.truth_csv.expanduser().resolve()
    wt_fasta = args.wt_fasta.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    for p, name in [
        (predictions_csv, "predictions_csv"),
        (truth_csv, "truth_csv"),
        (wt_fasta, "wt_fasta"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    summary, merged_rows = evaluate(
        predictions_csv=predictions_csv,
        truth_csv=truth_csv,
        wt_fasta=wt_fasta,
        id_column=str(args.id_column),
        score_column=str(args.score_column),
        wt_mismatch_policy=str(args.wt_mismatch_policy),
        query_name=str(args.query_name or wt_fasta.stem),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = str(args.model_name or predictions_csv.stem).strip()
    query_name = str(args.query_name or wt_fasta.stem).strip()
    out_dir = output_root / f"{query_name}__{model_name}__{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = out_dir / "merged_predictions.csv"
    ranked_csv = out_dir / "ranked_predictions.csv"
    metrics_json = out_dir / "metrics.json"

    summary["model_name"] = model_name
    summary["query_name"] = query_name

    _write_csv_rows(merged_csv, merged_rows)
    _write_csv_rows(
        ranked_csv,
        sorted(merged_rows, key=lambda row: (-float(row["ai_score"]), str(row["mut_id"]))),
    )
    metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[point-mut-bench] query={query_name} model={model_name}")
    print(f"[point-mut-bench] matched_prediction_rows={summary['matched_prediction_row_count']}")
    print(f"[point-mut-bench] spearman_rho={summary['spearman_rho']}")
    print(f"[point-mut-bench] ndcg_mean={summary['ndcg_mean']}")
    print(f"[point-mut-bench] benchmark_mean={summary['benchmark_mean']}")
    print(f"[point-mut-bench] ndcg_at_10={summary['ndcg_at_10']}")
    print(f"[point-mut-bench] ndcg_at_50={summary['ndcg_at_50']}")
    print(f"[point-mut-bench] ndcg_at_1pct={summary['ndcg_at_1pct']}")
    print(f"[point-mut-bench] mean_fitness_top10={summary['mean_fitness_top10']}")
    print(f"[point-mut-bench] mean_fitness_top50={summary['mean_fitness_top50']}")
    print(f"[point-mut-bench] mean_fitness_top1pct={summary['mean_fitness_top1pct']}")
    print(
        f"[point-mut-bench] first_top1pct_relative_activity_rank={summary['first_top1pct_relative_activity_rank']}"
    )
    print(f"[point-mut-bench] metrics_json={metrics_json}")


if __name__ == "__main__":
    main()
