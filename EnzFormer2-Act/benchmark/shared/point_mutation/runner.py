"""Run an explicit point-mutation benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from benchmark.shared.point_mutation import evaluate as metric_evaluator
from benchmark.shared.point_mutation.act_evaluator import (
    scan_checkpoint,
    scanner_output_to_predictions,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_SOURCES = frozenset({"predictions_csv", "checkpoint_scan"})


def _resolve_path(value: Any, base: Path) -> Path:
    if value is None:
        raise ValueError("A manifest path is required.")
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    manifest_relative = (base / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (_REPO_ROOT / candidate).resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark manifest must contain a mapping.")
    if not isinstance(payload.get("queries"), Mapping) or not isinstance(payload.get("runs"), list):
        raise ValueError("Benchmark manifest requires mapping 'queries' and list 'runs'.")
    return dict(payload)


def validate_manifest(*, manifest_path: Path) -> dict[str, Any]:
    """Validate manifest structure and report every missing local input."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(manifest_path)
    missing: list[str] = []
    errors: list[str] = []
    query_specs = payload["queries"]
    for query, raw_spec in query_specs.items():
        if not isinstance(raw_spec, Mapping):
            errors.append(f"{query}: query specification must be a mapping")
            continue
        for field in ("wt_fasta", "truth_csv"):
            try:
                path = _resolve_path(raw_spec.get(field), manifest_path.parent)
            except ValueError as exc:
                errors.append(f"{query}.{field}: {exc}")
                continue
            if not path.is_file():
                missing.append(f"{query}.{field}: {path}")
    seen: set[tuple[str, str]] = set()
    for index, run in enumerate(payload["runs"]):
        if not isinstance(run, Mapping):
            errors.append(f"runs[{index}] must be a mapping")
            continue
        query = str(run.get("query", "")).strip()
        method = str(run.get("method", "")).strip()
        if not query or query not in query_specs:
            errors.append(f"runs[{index}] refers to an undeclared query")
        if not method:
            errors.append(f"runs[{index}] requires method")
        key = (query, method)
        if key in seen:
            errors.append(f"duplicate benchmark run: {query}/{method}")
        seen.add(key)
        source = str(run.get("source", "predictions_csv"))
        if source not in VALID_SOURCES:
            errors.append(f"{query}/{method}: source must be one of {sorted(VALID_SOURCES)}")
            continue
        path_field = "predictions_csv" if source == "predictions_csv" else "checkpoint"
        try:
            path = _resolve_path(run.get(path_field), manifest_path.parent)
        except ValueError as exc:
            errors.append(f"{query}/{method}.{path_field}: {exc}")
            continue
        if not path.is_file():
            missing.append(f"{query}/{method}.{path_field}: {path}")
    return {
        "manifest": str(manifest_path),
        "missing": missing,
        "errors": errors,
        "valid": not missing and not errors,
    }


def _query_paths(
    payload: Mapping[str, Any], query: str, manifest_path: Path
) -> tuple[Path, Path, str]:
    spec = payload["queries"].get(query)
    if not isinstance(spec, Mapping):
        raise KeyError(f"Unknown benchmark query: {query}")
    return (
        _resolve_path(spec.get("wt_fasta"), manifest_path.parent),
        _resolve_path(spec.get("truth_csv"), manifest_path.parent),
        str(spec.get("seq_id", query)),
    )


def run_manifest(manifest_path: Path, output_root: Path) -> pd.DataFrame:
    report = validate_manifest(manifest_path=manifest_path)
    if not report["valid"]:
        raise ValueError(json.dumps(report, indent=2))
    payload = _load_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        query = str(run["query"])
        method = str(run["method"])
        wt_fasta, truth_csv, seq_id = _query_paths(payload, query, manifest_path)
        run_dir = output_root / f"{query}__{method}"
        source = str(run.get("source", "predictions_csv"))
        wt_mismatch_policy = str(payload["queries"][query].get("wt_mismatch_policy", "error"))
        if source == "predictions_csv":
            predictions = _resolve_path(run["predictions_csv"], manifest_path.parent)
            summary, _ = metric_evaluator.evaluate(
                predictions_csv=predictions,
                truth_csv=truth_csv,
                wt_fasta=wt_fasta,
                id_column=str(run.get("id_column", "auto")),
                score_column=str(run.get("score_column", "auto")),
                wt_mismatch_policy=wt_mismatch_policy,
                query_name=seq_id,
            )
        else:
            checkpoint = _resolve_path(run["checkpoint"], manifest_path.parent)
            sequence = metric_evaluator._read_fasta_sequence(wt_fasta)
            scan = scan_checkpoint(
                checkpoint,
                seq_id=seq_id,
                sequence=sequence,
                chunk_size=int(run.get("chunk_size", 64)),
                device=run.get("device"),
            )
            predictions = run_dir / "scanner_output.csv"
            run_dir.mkdir(parents=True, exist_ok=True)
            scanner_output_to_predictions(scan).to_csv(predictions, index=False)
            summary, _ = metric_evaluator.evaluate(
                predictions_csv=predictions,
                truth_csv=truth_csv,
                wt_fasta=wt_fasta,
                id_column="id",
                score_column="score",
                wt_mismatch_policy=wt_mismatch_policy,
                query_name=seq_id,
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        rows.append({"query": query, "method": method, "source": source, **summary})
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    return aggregate


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="enzformer2 act benchmark", description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = validate_manifest(manifest_path=args.manifest)
    print(json.dumps(report, indent=2))
    if args.validate_only:
        if not report["valid"]:
            raise SystemExit(1)
        return
    if not report["valid"]:
        raise SystemExit(1)
    run_manifest(args.manifest.expanduser().resolve(), args.output_root.expanduser().resolve())


if __name__ == "__main__":  # pragma: no cover
    main()
