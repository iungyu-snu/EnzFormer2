#!/usr/bin/env python3
"""Build hashed, leakage-audited EnzFormer2-Act corpus records."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from enzformer2.act import corpus as corpus_mod
from enzformer2.act import provenance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repository_path = (REPOSITORY_ROOT / path).resolve()
    return repository_path if repository_path.exists() else (base / path).resolve()


def _read_fasta_sequence(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return provenance.normalize_sequence(
        "".join(line for line in lines if line and not line.startswith(">"))
    )


def _read_truth_variants(path: Path) -> list[tuple[int, str, str]]:
    variants: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {str(column).lower(): str(column) for column in (reader.fieldnames or [])}
        position = next(
            (columns[name] for name in ("pos", "position", "ambler position") if name in columns),
            None,
        )
        wild_type = next(
            (columns[name] for name in ("wt_aa", "wt aa", "wt") if name in columns), None
        )
        mutant = next(
            (
                columns[name]
                for name in ("mut_aa", "mutant_aa", "mutant aa", "mut")
                if name in columns
            ),
            None,
        )
        if not (position and wild_type and mutant):
            return variants
        for row in reader:
            try:
                pos = int(str(row[position]).strip())
            except (KeyError, TypeError, ValueError):
                continue
            wt_aa = str(row.get(wild_type, "")).strip().upper()
            mut_aa = str(row.get(mutant, "")).strip().upper()
            if wt_aa and len(mut_aa) == 1:
                variants.append((pos, wt_aa, mut_aa))
    return variants


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), dict):
        raise ValueError("corpus manifest must contain a targets mapping")
    return payload


def build_target_corpus(
    target: str,
    spec: dict[str, Any],
    *,
    manifest_base: Path,
    output_root: Path,
    annotate_identity: bool,
) -> dict[str, Any]:
    wt_fasta = _resolve(spec["wt_fasta"], base=manifest_base)
    truth_csv = _resolve(spec["truth_csv"], base=manifest_base)
    mined_dir = _resolve(spec["mined_corpus_dir"], base=manifest_base)
    for path, label in (
        (wt_fasta, "wild-type FASTA"),
        (truth_csv, "truth CSV"),
        (mined_dir, "mined corpus"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{target}: {label} not found: {path}")
    single_dir = mined_dir / "single_fasta"
    manifest_tsv = single_dir / "manifest.tsv"
    query_seq = _read_fasta_sequence(wt_fasta)
    assayed = corpus_mod.build_assayed_mutants(query_seq, _read_truth_variants(truth_csv))
    downloads = corpus_mod.load_single_fasta_dir(
        single_dir, manifest_path=manifest_tsv, include_query_source=False
    )
    if not downloads:
        raise ValueError(f"{target}: mined corpus contains no usable sequences: {single_dir}")
    with_query = downloads + [corpus_mod.CorpusRecord("QUERY_INPUT", "", "query", query_seq)]
    strict = corpus_mod.exclude_exact_query(downloads, query_seq)
    results: dict[str, Any] = {}
    for variant, records in (("with_query", with_query), ("strict_no_query", strict)):
        filtered, removed = corpus_mod.exclude_assayed_mutants(records, assayed)
        if not filtered:
            raise ValueError(f"{target}/{variant}: no sequence remains after leakage filtering")
        if annotate_identity:
            corpus_mod.annotate_identity(filtered, query_seq)
        frozen = corpus_mod.freeze_corpus(
            f"{target}_{variant}",
            filtered,
            query_seq=query_seq,
            wt_seq=query_seq,
            assayed_mutants=assayed,
            source_meta={
                "target": target,
                "variant": variant,
                "mined_corpus_dir": str(mined_dir),
                "n_assayed_mutants": len(assayed),
                "removed_leaked_mutant_ids": removed,
            },
        )
        provenance.assert_no_leakage(frozen.leakage)
        output = output_root / "corpora" / target / variant
        artifacts = corpus_mod.write_corpus_audit(frozen, output, sequence_root=single_dir)
        results[variant] = {
            "corpus_hash": frozen.corpus_hash,
            "n_sequences": frozen.n_sequences,
            "n_assayed_mutants_checked": frozen.leakage.n_assayed_mutants_checked,
            "removed_leaked_mutant_ids": removed,
            "artifacts": artifacts,
        }
    return results


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--annotate-identity", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/enzformer2_act"))
    args = parser.parse_args(argv)
    manifest_path = _resolve(args.manifest, base=Path.cwd())
    payload = _load_manifest(manifest_path)
    output_root = _resolve(args.output_root, base=Path.cwd()) / args.run_id
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "code_commit": _git_commit(),
        "manifest": str(manifest_path),
        "manifest_sha256": provenance.sha256_file(manifest_path),
        "targets": {},
    }
    for target in args.targets:
        spec = payload["targets"].get(target)
        if not isinstance(spec, dict):
            raise KeyError(f"target is not declared in the manifest: {target}")
        result = build_target_corpus(
            target,
            spec,
            manifest_base=manifest_path.parent,
            output_root=output_root,
            annotate_identity=args.annotate_identity,
        )
        summary["targets"][target] = result
        strict = result["strict_no_query"]
        print(
            f"{target}: {strict['n_sequences']} strict sequences; corpus {strict['corpus_hash'][:12]}"
        )
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote corpus records to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
