"""Offline validation for a frozen enzyme profile and query sequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from enzformer2.profile import fasta_utils
from enzformer2.profile.profile import normalize_profile


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_corpus(manifest_path: Path) -> dict[str, object]:
    """Validate a single-FASTA manifest and return deterministic file hashes.

    Each manifest row must point to ``<manifest directory>/<basename>.fasta``.
    Optional ``file_sha256``/``sha256`` and ``sequence_sha256`` columns are
    checked when present. ``sha256`` is accepted for input compatibility; new
    manifests should use the explicit file and sequence hash names.
    """

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "basename" not in reader.fieldnames:
            raise ValueError(
                f"Frozen corpus manifest must contain a basename column: {manifest_path}"
            )
        for row in reader:
            rows.append({str(key): str(value or "").strip() for key, value in row.items()})

    if not rows:
        raise ValueError(f"Frozen corpus manifest has no sequence rows: {manifest_path}")

    file_hashes: list[tuple[str, str]] = []
    seen_basenames: set[str] = set()
    for row in rows:
        basename = row.get("basename", "")
        if not basename or Path(basename).name != basename or "/" in basename or "\\" in basename:
            raise ValueError(f"Invalid frozen corpus basename {basename!r}: {manifest_path}")
        if basename in seen_basenames:
            raise ValueError(f"Duplicate frozen corpus basename {basename!r}: {manifest_path}")
        seen_basenames.add(basename)
        fasta_path = manifest_path.parent / f"{basename}.fasta"
        if not fasta_path.is_file():
            raise FileNotFoundError(f"Frozen corpus FASTA not found for {basename!r}: {fasta_path}")
        records = fasta_utils.parse_fasta_records(fasta_path)
        if len(records) != 1:
            raise ValueError(f"Frozen corpus FASTA must contain one record: {fasta_path}")
        _, sequence = records[0]
        sequence = fasta_utils.sanitize_seq(sequence)
        if row.get("length"):
            try:
                expected_length = int(row["length"])
            except ValueError as exc:
                raise ValueError(f"Invalid length for {basename!r}: {row['length']!r}") from exc
            if expected_length != len(sequence):
                raise ValueError(
                    f"Length mismatch for {basename!r}: manifest={expected_length}, FASTA={len(sequence)}"
                )
        file_hash = _sha256_file(fasta_path)
        expected_file_hash = next(
            (row.get(column) for column in ("file_sha256", "sha256") if row.get(column)),
            None,
        )
        if expected_file_hash and expected_file_hash.lower() != file_hash:
            raise ValueError(
                f"File checksum mismatch for {basename!r}: "
                f"manifest={expected_file_hash}, file={file_hash}"
            )
        expected_sequence_hash = row.get("sequence_sha256")
        sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if expected_sequence_hash and expected_sequence_hash.lower() != sequence_hash:
            raise ValueError(
                f"Sequence checksum mismatch for {basename!r}: "
                f"manifest={expected_sequence_hash}, sequence={sequence_hash}"
            )
        file_hashes.append((basename, file_hash))

    digest = hashlib.sha256()
    for basename, file_hash in file_hashes:
        digest.update(f"{basename}\t{file_hash}\n".encode("utf-8"))
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "sequence_count": len(file_hashes),
        "corpus_sha256": digest.hexdigest(),
        "checksums_verified": any(
            any(row.get(column) for column in ("file_sha256", "sha256", "sequence_sha256"))
            for row in rows
        ),
    }


def copy_frozen_corpus(manifest_path: Path, destination: Path) -> dict[str, object]:
    """Copy only the validated manifest rows into a run-local corpus directory."""

    summary = validate_frozen_corpus(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, destination / manifest_path.name)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            basename = str(row.get("basename") or "").strip()
            shutil.copy2(
                manifest_path.parent / f"{basename}.fasta", destination / f"{basename}.fasta"
            )
    summary["copied_to"] = str(destination.resolve())
    return summary


def validate_frozen_retrieval_manifest(
    manifest_path: Path,
    *,
    profile_path: Path,
    corpus_manifest: Path | None = None,
) -> dict[str, object]:
    """Validate retrieval metadata and its optional profile/corpus links."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen retrieval manifest must contain an object: {manifest_path}")
    profile_hash = payload.get("profile_sha256")
    if profile_hash and str(profile_hash).lower() != _sha256_file(profile_path):
        raise ValueError(
            f"Frozen profile checksum does not match retrieval manifest: {profile_path}"
        )
    linked_corpus = corpus_manifest
    if linked_corpus is None and payload.get("corpus_manifest"):
        linked_corpus = Path(str(payload["corpus_manifest"]))
        if not linked_corpus.is_absolute():
            linked_corpus = manifest_path.parent / linked_corpus
    result: dict[str, object] = {
        "manifest": str(manifest_path.resolve()),
        "profile_sha256_verified": bool(profile_hash),
    }
    if linked_corpus is not None:
        corpus_summary = validate_frozen_corpus(linked_corpus.expanduser().resolve())
        expected_manifest_hash = payload.get("corpus_manifest_sha256")
        if (
            expected_manifest_hash
            and str(expected_manifest_hash).lower() != corpus_summary["manifest_sha256"]
        ):
            raise ValueError("Frozen corpus manifest checksum does not match retrieval manifest.")
        result["corpus"] = corpus_summary
    return result


def validate_frozen_profile(profile_path: Path, query_fasta: Path) -> dict[str, object]:
    """Validate a profile against one local FASTA record without network access."""

    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    header, query_sequence = fasta_utils.read_first_fasta(query_fasta)
    profile = normalize_profile(raw_profile, query_seq=query_sequence)
    if not profile.get("pfam_id") and not profile.get("interpro_id"):
        raise ValueError("Frozen profile must contain pfam_id or interpro_id.")
    return {
        "profile": str(profile_path.resolve()),
        "query_fasta": str(query_fasta.resolve()),
        "query_name": header or "QUERY_INPUT",
        "query_length": len(query_sequence),
        "pfam_id": profile.get("pfam_id"),
        "interpro_id": profile.get("interpro_id"),
        "confidence": profile.get("confidence"),
        "active_site_rules": len(profile.get("active_site_rules") or []),
        "network_access": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="enzformer2 profile --validate-only",
        description="Validate a frozen profile and query FASTA using local files only.",
    )
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frozen-profile", type=Path, required=True)
    parser.add_argument("--query-fasta", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, default=None)
    parser.add_argument("--retrieval-manifest", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.validate_only:
        parser.error("--validate-only is required for the offline validator.")
    for path, label in (
        (args.frozen_profile, "frozen profile"),
        (args.query_fasta, "query FASTA"),
    ):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    profile_path = args.frozen_profile.expanduser().resolve()
    query_fasta = args.query_fasta.expanduser().resolve()
    result = validate_frozen_profile(profile_path, query_fasta)
    if args.corpus_manifest is not None:
        result["corpus"] = validate_frozen_corpus(args.corpus_manifest.expanduser().resolve())
    if args.retrieval_manifest is not None:
        result["retrieval"] = validate_frozen_retrieval_manifest(
            args.retrieval_manifest.expanduser().resolve(),
            profile_path=profile_path,
            corpus_manifest=(
                args.corpus_manifest.expanduser().resolve() if args.corpus_manifest else None
            ),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
