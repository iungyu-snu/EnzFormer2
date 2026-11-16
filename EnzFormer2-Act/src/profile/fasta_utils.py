"""Query and downloaded-FASTA utilities for the profile pipeline."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from enzformer2.profile.runtime import ensure_dir


def sanitize_seq(seq: str) -> str:
    s = re.sub(r"\s+", "", seq or "").upper()
    # UniProt stream shards can occasionally leave a comma delimiter after a FASTA
    # record. Accept only that terminal artifact; internal punctuation stays invalid.
    if s.endswith(",") and re.fullmatch(r"[A-Z]+,+", s):
        s = s.rstrip(",")
    if not s:
        raise ValueError("empty sequence")
    if not re.fullmatch(r"[A-Z]+", s):
        raise ValueError("sequence must contain alphabetic amino-acid letters only")
    return s


def read_first_fasta(path: Path) -> Tuple[str, str]:
    header = ""
    seq = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    break
                header = line[1:].strip() or "QUERY_INPUT"
                continue
            seq += line
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return header or "QUERY_INPUT", sanitize_seq(seq)


def write_fasta(path: Path, header: str, seq: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i : i + 60] + "\n")


def parse_fasta_records(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line)
    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def extract_accession(header: str) -> Optional[str]:
    m = re.match(r"^(?:sp|tr)\|([^|]+)\|", header or "")
    if m:
        return m.group(1).strip()
    tokens = (header or "").split()
    token = tokens[0] if tokens else ""
    if re.fullmatch(r"[A-Z0-9]{6,10}", token):
        return token
    return None


def sanitize_basename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    s = s.strip("._-")
    return s or "sample"


_MANIFEST_FIELDS: Tuple[str, ...] = (
    "basename",
    "accession",
    "source",
    "length",
    "matched_accession",
    "file_sha256",
    "sequence_sha256",
)


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_MANIFEST_FIELDS), delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                length = int(str(row.get("length") or "0").strip() or "0")
            except (TypeError, ValueError):
                length = 0
            rows.append(
                {
                    "basename": str(row.get("basename") or "").strip(),
                    "accession": str(row.get("accession") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "length": length,
                    "matched_accession": str(row.get("matched_accession") or "").strip(),
                    "file_sha256": str(row.get("file_sha256") or "").strip(),
                    "sequence_sha256": str(row.get("sequence_sha256") or "").strip(),
                }
            )
    return rows


def enzyme_download_prefix(enzyme_type: str, profile: Optional[Dict[str, Any]] = None) -> str:
    base = str(enzyme_type or "").strip()
    if not base and isinstance(profile, dict):
        base = str(profile.get("enzyme_type_normalized") or "").strip()
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "enzyme"
    if len(slug) > 48:
        slug = slug[:48].rstrip("_")
    return slug


def split_single_fastas(
    records: Sequence[Tuple[str, str]],
    single_dir: Path,
    query_seq: str,
    query_name: str = "QUERY_INPUT",
    include_query: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Dedup by accession, then by sequence; write one FASTA per record.

    ``include_query`` controls query handling:

    * ``True`` (default, existing behavior): the query sequence is appended at the
      end with ``source="query"``. If a downloaded sequence is byte-identical to
      the query, its accession is stored on the query row as
      ``matched_accession`` (the byte-identical download is retained).
    * ``False`` (strict, leakage-safe): the query row is **not** appended and any
      downloaded sequence byte-identical to the query is **removed** from the
      corpus. The removed accession(s) are still reported via
      ``stats["query_identical_removed"]`` / ``stats["query_matched_accession"]``
      and on no manifest row, so provenance is preserved without leaking the
      exact query into unsupervised training.

    The default preserves the established manifest behavior.
    """

    ensure_dir(single_dir)
    stats = {
        "input_records": len(records),
        "dedup_accession_removed": 0,
        "dedup_sequence_removed": 0,
        "final_download_records": 0,
        "include_query": bool(include_query),
        "query_identical_removed": 0,
        "query_matched_accession": "",
    }

    dedup_by_acc: List[Tuple[str, str, Optional[str]]] = []
    seen_acc: set[str] = set()
    for header, seq_raw in records:
        seq = sanitize_seq(seq_raw)
        acc = extract_accession(header)
        if acc and acc in seen_acc:
            stats["dedup_accession_removed"] += 1
            continue
        if acc:
            seen_acc.add(acc)
        dedup_by_acc.append((header, seq, acc))

    dedup_final: List[Tuple[str, str, Optional[str]]] = []
    seen_seq: set[str] = set()
    for header, seq, acc in dedup_by_acc:
        if seq in seen_seq:
            stats["dedup_sequence_removed"] += 1
            continue
        seen_seq.add(seq)
        dedup_final.append((header, seq, acc))

    # In strict mode, drop any download byte-identical to the query BEFORE
    # writing, so the exact query sequence never enters the training corpus.
    query_matches = [acc for _, seq, acc in dedup_final if seq == query_seq]
    if not include_query:
        kept: List[Tuple[str, str, Optional[str]]] = []
        for header, seq, acc in dedup_final:
            if seq == query_seq:
                stats["query_identical_removed"] += 1
                continue
            kept.append((header, seq, acc))
        dedup_final = kept
        if query_matches:
            stats["query_matched_accession"] = query_matches[0] or ""

    used_basenames: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    seq_by_acc: Dict[str, str] = {}
    for i, (_, seq, acc) in enumerate(dedup_final, start=1):
        base_raw = acc or f"sample_{i:06d}"
        base = sanitize_basename(base_raw)
        if base in used_basenames:
            used_basenames[base] += 1
            base = f"{base}_{used_basenames[base]}"
        else:
            used_basenames[base] = 1

        out = single_dir / f"{base}.fasta"
        write_fasta(out, base, seq)
        rows.append(
            {
                "basename": base,
                "accession": acc or "",
                "source": "download",
                "length": len(seq),
                "matched_accession": "",
                "file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "sequence_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
            }
        )
        if acc:
            seq_by_acc[acc] = seq

    if include_query:
        matched = [acc for acc, seq in seq_by_acc.items() if seq == query_seq]
        query_basename = sanitize_basename(query_name) or "QUERY_INPUT"
        query_out = single_dir / f"{query_basename}.fasta"
        write_fasta(query_out, query_basename, query_seq)
        rows.append(
            {
                "basename": query_basename,
                "accession": "",
                "source": "query",
                "length": len(query_seq),
                "matched_accession": matched[0] if matched else "",
                "file_sha256": hashlib.sha256(query_out.read_bytes()).hexdigest(),
                "sequence_sha256": hashlib.sha256(query_seq.encode("utf-8")).hexdigest(),
            }
        )
        if matched and not stats["query_matched_accession"]:
            stats["query_matched_accession"] = matched[0]

    stats["final_download_records"] = sum(1 for r in rows if r["source"] == "download")
    return rows, stats
