"""Build filtered frozen corpora and write provenance artifacts.

A *frozen corpus* is an immutable, hashed set of homolog sequences plus the
audit artifacts a reviewer needs: ``corpus_manifest.tsv``,
``corpus_provenance.json``, ``sequence_identity.tsv``, ``leakage_audit.json``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import identity as identity_mod
from . import provenance as prov

_MANIFEST_FIELDS = (
    "basename",
    "accession",
    "source",
    "length",
    "matched_accession",
    "identity_to_query",
    "coverage",
)


@dataclass
class CorpusRecord:
    basename: str
    accession: str
    source: str
    sequence: str
    matched_accession: str = ""
    identity_to_query: Optional[float] = None
    coverage: Optional[float] = None

    @property
    def length(self) -> int:
        return len(self.sequence)


def _read_fasta_sequence(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return prov.normalize_sequence(
        "".join(line for line in lines if line and not line.startswith(">"))
    )


def load_single_fasta_dir(
    single_dir: Path,
    *,
    manifest_path: Optional[Path] = None,
    include_query_source: bool = False,
) -> List[CorpusRecord]:
    """Load records from a profile-run ``single_fasta/`` directory.

    If a manifest TSV is given it drives ordering and metadata; otherwise all
    ``*.fasta`` files are read in sorted order. ``source == "query"`` rows are
    dropped unless ``include_query_source`` is True (strict corpora exclude the
    query).
    """

    single_dir = Path(single_dir)
    records: List[CorpusRecord] = []
    if manifest_path is not None and Path(manifest_path).exists():
        with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source = str(row.get("source") or "").strip()
                if source == "query" and not include_query_source:
                    continue
                basename = str(row.get("basename") or "").strip()
                fasta = single_dir / f"{basename}.fasta"
                if not fasta.exists():
                    continue
                records.append(
                    CorpusRecord(
                        basename=basename,
                        accession=str(row.get("accession") or "").strip(),
                        source=source or "download",
                        sequence=_read_fasta_sequence(fasta),
                        matched_accession=str(row.get("matched_accession") or "").strip(),
                    )
                )
    else:
        for fasta in sorted(single_dir.glob("*.fasta")):
            records.append(
                CorpusRecord(
                    basename=fasta.stem,
                    accession=fasta.stem,
                    source="download",
                    sequence=_read_fasta_sequence(fasta),
                )
            )
    return records


def exclude_exact_query(records: Sequence[CorpusRecord], query_seq: str) -> List[CorpusRecord]:
    """Remove any record byte-identical to the query (strict leakage control)."""

    q = prov.normalize_sequence(query_seq)
    return [r for r in records if prov.normalize_sequence(r.sequence) != q]


def exclude_assayed_mutants(
    records: Sequence[CorpusRecord], assayed_mutants: Mapping[str, str]
) -> Tuple[List[CorpusRecord], List[str]]:
    """Remove records byte-identical to any assayed mutant (leakage control).

    Returns ``(kept_records, removed_mutant_ids)``. This is applied to BOTH the
    strict and with-query corpora: even in the transductive (with-query)
    condition, training on a labeled-mutant sequence would leak the assay. The
    query/WT itself is not an assayed mutant and is unaffected.
    """

    mut_seq_to_id: Dict[str, str] = {
        prov.normalize_sequence(s): mid for mid, s in assayed_mutants.items()
    }
    kept: List[CorpusRecord] = []
    removed: List[str] = []
    for rec in records:
        mid = mut_seq_to_id.get(prov.normalize_sequence(rec.sequence))
        if mid is not None:
            removed.append(mid)
        else:
            kept.append(rec)
    return kept, sorted(set(removed))


def annotate_identity(
    records: Sequence[CorpusRecord],
    query_seq: str,
    *,
    backend: Optional[identity_mod.IdentityBackend] = None,
) -> identity_mod.IdentityBackend:
    """Fill ``identity_to_query`` and ``coverage`` on each record in place."""

    b = backend or identity_mod.detect_backend(len(records))
    q = prov.normalize_sequence(query_seq)
    seqs = [r.sequence for r in records]
    idents = identity_mod.identity_to_query(seqs, q, backend=b)
    qlen = max(len(q), 1)
    for rec, ident in zip(records, idents):
        rec.identity_to_query = float(ident)
        rec.coverage = float(min(len(rec.sequence), qlen) / qlen)
    return b


def total_tokens(records: Sequence[CorpusRecord]) -> int:
    return sum(r.length for r in records)


@dataclass
class FrozenCorpus:
    name: str
    records: List[CorpusRecord]
    query_seq: str
    wt_seq: str
    identity_backend: str
    corpus_hash: str
    leakage: prov.LeakageAudit
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_sequences(self) -> int:
        return len(self.records)

    @property
    def total_tokens(self) -> int:
        return total_tokens(self.records)


def freeze_corpus(
    name: str,
    records: Sequence[CorpusRecord],
    *,
    query_seq: str,
    wt_seq: str,
    assayed_mutants: Optional[Mapping[str, str]] = None,
    identity_backend: Optional[identity_mod.IdentityBackend] = None,
    source_meta: Optional[Mapping[str, object]] = None,
) -> FrozenCorpus:
    """Compute hash + leakage audit and return an immutable descriptor."""

    backend = identity_backend or identity_mod.detect_backend(len(records))
    corpus_hash = prov.corpus_sha256(r.sequence for r in records)
    leakage = prov.audit_leakage(
        [r.sequence for r in records],
        query_seq=query_seq,
        wt_seq=wt_seq,
        assayed_mutants=assayed_mutants,
        corpus_accessions=[r.accession for r in records],
    )
    return FrozenCorpus(
        name=name,
        records=list(records),
        query_seq=query_seq,
        wt_seq=wt_seq,
        identity_backend=backend.label(),
        corpus_hash=corpus_hash,
        leakage=leakage,
        meta=dict(source_meta or {}),
    )


def write_corpus_audit(
    corpus: FrozenCorpus,
    out_dir: Path,
    *,
    sequence_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Write manifest, provenance, leakage, and a trainer manifest without
    copying per-sequence FASTAs.

    Mined corpora have tens of thousands of FASTAs already on disk; copying them
    per variant is wasteful. This writer records the frozen membership, hash, and
    leakage audit, and emits a ``train_manifest.tsv`` (RawSequenceDataset schema:
    ``seq_id,sequence_path,source,length,matched_accession``) that points
    the trainer at ``sequence_root`` (the existing single_fasta dir).
    """

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"corpus audit dir is not empty (immutable): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "corpus_manifest.tsv"
    identity_path = out_dir / "sequence_identity.tsv"
    train_manifest = out_dir / "train_manifest.tsv"
    with (
        manifest_path.open("w", encoding="utf-8", newline="") as mh,
        identity_path.open("w", encoding="utf-8", newline="") as ih,
        train_manifest.open("w", encoding="utf-8", newline="") as th,
    ):
        mw = csv.DictWriter(mh, fieldnames=list(_MANIFEST_FIELDS), delimiter="\t")
        mw.writeheader()
        iw = csv.DictWriter(
            ih,
            fieldnames=["basename", "accession", "identity_to_query", "coverage", "length"],
            delimiter="\t",
        )
        iw.writeheader()
        tw = csv.DictWriter(
            th,
            fieldnames=["seq_id", "sequence_path", "source", "length", "matched_accession"],
            delimiter="\t",
        )
        tw.writeheader()
        for rec in corpus.records:
            mw.writerow(
                {
                    "basename": rec.basename,
                    "accession": rec.accession,
                    "source": rec.source,
                    "length": rec.length,
                    "matched_accession": rec.matched_accession,
                    "identity_to_query": ""
                    if rec.identity_to_query is None
                    else f"{rec.identity_to_query:.6f}",
                    "coverage": "" if rec.coverage is None else f"{rec.coverage:.6f}",
                }
            )
            iw.writerow(
                {
                    "basename": rec.basename,
                    "accession": rec.accession,
                    "identity_to_query": ""
                    if rec.identity_to_query is None
                    else f"{rec.identity_to_query:.6f}",
                    "coverage": "" if rec.coverage is None else f"{rec.coverage:.6f}",
                    "length": rec.length,
                }
            )
            tw.writerow(
                {
                    "seq_id": rec.basename,
                    "sequence_path": f"{rec.basename}.fasta",
                    "source": rec.source,
                    "length": rec.length,
                    "matched_accession": rec.matched_accession,
                }
            )

    idents = [r.identity_to_query for r in corpus.records if r.identity_to_query is not None]
    prov.write_json(
        out_dir / "corpus_provenance.json",
        {
            "name": corpus.name,
            "corpus_sha256": corpus.corpus_hash,
            "n_sequences": corpus.n_sequences,
            "total_aa_tokens": corpus.total_tokens,
            "identity_backend": corpus.identity_backend,
            "query_sha256": prov.sequence_sha256(corpus.query_seq),
            "wt_sha256": prov.sequence_sha256(corpus.wt_seq),
            "manifest_sha256": prov.sha256_file(manifest_path),
            "sequence_root": str(sequence_root) if sequence_root else "",
            "identity_summary": _identity_summary(idents),
            "length_summary": _length_summary([r.length for r in corpus.records]),
            "source_meta": corpus.meta,
        },
    )
    prov.write_json(out_dir / "leakage_audit.json", corpus.leakage.to_dict())
    return {
        "corpus_manifest": str(manifest_path),
        "sequence_identity": str(identity_path),
        "train_manifest": str(train_manifest),
        "corpus_provenance": str(out_dir / "corpus_provenance.json"),
        "leakage_audit": str(out_dir / "leakage_audit.json"),
    }


def build_assayed_mutants(
    wt_seq: str, variants: Sequence[Tuple[int, str, str]], *, one_indexed: bool = True
) -> Dict[str, str]:
    """Construct full mutant sequences for leakage auditing from (pos, wt, mut)."""

    wt = prov.normalize_sequence(wt_seq)
    out: Dict[str, str] = {}
    for pos, wt_aa, mut_aa in variants:
        idx = int(pos) - 1 if one_indexed else int(pos)
        if idx < 0 or idx >= len(wt):
            continue
        mut_id = f"{wt_aa}{pos}{mut_aa}"
        out[mut_id] = wt[:idx] + str(mut_aa).upper() + wt[idx + 1 :]
    return out


def _identity_summary(idents: Sequence[float]) -> Dict[str, float]:
    if not idents:
        return {}
    import numpy as np

    arr = np.asarray(idents, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def _length_summary(lengths: Sequence[int]) -> Dict[str, float]:
    if not lengths:
        return {}
    import numpy as np

    arr = np.asarray(lengths, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
