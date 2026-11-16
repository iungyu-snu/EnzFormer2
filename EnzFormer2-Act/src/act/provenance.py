"""Provenance hashing and leakage auditing for frozen corpora.

Deterministic, dependency-free. Hashing is canonicalized so the same set of
sequences always yields the same corpus hash regardless of file ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sequence(seq: str) -> str:
    """Uppercase, strip whitespace. Used before any hashing/comparison."""

    return "".join(str(seq or "").split()).upper()


def sequence_sha256(seq: str) -> str:
    return sha256_text(normalize_sequence(seq))


def corpus_sha256(sequences: Iterable[str]) -> str:
    """Order-independent hash of a set of sequences.

    Sequences are normalized, per-sequence hashed, sorted, then the sorted
    digest list is hashed. This makes the corpus hash invariant to file/record
    ordering but sensitive to any added/removed/edited sequence.
    """

    digests = sorted(sequence_sha256(s) for s in sequences)
    joined = "\n".join(digests)
    return sha256_text(f"n={len(digests)}\n{joined}")


@dataclass
class LeakageAudit:
    """Result of checking a corpus against assayed/query sequences."""

    n_corpus_sequences: int
    total_aa_tokens: int
    query_sha256: str
    wt_sha256: str
    query_in_corpus: bool
    wt_in_corpus: bool
    leaked_mutant_ids: List[str] = field(default_factory=list)
    exact_duplicate_sequences: int = 0
    duplicate_accessions: List[str] = field(default_factory=list)
    n_assayed_mutants_checked: int = 0

    @property
    def has_leakage(self) -> bool:
        # WT presence is expected/allowed in a homolog corpus; leakage means an
        # assayed *mutant* sequence (a label-bearing point) entered training.
        return bool(self.leaked_mutant_ids)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_corpus_sequences": self.n_corpus_sequences,
            "total_aa_tokens": self.total_aa_tokens,
            "query_sha256": self.query_sha256,
            "wt_sha256": self.wt_sha256,
            "query_in_corpus": self.query_in_corpus,
            "wt_in_corpus": self.wt_in_corpus,
            "n_assayed_mutants_checked": self.n_assayed_mutants_checked,
            "leaked_mutant_ids": sorted(self.leaked_mutant_ids),
            "exact_duplicate_sequences": self.exact_duplicate_sequences,
            "duplicate_accessions": sorted(self.duplicate_accessions),
            "has_leakage": self.has_leakage,
        }


def audit_leakage(
    corpus_sequences: Sequence[str],
    *,
    query_seq: str,
    wt_seq: str,
    assayed_mutants: Mapping[str, str] | None = None,
    corpus_accessions: Sequence[str] | None = None,
) -> LeakageAudit:
    """Audit a corpus for leakage of query/WT/assayed-mutant sequences.

    ``assayed_mutants`` maps a mutant id (e.g. ``"A34G"``) to its full mutant
    sequence. Any assayed-mutant sequence found in the corpus is a hard failure.
    """

    norm_corpus = [normalize_sequence(s) for s in corpus_sequences]
    corpus_set = set(norm_corpus)
    total_tokens = sum(len(s) for s in norm_corpus)

    query_norm = normalize_sequence(query_seq)
    wt_norm = normalize_sequence(wt_seq)

    leaked: List[str] = []
    n_checked = 0
    if assayed_mutants:
        for mut_id, mut_seq in assayed_mutants.items():
            n_checked += 1
            if normalize_sequence(mut_seq) in corpus_set:
                leaked.append(str(mut_id))

    dup_accessions: List[str] = []
    if corpus_accessions is not None:
        seen: set[str] = set()
        for acc in corpus_accessions:
            acc = str(acc or "").strip()
            if not acc:
                continue
            if acc in seen:
                dup_accessions.append(acc)
            seen.add(acc)

    return LeakageAudit(
        n_corpus_sequences=len(norm_corpus),
        total_aa_tokens=total_tokens,
        query_sha256=sha256_text(query_norm),
        wt_sha256=sha256_text(wt_norm),
        query_in_corpus=query_norm in corpus_set,
        wt_in_corpus=wt_norm in corpus_set,
        leaked_mutant_ids=leaked,
        exact_duplicate_sequences=len(norm_corpus) - len(corpus_set),
        duplicate_accessions=dup_accessions,
        n_assayed_mutants_checked=n_checked,
    )


def assert_no_leakage(audit: LeakageAudit) -> None:
    """Raise if any assayed mutant sequence leaked into the corpus."""

    if audit.has_leakage:
        raise ValueError(
            "Leakage detected: assayed mutant sequences present in unsupervised "
            f"training corpus: {sorted(audit.leaked_mutant_ids)}"
        )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
