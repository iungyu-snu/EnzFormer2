"""Sequence-identity estimation for profile-guided corpus filtering."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


EXACT_MAX_CORPUS = 2000
_KMER_K = 5


def resolve_mmseqs_bin() -> Optional[str]:
    """Return a runnable MMseqs2 binary path, or ``None``."""

    env = os.environ.get("ENZFORMER2_MMSEQS_BIN")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    return shutil.which("mmseqs")


@dataclass(frozen=True)
class IdentityBackend:
    name: str
    version: str

    def label(self) -> str:
        return f"{self.name}/{self.version}"


def detect_backend(n_sequences: int, prefer: str = "auto") -> IdentityBackend:
    """Select an identity backend deterministically."""

    mmseqs = resolve_mmseqs_bin()
    if prefer == "mmseqs":
        if mmseqs:
            return IdentityBackend("mmseqs2", _mmseqs_version(mmseqs))
        return IdentityBackend("python_global", "v1")
    if prefer == "exact":
        return IdentityBackend("python_global", "v1")
    if prefer == "kmer":
        return IdentityBackend("python_kmer", "v1")
    if mmseqs:
        return IdentityBackend("mmseqs2", _mmseqs_version(mmseqs))
    if n_sequences <= EXACT_MAX_CORPUS:
        return IdentityBackend("python_global", "v1")
    return IdentityBackend("python_kmer", "v1")


def _mmseqs_version(mmseqs_bin: Optional[str] = None) -> str:
    mmseqs_bin = mmseqs_bin or resolve_mmseqs_bin()
    if not mmseqs_bin:
        return "unknown"
    try:  # pragma: no cover - only when MMseqs2 is installed
        out = subprocess.run([mmseqs_bin, "version"], capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr or "unknown").strip() or "unknown"
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
    ):  # pragma: no cover
        return "unknown"


def global_alignment_identity(
    a: str, b: str, *, match: int = 1, mismatch: int = -1, gap: int = -1
) -> float:
    """Return exact Needleman-Wunsch identity in ``[0, 1]``."""

    a = a.upper()
    b = b.upper()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(0, n + 1) * gap
    score[0, :] = np.arange(0, m + 1) * gap

    a_arr = np.frombuffer(a.encode("ascii"), dtype=np.uint8)
    b_arr = np.frombuffer(b.encode("ascii"), dtype=np.uint8)
    for i in range(1, n + 1):
        ai = a_arr[i - 1]
        row_prev = score[i - 1]
        row_cur = score[i]
        for j in range(1, m + 1):
            diag = row_prev[j - 1] + (match if ai == b_arr[j - 1] else mismatch)
            up = row_prev[j] + gap
            left = row_cur[j - 1] + gap
            row_cur[j] = diag if diag >= up and diag >= left else (up if up >= left else left)

    i, j = n, m
    ident = 0
    length = 0
    while i > 0 and j > 0:
        current = score[i, j]
        diagonal = score[i - 1, j - 1] + (match if a_arr[i - 1] == b_arr[j - 1] else mismatch)
        if current == diagonal:
            length += 1
            if a_arr[i - 1] == b_arr[j - 1]:
                ident += 1
            i -= 1
            j -= 1
        elif current == score[i - 1, j] + gap:
            length += 1
            i -= 1
        else:
            length += 1
            j -= 1
    length += i + j
    return ident / length if length else 0.0


def _kmer_set(seq: str, k: int = _KMER_K) -> set[str]:
    seq = seq.upper()
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def kmer_identity_estimate(a: str, b: str, k: int = _KMER_K) -> float:
    """Return the deterministic shared-k-mer containment estimate."""

    ka, kb = _kmer_set(a, k), _kmer_set(b, k)
    if not ka or not kb:
        return 1.0 if a.upper() == b.upper() else 0.0
    return len(ka & kb) / min(len(ka), len(kb))


def mmseqs_identity_to_query(
    sequences: Sequence[str],
    query: str,
    *,
    mmseqs_bin: Optional[str] = None,
    sensitivity: float = 7.5,
    threads: int = 4,
) -> List[float]:
    """Return MMseqs2 ``fident`` values for each sequence against ``query``."""

    mmseqs_bin = mmseqs_bin or resolve_mmseqs_bin()
    if not mmseqs_bin:
        raise RuntimeError("mmseqs binary not found; set ENZFORMER2_MMSEQS_BIN")

    idents = [0.0] * len(sequences)
    with tempfile.TemporaryDirectory(prefix="enzformer2_mmseqs_") as td:
        root = Path(td)
        query_fasta = root / "corpus.fasta"
        target_fasta = root / "query.fasta"
        result = root / "result.tsv"
        temp_root = root / "tmp"
        with query_fasta.open("w", encoding="utf-8") as handle:
            for index, sequence in enumerate(sequences):
                handle.write(f">{index}\n{sequence.upper()}\n")
        target_fasta.write_text(f">query\n{query.upper()}\n", encoding="utf-8")
        command = [
            mmseqs_bin,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(result),
            str(temp_root),
            "--format-output",
            "query,fident,qcov",
            "-s",
            str(sensitivity),
            "--max-seqs",
            "5",
            "--threads",
            str(threads),
            "-v",
            "1",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=3600)
        if result.exists():
            best: Dict[int, float] = {}
            for line in result.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    index = int(parts[0])
                    fident = float(parts[1])
                except ValueError:
                    continue
                if fident > best.get(index, -1.0):
                    best[index] = fident
            for index, value in best.items():
                if 0 <= index < len(idents):
                    idents[index] = value
    return idents


def identity_to_query(
    sequences: Sequence[str],
    query: str,
    *,
    backend: IdentityBackend | None = None,
) -> List[float]:
    """Return per-sequence identity using the selected or automatic backend."""

    selected = backend or detect_backend(len(sequences))
    if selected.name == "mmseqs2":
        try:
            return mmseqs_identity_to_query(sequences, query)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, KeyError, TypeError):
            return [global_alignment_identity(sequence, query) for sequence in sequences]
    if selected.name == "python_global":
        return [global_alignment_identity(sequence, query) for sequence in sequences]
    return [kmer_identity_estimate(sequence, query) for sequence in sequences]
