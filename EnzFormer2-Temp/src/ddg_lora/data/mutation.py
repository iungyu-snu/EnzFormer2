"""Single-point mutation parsing and validation.

Mutation strings come in flavors like "A123V", "p.Ala123Val", or per-column
(wt_aa, position, mut_aa). This module normalizes them into a single dataclass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ddg_lora.constants import AA_SET

# Common 3-letter -> 1-letter map for "p.Ala123Val" style strings.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_ONE_LETTER_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_THREE_LETTER_RE = re.compile(
    r"^p?\.?([A-Za-z]{3})(\d+)([A-Za-z]{3})$"
)


@dataclass(frozen=True)
class Mutation:
    """A single-point mutation.

    Attributes:
        wt: wild-type amino acid (1-letter).
        position: 1-indexed sequence position.
        mut: mutant amino acid (1-letter).
    """
    wt: str
    position: int
    mut: str

    def __post_init__(self) -> None:
        if self.wt not in AA_SET:
            raise ValueError(f"unknown wild-type amino acid: {self.wt!r}")
        if self.mut not in AA_SET:
            raise ValueError(f"unknown mutant amino acid: {self.mut!r}")
        if self.position < 1:
            raise ValueError(f"position must be 1-indexed and >= 1, got {self.position}")
        if self.wt == self.mut:
            raise ValueError(
                f"wt and mut are identical at position {self.position}: not a mutation"
            )

    def to_token(self) -> str:
        """Render as canonical 1-letter token, e.g. 'A123V'."""
        return f"{self.wt}{self.position}{self.mut}"

    def index0(self) -> int:
        """Return 0-indexed sequence position."""
        return self.position - 1


def parse_mutation(s: str) -> Mutation:
    """Parse a mutation string in '<WT><pos><MUT>' or 'p.<WT3><pos><MUT3>' form.

    Raises ValueError on any malformed input. Position must be 1-indexed.
    """
    if not isinstance(s, str):
        raise TypeError(f"mutation must be a string, got {type(s).__name__}")
    s = s.strip()
    if not s:
        raise ValueError("empty mutation string")

    m1 = _ONE_LETTER_RE.match(s.upper())
    if m1:
        wt, pos, mut = m1.group(1), int(m1.group(2)), m1.group(3)
        return Mutation(wt=wt, position=pos, mut=mut)

    m3 = _THREE_LETTER_RE.match(s)
    if m3:
        wt3, pos, mut3 = m3.group(1).upper(), int(m3.group(2)), m3.group(3).upper()
        if wt3 not in _THREE_TO_ONE or mut3 not in _THREE_TO_ONE:
            raise ValueError(f"unknown 3-letter amino acid in {s!r}")
        return Mutation(
            wt=_THREE_TO_ONE[wt3], position=pos, mut=_THREE_TO_ONE[mut3]
        )

    raise ValueError(f"could not parse mutation string: {s!r}")


def validate_against_sequence(seq: str, mut: Mutation) -> None:
    """Confirm that `seq[mut.index0()] == mut.wt`. Raises ValueError if not.

    Use this on the wild-type protein sequence before constructing inputs.
    """
    idx = mut.index0()
    if idx >= len(seq):
        raise ValueError(
            f"mutation position {mut.position} out of range for sequence of length {len(seq)}"
        )
    if seq[idx] != mut.wt:
        raise ValueError(
            f"sequence has {seq[idx]!r} at position {mut.position}, "
            f"but mutation expects wild-type {mut.wt!r}"
        )
