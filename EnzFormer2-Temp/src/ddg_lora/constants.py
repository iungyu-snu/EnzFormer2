"""Project-wide constants. Source-of-truth for amino-acid alphabet and label codes."""
from __future__ import annotations

AMINO_ACIDS_20: tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
)

AA_SET = frozenset(AMINO_ACIDS_20)

# Binary label codes used after neutral exclusion.
LABEL_STABILIZING = 1
LABEL_DESTABILIZING = 0

# Default ΔΔG thresholds in kcal/mol (post sign-normalization where
# positive means destabilizing). Configs may override.
DEFAULT_STABILIZING_MAX = -0.5
DEFAULT_DESTABILIZING_MIN = 0.5
