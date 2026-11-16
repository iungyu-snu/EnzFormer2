"""Rank prospective Act substitutions for one query sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from enzformer2.act.scoring import load_act_model, scan_single_mutants
from enzformer2.act.training_common import resolve_device
from enzformer2.profile import fasta_utils


def _sequence_from_inputs(sequence: str | None, fasta: Path | None) -> tuple[str, str]:
    if (sequence is None) == (fasta is None):
        raise ValueError("Provide exactly one of --sequence and --fasta.")
    if fasta is not None:
        header, value = fasta_utils.read_first_fasta(fasta.expanduser().resolve())
        return header or fasta.stem, value.strip().replace("\n", "").upper()
    assert sequence is not None
    return "query", sequence.strip().replace("\n", "").upper()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="enzformer2 act recommend",
        description="Rank candidate single substitutions with an Act checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--seq-id", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    seq_id, sequence = _sequence_from_inputs(args.sequence, args.fasta)
    device = resolve_device(args.device)
    model, _ = load_act_model(args.checkpoint, device=device)
    scores = scan_single_mutants(model, args.seq_id or seq_id, sequence, chunk_size=args.chunk_size)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.head(args.top_k).to_csv(output, index=False)
    print(f"Wrote {min(args.top_k, len(scores))} recommendations to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
