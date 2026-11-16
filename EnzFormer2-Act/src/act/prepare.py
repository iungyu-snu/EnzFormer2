"""Prepare an audited qwoff corpus and training configuration from profile output."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from enzformer2.config.loader import config_to_serializable_dict
from enzformer2.config.schemas import ActTrainConfig
from enzformer2.data.raw_sequence_dataset import RawSequenceDataset
from enzformer2.profile.fasta_utils import parse_fasta_records, sanitize_seq
from enzformer2.profile.offline import validate_frozen_corpus


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="enzformer2 act prepare", description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--query-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    if args.max_sequences is not None and args.max_sequences < 2:
        raise ValueError("--max-sequences must be at least 2.")
    source = args.corpus_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    targets = [output / name for name in ("train.tsv", "train.yaml", "corpus_audit.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Prepared inputs already exist; use a new --output-dir.")
    corpus = validate_frozen_corpus(source / "manifest.tsv")
    query_records = parse_fasta_records(args.query_fasta.expanduser().resolve())
    if len(query_records) != 1:
        raise ValueError("--query-fasta must contain exactly one sequence.")
    query = sanitize_seq(query_records[0][1])
    alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    if not set(query) <= alphabet:
        raise ValueError("Query must contain only canonical amino acids.")
    dataset = RawSequenceDataset(source / "manifest.tsv", sequence_root=source)
    seen, eligible, excluded = set(), [], []
    for record in dataset.records:
        sequence = record["sequence"]
        reason = None
        if not set(sequence) <= alphabet:
            reason = "noncanonical_residue"
        elif sequence in seen:
            reason = "duplicate_sequence"
        elif len(sequence) == len(query) and sum(a != b for a, b in zip(sequence, query)) <= 1:
            reason = "query_or_single_substitution"
        if reason:
            excluded.append({"seq_id": record["seq_id"], "reason": reason})
            continue
        seen.add(sequence)
        eligible.append({"seq_id": record["seq_id"], "sequence": sequence})
    selected = sorted(eligible, key=lambda row: row["seq_id"])[: args.max_sequences]
    if len(selected) < 2:
        raise ValueError("At least two eligible sequences must remain after filtering.")
    config = ActTrainConfig()
    config.data.manifest_path = targets[0]
    config.data.sequence_column = "sequence"
    config.data.seq_id_column = "seq_id"
    config.runtime.epochs = args.epochs
    config.runtime.batch_size = args.batch_size
    config.runtime.amp_dtype = args.precision
    config.runtime.validation_fraction = args.validation_fraction
    config.runtime.seed = args.seed
    config.runtime.backbone_max_masked_rows_per_call = 1
    config.runtime.__post_init__()
    config.checkpoint.output_dir = output
    serialized = config_to_serializable_dict(config)
    serialized["data"]["manifest_path"] = "train.tsv"
    serialized["checkpoint"]["output_dir"] = "."
    audit = {
        "source_corpus": corpus,
        "query_sequence": query,
        "excluded": excluded,
        "eligible": len(eligible),
        "selected": [row["seq_id"] for row in selected],
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(targets[0], sep="\t", index=False)
    targets[1].write_text(yaml.safe_dump(serialized, sort_keys=False), encoding="utf-8")
    targets[2].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"selected": len(selected), "excluded": len(excluded), "config": str(targets[1])},
            indent=2,
        )
    )
