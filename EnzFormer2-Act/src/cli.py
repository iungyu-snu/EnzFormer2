"""Public command line entry point for EnzFormer2-Act."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzformer2",
        description="EnzFormer2 profile, activity-scoring, and benchmark workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser(
        "profile",
        help="Generate or validate an enzyme family profile and retrieve sequences.",
    )
    profile.set_defaults(handler="profile")

    act = subparsers.add_parser(
        "act",
        help="EnzFormer2-Act family-specialized activity workflows.",
    )
    act_subparsers = act.add_subparsers(dest="act_command", required=True)
    for name, help_text in (
        ("prepare", "Prepare a profile-derived corpus and LoRA training configuration."),
        ("train", "Train a LoRA adapter from a prepared corpus."),
        ("evaluate", "Score a checkpoint and evaluate against measured activity."),
        ("score", "Score all canonical single substitutions with a checkpoint."),
        ("benchmark", "Run a point-mutation benchmark manifest."),
        ("recommend", "Rank candidate substitutions for a query sequence."),
    ):
        child = act_subparsers.add_parser(name, help=help_text)
        child.set_defaults(handler=f"act_{name}")
    return parser


def _dispatch(handler: str, argv: list[str]) -> None:
    if handler == "act_prepare":
        from enzformer2.act.prepare import main

        main(argv)
        return

    if handler == "act_evaluate":
        from benchmark.shared.point_mutation.act_evaluator import main

        main(argv)
        return

    if handler == "profile":
        if "--validate-only" in argv:
            from enzformer2.profile.offline import main
        else:
            from enzformer2.profile.cli import main

        main(argv)
        return

    if handler == "act_train":
        from enzformer2.act.training import main

        main(argv)
        return

    if handler == "act_score":
        from enzformer2.act.scoring import main

        main(argv)
        return

    if handler == "act_benchmark":
        from benchmark.shared.point_mutation.runner import main

        main(argv)
        return

    if handler == "act_recommend":
        from enzformer2.act.recommend_mutants import main

        main(argv)
        return

    raise ValueError(f"Unknown EnzFormer2 command: {handler}")


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(argv) if argv is not None else sys.argv[1:]
    try:
        if len(raw) >= 2 and any(arg in {"-h", "--help"} for arg in raw[1:]):
            if raw[0] == "profile":
                _dispatch("profile", [*raw[1:]])
                return
            if raw[0] == "act" and len(raw) >= 3:
                _dispatch(f"act_{raw[1]}", [*raw[2:]])
                return

        parser = _build_parser()
        parsed, remaining = parser.parse_known_args(raw)
        _dispatch(str(parsed.handler), list(remaining))
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":  # pragma: no cover
    main()
