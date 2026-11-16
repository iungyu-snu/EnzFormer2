"""Installed EnzFormer2-Temp commands with lazy scientific dispatch."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

import yaml


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="EnzFormer2-Temp workflows")
    parser.add_argument("command", choices=("train", "score", "evaluate"))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    if command != "train":
        module = {"score": "ddg_lora.scoring", "evaluate": "ddg_lora.evaluation.cli"}[command]
        return importlib.import_module(module).main(argv[1:])
    train = argparse.ArgumentParser(description="Train EnzFormer2-Temp from a YAML configuration")
    train.add_argument("--config", required=True)
    train.add_argument("--validate-config", action="store_true",
                       help="Check schema and local input availability without loading a model")
    args = train.parse_args(argv[1:])
    from ddg_lora.config_validation import validate_config

    if args.validate_config:
        try:
            report = validate_config(args.config)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
        print(json.dumps(report, indent=2))
        return 0
    from ddg_lora.training.regression_trainer import run_from_config
    from ddg_lora.utils.io import load_yaml

    cfg = load_yaml(args.config)
    if "test_path" in cfg.get("data", {}):
        print("error: data.test_path must not appear in a training configuration", file=sys.stderr)
        return 4
    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
