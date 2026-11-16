"""Filesystem + YAML helpers used by scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def dump_json(data: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def ensure_run_dir(out_dir: str | Path, exist_ok: bool = False) -> Path:
    """Create the run directory. By default refuses to overwrite an existing run."""
    out = Path(out_dir)
    if out.exists() and not exist_ok:
        raise FileExistsError(
            f"Run directory already exists: {out}. "
            "Pick a new run name or pass exist_ok=True explicitly."
        )
    out.mkdir(parents=True, exist_ok=exist_ok)
    return out
