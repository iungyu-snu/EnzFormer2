"""Verify the paper ensemble and archived validation threshold without a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddg_lora.ensemble import FINAL_SEEDS, bundle_members, file_hash  # noqa: E402


def verify(checkpoint_dir: Path) -> dict:
    if FINAL_SEEDS != (42, 43, 44):
        raise ValueError("The runtime must select paper seeds 42, 43 and 44")
    members = bundle_members(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "ensemble.json").read_text(encoding="utf-8"))
    if len(members) != 3:
        raise ValueError("The paper release requires three ensemble members")
    default = yaml.safe_load((ROOT / "configs/paper/enzformer2_temp.yaml").read_text(encoding="utf-8"))
    first = yaml.safe_load((members[0] / "train_config.yaml").read_text(encoding="utf-8"))
    if default != first:
        raise ValueError("The default paper configuration must match the archived seed42 configuration")
    threshold = json.loads((checkpoint_dir / "threshold.json").read_text(encoding="utf-8"))
    if (threshold.get("fit_split") != "validation" or threshold.get("metric_name") != "mcc"
            or threshold.get("threshold_score_neg_ddg_hat") != -0.18037773685253233):
        raise ValueError("The archived paper validation threshold is inconsistent")
    binding = threshold.get("checkpoint", {})
    if (binding.get("aggregation") != "mean_decoded_ddg"
            or binding.get("model_name") != manifest["model_name"]
            or binding.get("inference_mode") != "ensemble_mean"):
        raise ValueError("Threshold aggregation/model identity disagrees with the ensemble")
    expected = [(m["seed"], m["sha256"]) for m in manifest["members"]]
    observed = [(m.get("seed"), m.get("sha256")) for m in binding.get("members", [])]
    if observed != expected:
        raise ValueError("Threshold checkpoint hashes disagree with the released members")
    return {
        "passed": True,
        "model_name": manifest["model_name"],
        "seeds": list(FINAL_SEEDS),
        "aggregation": manifest["aggregation"],
        "member_files_hash_verified": sum(len(m["sha256"]) for m in manifest["members"]),
        "ensemble_manifest_sha256": file_hash(checkpoint_dir / "ensemble.json"),
        "threshold_sha256": file_hash(checkpoint_dir / "threshold.json"),
        "threshold_score_neg_ddg_hat": threshold["threshold_score_neg_ddg_hat"],
        "threshold_member_hashes_match": True,
        "default_config_matches_seed42": True,
        "scope": "Release identity and integrity; no GPU inference or benchmark rerun.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "checkpoints/enzformer2_temp")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.checkpoint_dir)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
