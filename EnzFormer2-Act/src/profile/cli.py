"""CLI for profile-guided enzyme-family sequence collection.

Usage (via the public package facade)::

    python -m enzformer2.cli profile \\
        --enzyme-type "chitinase" \\
        --query-fasta /path/to/query.fasta \\
        --llm-provider openrouter \\
        --llm-model <model-id> \\
        --download-limit 500

This command builds an enzyme profile with an LLM when a frozen profile is not
provided, then mines UniProt/InterPro/Pfam for candidate accessions, downloads,
and filters
FASTAs, and writes a single-FASTA ``manifest.tsv`` plus the original query.
No training, scanning, labeling, or embedding is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from enzformer2.profile import fasta_utils
from enzformer2.profile.llm import (
    DEFAULT_QUERY_FASTA_LLM_MODEL,
    DEFAULT_QUERY_FASTA_LLM_PROVIDER,
    LLM_PROVIDER_CHOICES,
    check_openai_compatible_server,
    resolve_openai_like_runtime,
)
from enzformer2.profile.profile import (
    apply_minimal_conserved_motif_overrides,
    apply_user_active_site_overrides,
    generate_profile,
    normalize_profile,
    resolve_download_length_window,
    strip_reviewed_filter,
    write_mapping_failure_report,
)
from enzformer2.profile.runtime import ensure_dir, make_logger, run_logged, write_json


_log = make_logger("enzformer2.profile")


DEFAULT_OUTPUT_ROOT = Path("outputs/enzformer2_profile")
DEFAULT_MAPPING_THRESHOLD = 0.75
DOTENV_ALLOWED_PREFIXES = (
    "OPENROUTER_",
    "OPENAI_",
    "LLM_",
    "QUERY_FASTA_LLM_",
)
_DOTENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dotenv_candidate_paths() -> List[Path]:
    paths: List[Path] = []
    for path in (_repo_root() / ".env", Path.cwd() / ".env"):
        resolved = path.expanduser().resolve()
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _parse_dotenv_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not _DOTENV_KEY_RE.match(key):
        return None
    if not key.startswith(DOTENV_ALLOWED_PREFIXES):
        return None

    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if raw_value.strip().startswith('"'):
            value = value.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\"", '"')
    else:
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return key, value


def load_cli_dotenv(paths: Sequence[Path] | None = None) -> Dict[str, str]:
    """Load local LLM-related secrets from .env without exposing values.

    Values from later files override earlier files, and loaded values override
    the process environment. This keeps the optional live-profile route aligned
    with a repo-local `.env` without copying credentials into run artifacts.
    """

    loaded: Dict[str, str] = {}
    for path in list(paths) if paths is not None else _dotenv_candidate_paths():
        env_path = path.expanduser().resolve()
        if not env_path.exists() or not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_assignment(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ[key] = value
            loaded[key] = str(env_path)
    return loaded


def _validate_run_id(run_id: str) -> str:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("--run-id must not be empty")
    if rid in {".", ".."} or "/" in rid or "\\" in rid:
        raise ValueError(f"--run-id must be a single directory name, not a path (got {run_id!r})")
    if Path(rid).is_absolute():
        raise ValueError(f"--run-id must not be an absolute path: {run_id!r}")
    return rid


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzformer2 profile",
        description=(
            "Build a query-aware enzyme FASTA set via LLM-guided UniProt/InterPro mining. "
            "A frozen profile can be supplied to avoid live LLM generation."
        ),
    )
    parser.add_argument("--enzyme-type", required=True, help="Enzyme name / query phrase.")
    qgroup = parser.add_mutually_exclusive_group(required=True)
    qgroup.add_argument("--query-seq", default=None, help="Raw query amino-acid sequence.")
    qgroup.add_argument(
        "--query-fasta",
        type=Path,
        default=None,
        help="Path to a single-record query FASTA.",
    )
    parser.add_argument(
        "--query-details-file",
        type=Path,
        default=None,
        help=(
            "Optional text/markdown file with query-enzyme details used to derive "
            "query-specific minimal conserved active-site motifs."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output subdirectory name (default: UTC timestamp).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for run artifacts (default: outputs/enzformer2_profile).",
    )
    parser.add_argument(
        "--llm-provider",
        default=DEFAULT_QUERY_FASTA_LLM_PROVIDER,
        choices=list(LLM_PROVIDER_CHOICES),
        help="Provider for optional live profile generation; required without --frozen-profile.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_QUERY_FASTA_LLM_MODEL,
        help=f"Model id for live profile generation (default: {DEFAULT_QUERY_FASTA_LLM_MODEL}).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="LLM API base URL override.",
    )
    parser.add_argument(
        "--mapping-threshold",
        type=float,
        default=DEFAULT_MAPPING_THRESHOLD,
        help="Minimum profile confidence to proceed (default: 0.75).",
    )
    parser.add_argument("--download-limit", type=int, default=None)
    parser.add_argument(
        "--input-ids-file",
        type=Path,
        default=None,
        help="Frozen accession list; skips live Pfam/InterPro accession discovery.",
    )
    parser.add_argument("--download-min-len", type=int, default=None)
    parser.add_argument("--download-max-len", type=int, default=None)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--skip-filtering", action="store_true")
    parser.add_argument("--skip-active-site-filter", action="store_true")
    parser.add_argument(
        "--enable-active-site-filter",
        action="store_true",
        help=(
            "Apply query/profile motif and active-site rules without automatic admission checks."
        ),
    )
    parser.add_argument("--active-site-motif-regex", default=None)
    parser.add_argument("--active-site-rules-json", default=None)
    parser.add_argument(
        "--active-site-policy",
        default=None,
        choices=["all_required", "min_required"],
    )
    parser.add_argument("--active-site-min-hits", type=int, default=None)
    parser.add_argument(
        "--frozen-profile",
        type=Path,
        default=None,
        help="Use an existing normalized enzyme_profile.json without an LLM call.",
    )
    parser.add_argument(
        "--frozen-corpus-manifest",
        type=Path,
        default=None,
        help=(
            "Reuse a validated single-FASTA corpus manifest without contacting UniProt. "
            "Requires --frozen-profile."
        ),
    )
    parser.add_argument(
        "--frozen-retrieval-manifest",
        type=Path,
        default=None,
        help="Validate frozen retrieval metadata and its profile/corpus checksums.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing artifacts under --run-id when present.",
    )
    return parser


def _load_query(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(query_name, query_sequence)`` from either --query-seq or --query-fasta."""

    if args.query_fasta is not None:
        header, seq = fasta_utils.read_first_fasta(args.query_fasta.expanduser().resolve())
        return header or "QUERY_INPUT", seq
    seq = fasta_utils.sanitize_seq(str(args.query_seq or ""))
    return "QUERY_INPUT", seq


def _load_query_details_file(
    path: Optional[Path],
    input_dir: Path,
) -> tuple[str, Optional[str]]:
    if path is None:
        return "", None
    source_path = path.expanduser().resolve()
    text = source_path.read_text(encoding="utf-8")
    copied_path = input_dir / "query_details.txt"
    copied_path.write_text(text, encoding="utf-8")
    return text, str(source_path)


def _require_requests() -> None:
    try:
        import requests  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment check
        raise RuntimeError(
            "The 'requests' package is required for UniProt/InterPro downloads. "
            "Install it (e.g. `pip install requests`) and re-run."
        ) from exc


def _preflight_llm(
    *,
    provider: str,
    model: str,
    base_url: Optional[str],
) -> Dict[str, Any]:
    if not str(provider or "").strip() or not str(model or "").strip():
        raise ValueError(
            "Live profile generation requires --llm-provider and a non-empty model identifier."
        )
    runtime = resolve_openai_like_runtime(
        provider=provider, model=model, base_url=base_url, field_name="llm-provider"
    )
    info = check_openai_compatible_server(
        provider=runtime.provider,
        base_url=runtime.base_url,
        model=runtime.model,
        timeout_sec=10.0,
    )
    return {
        "provider": runtime.provider,
        "model": runtime.model,
        "base_url": runtime.base_url,
        "server_info": info,
    }


def _write_step_status(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def _build_downloader_cmd(
    *,
    profile_path: Path,
    pfam_id: str,
    interpro_id: str,
    outdir: Path,
    output_prefix: str,
    length_window: Dict[str, Any],
    args: argparse.Namespace,
    profile: Dict[str, Any],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "enzformer2.profile.downloader",
        "--profile-json",
        str(profile_path),
    ]
    if pfam_id:
        cmd.extend(["--pfam", pfam_id])
    else:
        cmd.append("--skip-pfam")
    if interpro_id:
        cmd.extend(["--interpro", interpro_id])
    else:
        cmd.append("--skip-interpro")
    cmd.extend(["--outdir", str(outdir)])
    cmd.extend(["--output-prefix", output_prefix])
    cmd.extend(["--preset", "none"])
    cmd.extend(["--min-len", str(int(length_window["min_len"]))])
    cmd.extend(["--max-len", str(int(length_window["max_len"]))])
    if args.input_ids_file is not None:
        cmd.extend(["--input-ids-file", str(args.input_ids_file.expanduser().resolve())])
    if args.download_limit is not None:
        cmd.extend(["--limit", str(int(args.download_limit))])
    if args.reviewed_only:
        cmd.append("--reviewed-only")
    if args.skip_filtering:
        cmd.append("--skip-filtering")
    if args.enable_active_site_filter:
        cmd.append("--enable-active-site-filter")
    if args.skip_active_site_filter:
        cmd.append("--skip-active-site-filter")
    extra_query = strip_reviewed_filter(
        str(profile["extra_uniprot_query"]) if profile.get("extra_uniprot_query") else None
    )
    if extra_query:
        cmd.extend(["--extra-query", extra_query])
    if profile.get("motif_regex") and (
        args.enable_active_site_filter or args.active_site_motif_regex is not None
    ):
        cmd.extend(["--motif-regex", str(profile["motif_regex"])])
    if profile.get("active_site_policy"):
        cmd.extend(["--active-site-policy", str(profile["active_site_policy"])])
    if profile.get("active_site_min_hits") is not None:
        cmd.extend(["--active-site-min-hits", str(int(profile["active_site_min_hits"]))])
    if profile.get("active_site_rules") and (
        args.enable_active_site_filter or args.active_site_rules_json is not None
    ):
        cmd.extend(["--active-site-rules-json", json.dumps(profile["active_site_rules"])])
    return cmd


def main(argv: Sequence[str] | None = None) -> None:
    load_cli_dotenv()

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_id = _validate_run_id(args.run_id or _utc_run_id())
    run_dir = args.output_root.expanduser().resolve() / run_id
    input_dir = run_dir / "input"
    llm_dir = run_dir / "llm"
    download_dir = run_dir / "download"
    single_dir = run_dir / "single_fasta"
    logs_dir = run_dir / "logs"
    for d in (input_dir, llm_dir, download_dir, single_dir, logs_dir):
        ensure_dir(d)

    step_status_path = logs_dir / "step_status.json"
    run_config_path = logs_dir / "run_config.json"
    status: Dict[str, Any] = {"run_id": run_id, "failed": False, "steps": {}}
    _write_step_status(step_status_path, status)

    try:
        # Runtime preflight.
        _require_requests()
        if args.frozen_corpus_manifest is not None and args.frozen_profile is None:
            raise ValueError("--frozen-corpus-manifest requires --frozen-profile.")
        if args.frozen_retrieval_manifest is not None and args.frozen_profile is None:
            raise ValueError("--frozen-retrieval-manifest requires --frozen-profile.")
        if args.frozen_profile is not None:
            provider = "frozen-profile"
            model = "none"
            base_url = None
            _log("using frozen profile; no LLM request will be made")
        else:
            llm_preflight = _preflight_llm(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
            )
            provider = llm_preflight["provider"]
            model = llm_preflight["model"]
            base_url = llm_preflight["base_url"]
            _log(f"llm preflight ok: provider={provider} model={model} base_url={base_url}")

        # Query bootstrap.
        query_name, query_seq = _load_query(args)
        query_fasta_path = input_dir / "query_input.fasta"
        fasta_utils.write_fasta(query_fasta_path, query_name, query_seq)
        query_details_text, query_details_source = _load_query_details_file(
            args.query_details_file,
            input_dir,
        )
        status["steps"]["bootstrap"] = {
            "query_length": len(query_seq),
            "query_fasta": str(query_fasta_path),
            "query_name": query_name,
            "query_details_file": query_details_source,
            "query_details_chars": len(query_details_text),
        }
        _write_step_status(step_status_path, status)

        write_json(
            run_config_path,
            {
                "run_id": run_id,
                "enzyme_type": args.enzyme_type,
                "query_name": query_name,
                "query_length": len(query_seq),
                "query_details_file": query_details_source,
                "query_details_chars": len(query_details_text),
                "output_root": str(args.output_root.expanduser().resolve()),
                "llm": {
                    "mode": "frozen_profile"
                    if args.frozen_profile is not None
                    else "live_generation",
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                },
                "frozen_profile": (
                    str(args.frozen_profile.expanduser().resolve())
                    if args.frozen_profile is not None
                    else None
                ),
                "frozen_corpus_manifest": (
                    str(args.frozen_corpus_manifest.expanduser().resolve())
                    if args.frozen_corpus_manifest is not None
                    else None
                ),
                "frozen_retrieval_manifest": (
                    str(args.frozen_retrieval_manifest.expanduser().resolve())
                    if args.frozen_retrieval_manifest is not None
                    else None
                ),
                "input_ids_file": (
                    str(args.input_ids_file.expanduser().resolve())
                    if args.input_ids_file is not None
                    else None
                ),
                "mapping_threshold": float(args.mapping_threshold),
                "download_limit": args.download_limit,
                "reviewed_only": bool(args.reviewed_only),
                "skip_filtering": bool(args.skip_filtering),
                "skip_active_site_filter": bool(args.skip_active_site_filter),
                "motif_filter_mode": (
                    "skip"
                    if args.skip_active_site_filter
                    else "explicit"
                    if args.enable_active_site_filter
                    else "automatic"
                ),
            },
        )
        if args.download_limit is None:
            _log(
                "warning: --download-limit not set. Large InterPro/Pfam families "
                "can pull many thousands of accessions; consider --download-limit."
            )

        # Profile generation or loading.
        enzyme_profile_path = llm_dir / "enzyme_profile.json"
        if args.frozen_profile is not None:
            frozen_profile_path = args.frozen_profile.expanduser().resolve()
            profile = normalize_profile(
                json.loads(frozen_profile_path.read_text(encoding="utf-8")),
                query_seq=query_seq,
            )
            enzyme_profile_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log(f"loaded frozen profile: {frozen_profile_path}")
        elif args.resume and enzyme_profile_path.exists():
            profile = json.loads(enzyme_profile_path.read_text(encoding="utf-8"))
            _log(f"resume: reusing enzyme_profile.json ({enzyme_profile_path})")
        else:
            profile = generate_profile(
                provider=provider,
                base_url=base_url,
                model=model,
                enzyme_type=args.enzyme_type,
                query_seq=query_seq,
                llm_dir=llm_dir,
                query_name=query_name,
                query_details_text=query_details_text,
                max_tokens=8192,
                max_attempts=1,
            )

        profile, applied_minimal_motif = apply_minimal_conserved_motif_overrides(
            profile,
            query_name=query_name,
            query_seq=query_seq,
            query_aliases=(args.enzyme_type, str(profile.get("enzyme_type_normalized") or "")),
        )
        if applied_minimal_motif:
            enzyme_profile_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log(
                "minimal conserved motif overrides applied: "
                f"query={query_name} patterns={applied_minimal_motif['patterns']}"
            )

        user_active_site_override_requested = (
            args.active_site_motif_regex is not None
            or args.active_site_rules_json is not None
            or args.active_site_policy is not None
            or args.active_site_min_hits is not None
        )
        if user_active_site_override_requested:
            profile, applied_override = apply_user_active_site_overrides(
                profile,
                query_seq=query_seq,
                motif_regex_override=args.active_site_motif_regex,
                rules_json_override=args.active_site_rules_json,
                policy_override=args.active_site_policy,
                min_hits_override=args.active_site_min_hits,
            )
            enzyme_profile_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log(f"user active-site overrides applied: {sorted(applied_override)}")

        if args.frozen_retrieval_manifest is not None:
            from enzformer2.profile.offline import validate_frozen_retrieval_manifest

            retrieval_summary = validate_frozen_retrieval_manifest(
                args.frozen_retrieval_manifest.expanduser().resolve(),
                profile_path=args.frozen_profile.expanduser().resolve(),
                corpus_manifest=(
                    args.frozen_corpus_manifest.expanduser().resolve()
                    if args.frozen_corpus_manifest is not None
                    else None
                ),
            )
            status["steps"]["frozen_retrieval_validated"] = retrieval_summary
            _write_step_status(step_status_path, status)

        pfam_id = str(profile.get("pfam_id") or "").strip()
        interpro_id = str(profile.get("interpro_id") or "").strip()
        confidence = float(profile.get("confidence") or 0.0)
        status["steps"]["profile_generated"] = {
            "pfam_id": pfam_id,
            "interpro_id": interpro_id,
            "confidence": confidence,
        }
        _write_step_status(step_status_path, status)

        if confidence < float(args.mapping_threshold) or (not pfam_id and not interpro_id):
            report_path = write_mapping_failure_report(
                llm_dir=llm_dir,
                enzyme_type=args.enzyme_type,
                profile=profile,
                threshold=float(args.mapping_threshold),
                reason=(
                    "confidence_below_threshold"
                    if confidence < float(args.mapping_threshold)
                    else "missing_pfam_and_interpro"
                ),
            )
            status["failed"] = True
            status["failure_reason"] = (
                f"profile mapping insufficient (confidence={confidence:.3f} "
                f"pfam={pfam_id!r} interpro={interpro_id!r}); see {report_path}"
            )
            _write_step_status(step_status_path, status)
            raise RuntimeError(status["failure_reason"])

        # Retrieval configuration and execution.
        enzyme_prefix = fasta_utils.enzyme_download_prefix(args.enzyme_type, profile)
        output_prefix = f"{enzyme_prefix}_union"
        length_window = resolve_download_length_window(
            query_seq=query_seq,
            min_len_override=args.download_min_len,
            max_len_override=args.download_max_len,
        )
        write_json(logs_dir / "download_length_window.json", length_window)

        filtered_fasta = download_dir / f"{output_prefix}.filtered.fasta"
        raw_fasta = download_dir / f"{output_prefix}.raw.fasta"
        retrieval_config_path = llm_dir / "retrieval_config.json"
        profile_sha256 = hashlib.sha256(enzyme_profile_path.read_bytes()).hexdigest()
        write_json(
            retrieval_config_path,
            {
                "profile_json": str(enzyme_profile_path),
                "profile_sha256": profile_sha256,
                "pfam_id": pfam_id or None,
                "interpro_id": interpro_id or None,
                "input_ids_file": (
                    str(args.input_ids_file.expanduser().resolve())
                    if args.input_ids_file is not None
                    else None
                ),
                "reviewed_only": bool(args.reviewed_only),
                "filtering": {
                    "skip_filtering": bool(args.skip_filtering),
                    "enable_active_site_filter": bool(args.enable_active_site_filter),
                    "skip_active_site_filter": bool(args.skip_active_site_filter),
                    "motif_filter_mode": (
                        "skip"
                        if args.skip_active_site_filter
                        else "explicit"
                        if args.enable_active_site_filter
                        else "automatic"
                    ),
                    "motif_regex": profile.get("motif_regex"),
                    "active_site_policy": profile.get("active_site_policy"),
                    "active_site_min_hits": profile.get("active_site_min_hits"),
                    "active_site_rules": profile.get("active_site_rules") or [],
                },
                "length_window": length_window,
                "frozen_corpus_manifest": (
                    str(args.frozen_corpus_manifest.expanduser().resolve())
                    if args.frozen_corpus_manifest is not None
                    else None
                ),
                "frozen_retrieval_manifest": (
                    str(args.frozen_retrieval_manifest.expanduser().resolve())
                    if args.frozen_retrieval_manifest is not None
                    else None
                ),
            },
        )
        status["steps"]["retrieval_config"] = {
            "path": str(retrieval_config_path),
            "profile_sha256": profile_sha256,
        }
        _write_step_status(step_status_path, status)
        frozen_corpus_summary = None
        if args.frozen_corpus_manifest is not None:
            from enzformer2.profile.offline import copy_frozen_corpus

            frozen_corpus_summary = copy_frozen_corpus(
                args.frozen_corpus_manifest.expanduser().resolve(), single_dir
            )
            retrieval_config = json.loads(retrieval_config_path.read_text(encoding="utf-8"))
            retrieval_config["frozen_corpus"] = frozen_corpus_summary
            write_json(retrieval_config_path, retrieval_config)
            status["steps"]["download_completed"] = {
                "records": frozen_corpus_summary["sequence_count"],
                "source": "frozen_corpus",
                "corpus": frozen_corpus_summary,
            }
            _write_step_status(step_status_path, status)
        elif args.resume and filtered_fasta.exists() and raw_fasta.exists():
            _log(f"resume: reusing downloader outputs under {download_dir}")
        else:
            cmd = _build_downloader_cmd(
                profile_path=enzyme_profile_path,
                pfam_id=pfam_id,
                interpro_id=interpro_id,
                outdir=download_dir,
                output_prefix=output_prefix,
                length_window=length_window,
                args=args,
                profile=profile,
            )
            run_logged(cmd=cmd, cwd=_repo_root(), log_path=logs_dir / "download_step.log")

        # Split downloaded records into one FASTA file per sequence.
        if frozen_corpus_summary is not None:
            manifest_path = single_dir / args.frozen_corpus_manifest.name
            manifest_rows = fasta_utils.read_manifest(manifest_path)
            dedup_stats = {
                "final_download_records": sum(
                    1 for row in manifest_rows if row.get("source") != "query"
                ),
                "dedup_accession_removed": 0,
                "dedup_sequence_removed": 0,
            }
            status["steps"]["single_fasta"] = {
                "final_download_records": dedup_stats["final_download_records"],
                "dedup_accession_removed": 0,
                "dedup_sequence_removed": 0,
                "manifest": str(manifest_path),
                "source": "frozen_corpus",
            }
            _write_step_status(step_status_path, status)
            _log(
                f"run_id={run_id} "
                f"download_records={dedup_stats['final_download_records']} "
                f"manifest={manifest_path}"
            )
            print(f"profile run complete: run_id={run_id}")
            print(f"manifest={manifest_path}")
            return

        final_source_fasta = filtered_fasta if filtered_fasta.exists() else raw_fasta
        if not final_source_fasta.exists():
            raise RuntimeError(
                f"downloader did not produce any FASTA output: expected {filtered_fasta} or {raw_fasta}"
            )
        download_records = fasta_utils.parse_fasta_records(final_source_fasta)
        status["steps"]["download_completed"] = {
            "records": len(download_records),
            "filtered_fasta": str(filtered_fasta) if filtered_fasta.exists() else None,
            "raw_fasta": str(raw_fasta) if raw_fasta.exists() else None,
            "used_fasta": str(final_source_fasta),
        }
        _write_step_status(step_status_path, status)

        manifest_rows, dedup_stats = fasta_utils.split_single_fastas(
            records=download_records,
            single_dir=single_dir,
            query_seq=query_seq,
            query_name="QUERY_INPUT",
        )
        manifest_path = single_dir / "manifest.tsv"
        fasta_utils.write_manifest(manifest_path, manifest_rows)

        status["steps"]["single_fasta"] = {
            "final_download_records": dedup_stats["final_download_records"],
            "dedup_accession_removed": dedup_stats["dedup_accession_removed"],
            "dedup_sequence_removed": dedup_stats["dedup_sequence_removed"],
            "manifest": str(manifest_path),
        }
        _write_step_status(step_status_path, status)

        _log(
            f"run_id={run_id} "
            f"download_records={dedup_stats['final_download_records']} "
            f"manifest={manifest_path}"
        )
        print(f"profile run complete: run_id={run_id}")
        print(f"manifest={manifest_path}")
        print(f"filtered_fasta={filtered_fasta}")
    except Exception as exc:
        status["failed"] = True
        status.setdefault("failure_reason", f"{type(exc).__name__}: {exc}")
        status["traceback"] = traceback.format_exc()
        _write_step_status(step_status_path, status)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
