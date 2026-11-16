"""Enzyme profile generation, UniProt grounding, and active-site validation.

The module covers structured profile creation and retrieval constraints; it
does not create activity labels or downstream embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from enzformer2.profile.llm import request_chat_text
from enzformer2.profile.runtime import make_logger

_log = make_logger("enzformer2.profile")
DEFAULT_MINIMAL_MOTIF_TABLE = Path(__file__).with_name("Minimal_conserved_motif_regex.md")
GROUNDING_CONTAMINATION_RULES = Path(__file__).with_name("grounding_contamination.json")
PROFILE_SCHEMA_VERSION = "1.0"
PROFILE_PROMPT_VERSION = "1.0"
MAX_MINIMAL_MOTIF_SPAN = 10
MIN_MOTIF_CONSTRAINED_POSITIONS = 4
MIN_MOTIF_UNIQUE_SEQUENCES = 5000
_PROFILE_SYSTEM_PROMPT_RESOURCE = "prompts/profile.system.txt"
_PROFILE_USER_PROMPT_RESOURCE = "prompts/profile.user.txt"


# JSON helpers


def extract_first_json_block(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip().startswith("```"):
            s = "\n".join(lines[1:-1]).strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            _, end = dec.raw_decode(s[i:])
            return s[i : i + end]
        except json.JSONDecodeError:
            continue
    return s


def parse_json_object(text: str) -> Dict[str, Any]:
    s = extract_first_json_block(text)
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    return obj


def llm_chat(
    *,
    provider: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1400,
    json_response: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    json_tool: Optional[Dict[str, Any]] = None,
    retries: int = 0,
    timeout_sec: float = 600.0,
) -> str:
    return request_chat_text(
        provider=provider,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format_json=json_response,
        response_json_schema=json_schema,
        json_tool=json_tool,
        retries=retries,
        timeout_sec=timeout_sec,
    )


def _profile_from_llm_raw(
    raw: str,
    *,
    grounding: Dict[str, Any],
    query_seq: str,
) -> Dict[str, Any]:
    parsed = parse_json_object(raw)
    profile = normalize_profile(parsed, query_seq=query_seq)
    return apply_grounding_to_profile(profile, grounding, query_seq=query_seq)


PROFILE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "enzyme_profile",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "enzyme_type_normalized": {"type": "string"},
            "pfam_id": {"type": "string"},
            "interpro_id": {"type": "string"},
            "motif_regex": {"type": ["string", "null"]},
            "minimal_conserved_motif_regexes": {"type": "array", "items": {"type": "string"}},
            "minimal_conserved_motif_rationale": {"type": "string"},
            "minimal_conserved_motif_source": {"type": "string"},
            "extra_uniprot_query": {"type": ["string", "null"]},
            "positive_tokens": {"type": "array", "items": {"type": "string"}},
            "negative_tokens": {"type": "array", "items": {"type": "string"}},
            "ec_numbers": {"type": "array", "items": {"type": "string"}},
            "go_terms": {"type": "array", "items": {"type": "string"}},
            "active_site_rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "regex": {"type": ["string", "null"]},
                        "position_1based": {"type": ["integer", "null"]},
                        "allowed_aas": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "tolerance": {"type": "integer"},
                        "required": {"type": "boolean"},
                    },
                    "required": [
                        "id",
                        "regex",
                        "position_1based",
                        "allowed_aas",
                        "tolerance",
                        "required",
                    ],
                },
            },
            "active_site_policy": {"type": "string", "enum": ["all_required", "min_required"]},
            "active_site_min_hits": {"type": ["integer", "null"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pfam_id": {"type": "string"},
                        "interpro_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["pfam_id", "interpro_id", "reason", "confidence"],
                },
            },
        },
        "required": [
            "enzyme_type_normalized",
            "pfam_id",
            "interpro_id",
            "motif_regex",
            "minimal_conserved_motif_regexes",
            "minimal_conserved_motif_rationale",
            "minimal_conserved_motif_source",
            "extra_uniprot_query",
            "positive_tokens",
            "negative_tokens",
            "ec_numbers",
            "go_terms",
            "active_site_rules",
            "active_site_policy",
            "active_site_min_hits",
            "confidence",
            "candidates",
        ],
    },
}

PROFILE_JSON_TOOL: Dict[str, Any] = {
    "name": "emit_enzyme_profile",
    "description": "Emit the query-specific enzyme profile JSON.",
    "parameters": PROFILE_JSON_SCHEMA["schema"],
}


# UniProt grounding


def _split_multi_values(raw: str) -> List[str]:
    if not raw:
        return []
    out: List[str] = []
    for token in re.split(r"[;,]", raw):
        t = token.strip()
        if t:
            out.append(t)
    return out


def _pick_field(row: Dict[str, str], keywords: Sequence[str]) -> str:
    lower_map = {k.lower(): v for k, v in row.items()}
    for key_l, val in lower_map.items():
        if all(kw in key_l for kw in keywords):
            return val
    return ""


def fetch_uniprot_grounding(enzyme_type: str, size: int = 25) -> Dict[str, Any]:
    queries = [
        f'"{enzyme_type}" AND reviewed:true',
        f"{enzyme_type} AND reviewed:true",
    ]
    queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    last_err = ""
    for q in queries:
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {
                "query": q,
                "fields": "accession,protein_name,xref_pfam,xref_interpro,ec,go_id",
                "format": "tsv",
                "size": str(int(size)),
            }
        )
        try:
            req = urllib.request.Request(url=url, method="GET")
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = str(exc)
            continue

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 1:
            continue

        header = lines[0].split("\t")
        pfam_counter: Counter[str] = Counter()
        interpro_counter: Counter[str] = Counter()
        pair_counter: Counter[Tuple[str, str]] = Counter()
        ec_counter: Counter[str] = Counter()
        sample_entries: List[Dict[str, str]] = []

        for row_line in lines[1:]:
            cols = row_line.split("\t")
            row = {h: (cols[i] if i < len(cols) else "") for i, h in enumerate(header)}

            acc = _pick_field(row, ("entry",))
            pname = _pick_field(row, ("protein",))
            if acc:
                sample_entries.append({"accession": acc, "protein_name": pname})

            pfams = [
                x.upper()
                for x in _split_multi_values(_pick_field(row, ("pfam",)))
                if re.fullmatch(r"PF\d{5}", x.upper())
            ]
            interpros = [
                x.upper()
                for x in _split_multi_values(_pick_field(row, ("interpro",)))
                if re.fullmatch(r"IPR\d{6}", x.upper())
            ]
            ecs = [
                x
                for x in _split_multi_values(_pick_field(row, ("ec",)))
                if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", x)
            ]

            pfam_counter.update(pfams)
            interpro_counter.update(interpros)
            ec_counter.update(ecs)
            for p in pfams:
                for ip in interpros:
                    pair_counter[(p, ip)] += 1

        if not sample_entries:
            continue

        return {
            "query": q,
            "hit_count": len(sample_entries),
            "top_pfams": [{"id": k, "count": int(v)} for k, v in pfam_counter.most_common(8)],
            "top_interpros": [
                {"id": k, "count": int(v)} for k, v in interpro_counter.most_common(8)
            ],
            "top_pairs": [
                {"pfam_id": p, "interpro_id": ip, "count": int(v)}
                for (p, ip), v in pair_counter.most_common(12)
            ],
            "top_ec_numbers": [{"id": k, "count": int(v)} for k, v in ec_counter.most_common(8)],
            "sample_entries": sample_entries[:10],
        }

    return {
        "query": queries[0] if queries else "",
        "hit_count": 0,
        "top_pfams": [],
        "top_interpros": [],
        "top_pairs": [],
        "top_ec_numbers": [],
        "sample_entries": [],
        "error": last_err,
    }


@lru_cache(maxsize=1)
def _load_grounding_contamination_rules() -> List[Dict[str, Any]]:
    try:
        payload = json.loads(GROUNDING_CONTAMINATION_RULES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Grounding contamination rules are missing or invalid: {GROUNDING_CONTAMINATION_RULES}"
        ) from exc
    rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(rules, list):
        raise ValueError(
            f"Grounding contamination rules must contain a rules list: {GROUNDING_CONTAMINATION_RULES}"
        )
    return [rule for rule in rules if isinstance(rule, dict)]


def _derive_grounding_contamination_negative_tokens(
    *,
    enzyme_type: str,
    grounding: Dict[str, Any],
) -> List[str]:
    """Return contamination negative tokens for enzyme families where UniProt
    grounding frequently picks up fused/partner contaminant hits.

    Rules are versioned in ``grounding_contamination.json`` and are activated
    only when their configured enzyme-name pattern matches.
    """

    enzyme_type_l = str(enzyme_type or "").strip().lower()

    sample_entries = grounding.get("sample_entries") or []
    sample_text = "\n".join(
        str(item.get("protein_name") or "") for item in sample_entries if isinstance(item, dict)
    ).lower()
    top_pfams = {
        str(item.get("id") or "").strip().upper()
        for item in (grounding.get("top_pfams") or [])
        if isinstance(item, dict)
    }
    top_ecs = {
        str(item.get("id") or "").strip()
        for item in (grounding.get("top_ec_numbers") or [])
        if isinstance(item, dict)
    }

    out: List[str] = []

    def _add(*tokens: str) -> None:
        for token in tokens:
            tok = str(token).strip()
            if tok and tok not in out:
                out.append(tok)

    for rule in _load_grounding_contamination_rules():
        enzyme_patterns = [str(value).lower() for value in rule.get("enzyme_type_any", [])]
        if not any(pattern and pattern in enzyme_type_l for pattern in enzyme_patterns):
            continue
        for pfam_id, tokens in (rule.get("pfam_tokens") or {}).items():
            if str(pfam_id).upper() in top_pfams:
                _add(*(str(token) for token in tokens))
        for ec_id, tokens in (rule.get("ec_tokens") or {}).items():
            if str(ec_id) in top_ecs:
                _add(*(str(token) for token in tokens))
        for sample_token, tokens in (rule.get("protein_name_tokens") or {}).items():
            if str(sample_token).lower() in sample_text:
                _add(*(str(token) for token in tokens))
    return out


def normalize_uniprot_query_fields(extra_query: Optional[str]) -> Optional[str]:
    q = (extra_query or "").strip()
    if not q:
        return None
    # UniProt REST uses organism_name/organism_id; older examples often use organism.
    q = re.sub(r"(?i)(?<![A-Za-z0-9_])organism\s*:", "organism_name:", q)
    return q


def strip_reviewed_filter(extra_query: Optional[str]) -> Optional[str]:
    q = (extra_query or "").strip()
    if not q:
        return None
    q = normalize_uniprot_query_fields(q) or ""
    q = re.sub(r"(?i)\breviewed\s*:\s*(?:true|false)\b", "", q)
    q = re.sub(r"\(\s*\)", "", q)
    q = re.sub(r"(?i)\b(AND|OR)\s+\)", ")", q)
    q = re.sub(r"(?i)\(\s*(AND|OR)\b", "(", q)
    q = re.sub(r"(?i)\b(AND|OR)\s+(AND|OR)\b", r"\1", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"(?i)^(AND|OR)\b\s*", "", q).strip()
    q = re.sub(r"(?i)\s*\b(AND|OR)$", "", q).strip()
    if not q:
        return None
    acc_only = re.fullmatch(
        r"(?i)\(?\s*(?:accession\s*:\s*)?([AOPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})\s*\)?",
        q,
    )
    if acc_only:
        return None
    return q


# Active-site rule + motif post-validation (against query sequence)


def _normalize_active_site_rule(rule: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(rule, dict):
        return None

    rule_id = str(rule.get("id") or f"rule_{index}").strip() or f"rule_{index}"

    regex_raw = rule.get("regex")
    regex = str(regex_raw).strip() if regex_raw is not None else ""
    regex = regex or None

    pos_raw = rule.get("position_1based")
    try:
        pos_val = int(pos_raw) if pos_raw is not None and str(pos_raw).strip() else None
    except (TypeError, ValueError):
        pos_val = None
    if pos_val is not None and pos_val <= 0:
        pos_val = None

    allowed_raw = rule.get("allowed_aas")
    allowed: List[str] = []
    if isinstance(allowed_raw, list):
        for aa in allowed_raw:
            tok = str(aa).strip().upper()
            if re.fullmatch(r"[A-Z]", tok) and tok not in allowed:
                allowed.append(tok)
    allowed_out = allowed or None

    tol_raw = rule.get("tolerance")
    try:
        tol_val = int(tol_raw) if tol_raw is not None and str(tol_raw).strip() else 0
    except (TypeError, ValueError):
        tol_val = 0
    if tol_val < 0:
        tol_val = 0

    required = bool(rule.get("required", True))

    if regex is None and pos_val is None:
        return None

    return {
        "id": rule_id,
        "regex": regex,
        "position_1based": pos_val,
        "allowed_aas": allowed_out,
        "tolerance": tol_val,
        "required": required,
    }


def _normalize_protein_sequence(seq: Optional[str]) -> str:
    return re.sub(r"[^A-Za-z]", "", str(seq or "")).upper()


def _normalize_motif_pattern(motif_regex: Optional[str]) -> str:
    pat = str(motif_regex or "").strip()
    if not pat:
        return ""
    if re.search(r"[.\[\](){}|+*?^$\\]", pat):
        return pat
    return re.escape(pat).replace("X", ".").replace("x", ".")


def _query_has_motif(seq: str, motif_regex: Optional[str]) -> bool:
    pat = _normalize_motif_pattern(motif_regex)
    if not pat:
        return True
    try:
        return re.search(pat, seq, flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _benchmark_id_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _collapse_redundant_motif_patterns(patterns: Sequence[str]) -> List[str]:
    """Drop narrower alternatives when the table also provides a broader one."""

    unique: List[str] = []
    for pattern in patterns:
        pat = str(pattern or "").strip()
        if pat and pat not in unique:
            unique.append(pat)

    collapsed: List[str] = []
    for pat in unique:
        redundant = False
        for other in unique:
            if other == pat:
                continue
            if pat in other:
                redundant = True
                break
            try:
                if re.fullmatch(other, pat, flags=re.IGNORECASE):
                    redundant = True
                    break
            except re.error:
                continue
        if not redundant:
            collapsed.append(pat)
    return collapsed


def _compose_required_motif_regex(patterns: Sequence[str]) -> Optional[str]:
    pats = _collapse_redundant_motif_patterns(patterns)
    if not pats:
        return None
    if len(pats) == 1:
        return pats[0]
    return "".join(f"(?=.*(?:{pat}))" for pat in pats)


def _to_nonempty_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        raw_values = re.split(r"(?:<br\s*/?>|\r?\n)+", text, flags=re.IGNORECASE)

    out: List[str] = []
    for raw in raw_values:
        s = str(raw or "").strip()
        s = re.sub(r"^`|`$", "", s).strip()
        if s and s not in out:
            out.append(s)
    return out


def _extract_minimal_conserved_motif_patterns(obj: Dict[str, Any]) -> List[str]:
    for key in (
        "minimal_conserved_motif_regexes",
        "minimal_conserved_motif_patterns",
        "minimal_conserved_motif_regex",
    ):
        patterns = _to_nonempty_str_list(obj.get(key))
        if patterns:
            return _collapse_redundant_motif_patterns(patterns)
    return []


def _validate_regex_or_raise(pattern: str, *, field_name: str) -> None:
    try:
        re.compile(_normalize_motif_pattern(pattern), flags=re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"{field_name} is not a valid regex: {pattern!r}") from exc


def _regex_consumption_metrics(pattern: str) -> Tuple[int, int]:
    span = 0
    constrained = 0
    index = 0
    while index < len(pattern):
        token = pattern[index]
        if token in "^$":
            index += 1
            continue
        if token in "()|*+?":
            return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
        if token == "\\":
            if index + 1 >= len(pattern):
                return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
            token_span = 1
            token_constrained = 1
            index += 2
        elif token == "[":
            closing = index + 1
            while closing < len(pattern):
                if pattern[closing] == "]" and pattern[closing - 1] != "\\":
                    break
                closing += 1
            if closing >= len(pattern):
                return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
            token_span = 1
            token_constrained = 1
            index = closing + 1
        elif token == ".":
            token_span = 1
            token_constrained = 0
            index += 1
        elif token == "{":
            return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
        else:
            token_span = 1
            token_constrained = 1
            index += 1
        minimum = 1
        maximum = 1
        if index < len(pattern) and pattern[index] == "{":
            closing = pattern.find("}", index + 1)
            if closing < 0:
                return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
            bounds = pattern[index + 1 : closing].split(",", 1)
            if not bounds[0].isdigit():
                return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
            minimum = int(bounds[0])
            if len(bounds) == 1:
                maximum = minimum
            elif not bounds[1].isdigit():
                return MAX_MINIMAL_MOTIF_SPAN + 1, constrained
            else:
                maximum = int(bounds[1])
            index = closing + 1
        span += maximum * token_span
        constrained += minimum * token_constrained
    return span, constrained


def minimal_motif_pattern_metrics(patterns: Sequence[str]) -> Dict[str, int]:
    normalized = _collapse_redundant_motif_patterns(patterns)
    metrics = [
        _regex_consumption_metrics(_normalize_motif_pattern(pattern)) for pattern in normalized
    ]
    return {
        "pattern_count": len(normalized),
        "max_span": max((item[0] for item in metrics), default=0),
        "constrained_positions": sum(item[1] for item in metrics),
    }


def minimal_motif_filter_decision(
    patterns: Sequence[str],
    *,
    unique_sequences_after_filter: int,
) -> Dict[str, Any]:
    metrics = minimal_motif_pattern_metrics(patterns)
    eligible = bool(metrics["pattern_count"])
    reason = "eligible"
    if not eligible:
        reason = "no_motif"
    elif metrics["max_span"] > MAX_MINIMAL_MOTIF_SPAN:
        eligible = False
        reason = "motif_span"
    elif metrics["constrained_positions"] < MIN_MOTIF_CONSTRAINED_POSITIONS:
        eligible = False
        reason = "motif_information"
    elif int(unique_sequences_after_filter) < MIN_MOTIF_UNIQUE_SEQUENCES:
        eligible = False
        reason = "corpus_size"
    return {
        **metrics,
        "unique_sequences_after_filter": int(unique_sequences_after_filter),
        "max_span_limit": MAX_MINIMAL_MOTIF_SPAN,
        "min_constrained_positions": MIN_MOTIF_CONSTRAINED_POSITIONS,
        "min_unique_sequences": MIN_MOTIF_UNIQUE_SEQUENCES,
        "applied": eligible,
        "reason": reason,
    }


def _minimal_motif_active_site_rules(
    patterns: Sequence[str],
    *,
    rule_prefix: str = "minimal_conserved_motif",
) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"{rule_prefix}_{idx}",
            "regex": pattern,
            "position_1based": None,
            "allowed_aas": None,
            "tolerance": 0,
            "required": True,
        }
        for idx, pattern in enumerate(patterns, start=1)
    ]


def load_minimal_conserved_motif_table(
    path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Read the repo-local benchmark motif table.

    The table is intentionally human-editable markdown. Regexes must be written
    in backticks in the second column.
    """

    table_path = Path(path) if path is not None else DEFAULT_MINIMAL_MOTIF_TABLE
    if not table_path.exists():
        raise FileNotFoundError(
            "Minimal conserved motif table not found: "
            f"{table_path}. Supply motif_table_path or restore the tracked benchmark table."
        )

    out: Dict[str, List[str]] = {}
    for raw_line in table_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line or "Minimal conserved" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        benchmark_id = re.sub(r"[*`]", "", cells[0]).strip()
        patterns = re.findall(r"`([^`]+)`", cells[1])
        patterns = [pat.strip() for pat in patterns if pat.strip()]
        if benchmark_id and patterns:
            out[_benchmark_id_key(benchmark_id)] = patterns
    return out


def apply_minimal_conserved_motif_overrides(
    profile: Dict[str, Any],
    *,
    query_name: str,
    query_seq: str,
    motif_table_path: Optional[Path] = None,
    query_aliases: Sequence[str] = (),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    table = load_minimal_conserved_motif_table(motif_table_path)
    lookup_keys = [
        _benchmark_id_key(value)
        for value in (query_name, *query_aliases)
        if _benchmark_id_key(value)
    ]
    matched_key = next((key for key in lookup_keys if key in table), None)
    if matched_key is None:
        matched_key = next(
            (
                table_key
                for lookup_key in lookup_keys
                for table_key in sorted(table, key=len, reverse=True)
                if table_key in lookup_key
            ),
            None,
        )
    patterns = table.get(matched_key or "")
    if not patterns:
        return dict(profile), {}

    collapsed_patterns = _collapse_redundant_motif_patterns(patterns)
    motif_regex = _compose_required_motif_regex(collapsed_patterns)
    if not motif_regex:
        return dict(profile), {}

    seq = _normalize_protein_sequence(query_seq)
    if seq and not _query_has_motif(seq, motif_regex):
        raise ValueError(
            f"Minimal conserved motif regex does not match query {query_name!r}: {motif_regex!r}"
        )

    rules = _minimal_motif_active_site_rules(
        collapsed_patterns,
        rule_prefix=f"{_benchmark_id_key(query_name) or 'benchmark'}_minimal_motif",
    )

    out = dict(profile)
    out["motif_regex"] = motif_regex
    out["active_site_rules"] = rules
    out["active_site_policy"] = "all_required"
    out["active_site_min_hits"] = None
    motif_source = str(
        (Path(motif_table_path) if motif_table_path is not None else DEFAULT_MINIMAL_MOTIF_TABLE)
    )
    out = normalize_profile(out, query_seq=query_seq)
    out["motif_source"] = "minimal_motif_table"
    out["minimal_conserved_motif_source"] = motif_source
    out["minimal_conserved_motif_patterns"] = collapsed_patterns

    applied = {
        "query_name": query_name,
        "motif_key": matched_key,
        "motif_regex": out.get("motif_regex"),
        "patterns": collapsed_patterns,
        "active_site_rules_count": len(out.get("active_site_rules") or []),
        "source": out.get("minimal_conserved_motif_source"),
    }
    return out, applied


def _query_matches_position_rule(
    seq: str,
    pos_1based: int,
    allowed_aas: Optional[List[str]],
    tolerance: int,
) -> bool:
    if pos_1based <= 0 or not seq:
        return False
    seq_len = len(seq)
    center = int(pos_1based)
    tol = max(0, int(tolerance or 0))
    allowed = [str(aa).strip().upper() for aa in (allowed_aas or []) if str(aa).strip()]
    for p in range(center - tol, center + tol + 1):
        idx = p - 1
        if idx < 0 or idx >= seq_len:
            continue
        if not allowed or seq[idx].upper() in allowed:
            return True
    return False


def _query_matches_active_site_rule(seq: str, rule: Dict[str, Any]) -> bool:
    regex = rule.get("regex")
    pos = rule.get("position_1based")
    allowed_aas = rule.get("allowed_aas")
    tolerance = int(rule.get("tolerance") or 0)

    regex_ok = True
    if regex:
        regex_ok = _query_has_motif(seq, str(regex))

    pos_ok = True
    if pos is not None:
        pos_ok = _query_matches_position_rule(
            seq=seq,
            pos_1based=int(pos),
            allowed_aas=(allowed_aas if isinstance(allowed_aas, list) else None),
            tolerance=tolerance,
        )

    return bool(regex_ok and pos_ok)


def _sanitize_profile_constraints_for_query(
    profile: Dict[str, Any], query_seq: Optional[str]
) -> Dict[str, Any]:
    seq = _normalize_protein_sequence(query_seq)
    if not seq:
        return dict(profile)

    out = dict(profile)
    motif = str(out.get("motif_regex") or "").strip()
    if motif and not _query_has_motif(seq, motif):
        out["motif_regex"] = None

    active_rules: List[Dict[str, Any]] = []
    raw_rules = out.get("active_site_rules")
    if isinstance(raw_rules, list):
        for idx, rule in enumerate(raw_rules, start=1):
            norm = _normalize_active_site_rule(rule, idx)
            if norm is None:
                continue
            if _query_matches_active_site_rule(seq, norm):
                active_rules.append(norm)
    out["active_site_rules"] = active_rules

    min_hits = out.get("active_site_min_hits")
    try:
        min_hits_val = int(min_hits) if min_hits is not None else None
    except (TypeError, ValueError):
        min_hits_val = None
    if min_hits_val is not None and min_hits_val <= 0:
        min_hits_val = None
    if min_hits_val is not None and active_rules:
        min_hits_val = max(1, min(min_hits_val, len(active_rules)))
    if not active_rules:
        min_hits_val = None
    out["active_site_min_hits"] = min_hits_val
    return out


def _parse_active_site_rules_json_text(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return []
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --active-site-rules-json: {exc}") from exc
    if not isinstance(obj, list):
        raise ValueError("--active-site-rules-json must be a JSON list")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(obj, start=1):
        norm = _normalize_active_site_rule(item, idx)
        if norm is None:
            raise ValueError(
                f"--active-site-rules-json contains an invalid rule at index {idx}; "
                "each rule must provide regex and/or position_1based"
            )
        out.append(norm)
    return out


def apply_user_active_site_overrides(
    profile: Dict[str, Any],
    *,
    query_seq: str,
    motif_regex_override: Optional[str] = None,
    rules_json_override: Optional[str] = None,
    policy_override: Optional[str] = None,
    min_hits_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(profile)
    seq = _normalize_protein_sequence(query_seq)
    applied: Dict[str, Any] = {}
    parsed_rules = _parse_active_site_rules_json_text(rules_json_override)
    constraint_override = (
        motif_regex_override is not None
        or parsed_rules is not None
        or policy_override is not None
        or min_hits_override is not None
    )

    if constraint_override:
        for key in (
            "minimal_conserved_motif_regexes",
            "minimal_conserved_motif_regex",
            "minimal_conserved_motif_patterns",
            "minimal_conserved_motif_source",
            "minimal_conserved_motif_rationale",
        ):
            out.pop(key, None)

    if motif_regex_override is not None:
        motif = str(motif_regex_override).strip()
        if motif and seq and not _query_has_motif(seq, motif):
            raise ValueError(
                f"User-provided active-site motif does not match the query sequence: {motif!r}"
            )
        out["motif_regex"] = motif or None
        applied["motif_regex"] = out["motif_regex"]
        if parsed_rules is None:
            out["active_site_rules"] = []
            out["active_site_min_hits"] = None

    if parsed_rules is not None:
        bad_rule_ids = [
            str(rule.get("id") or "")
            for rule in parsed_rules
            if seq and not _query_matches_active_site_rule(seq, rule)
        ]
        if bad_rule_ids:
            raise ValueError(
                "User-provided active-site rules do not match the query sequence: "
                + ", ".join(bad_rule_ids)
            )
        out["active_site_rules"] = parsed_rules
        applied["active_site_rules"] = parsed_rules

    if policy_override is not None:
        policy = str(policy_override).strip().lower()
        if policy not in ("all_required", "min_required"):
            raise ValueError("--active-site-policy must be one of: all_required, min_required")
        out["active_site_policy"] = policy
        applied["active_site_policy"] = policy

    if min_hits_override is not None:
        min_hits = int(min_hits_override)
        if min_hits <= 0:
            raise ValueError("--active-site-min-hits must be > 0 when provided")
        out["active_site_min_hits"] = min_hits
        applied["active_site_min_hits"] = min_hits

    out = normalize_profile(out, query_seq=query_seq)
    if constraint_override:
        out["motif_source"] = "user_override"
        for key in (
            "minimal_conserved_motif_regexes",
            "minimal_conserved_motif_regex",
            "minimal_conserved_motif_patterns",
            "minimal_conserved_motif_source",
            "minimal_conserved_motif_rationale",
        ):
            out.pop(key, None)

    if (
        motif_regex_override is not None
        and str(motif_regex_override).strip()
        and not out.get("motif_regex")
    ):
        raise ValueError(
            f"User-provided active-site motif was not retained after validation: {motif_regex_override!r}"
        )
    if parsed_rules is not None and len(out.get("active_site_rules") or []) != len(parsed_rules):
        raise ValueError("User-provided active-site rules were not retained after validation")

    return out, applied


# Download length window


def compute_query_aware_download_length_window(query_seq: str) -> Tuple[int, int]:
    seq = _normalize_protein_sequence(query_seq)
    if not seq:
        raise ValueError(
            "Query sequence is required to compute a query-aware download length window"
        )
    query_len = len(seq)
    min_len = max(30, int(math.floor(query_len * 0.75)))
    max_len = max(min_len + 20, int(math.ceil(query_len * 1.35)))
    return min_len, max_len


def resolve_download_length_window(
    *,
    query_seq: str,
    min_len_override: Optional[int],
    max_len_override: Optional[int],
) -> Dict[str, Any]:
    dyn_min, dyn_max = compute_query_aware_download_length_window(query_seq)
    min_len = int(min_len_override) if min_len_override is not None else dyn_min
    max_len = int(max_len_override) if max_len_override is not None else dyn_max
    if min_len <= 0 or max_len <= 0:
        raise ValueError("download length bounds must be positive integers")
    if min_len > max_len:
        raise ValueError(f"download length window is invalid: min_len={min_len} max_len={max_len}")
    mode = (
        "manual_override"
        if (min_len_override is not None or max_len_override is not None)
        else "query_aware"
    )
    return {
        "mode": mode,
        "query_length": len(_normalize_protein_sequence(query_seq)),
        "min_len": min_len,
        "max_len": max_len,
        "dynamic_min_len": dyn_min,
        "dynamic_max_len": dyn_max,
    }


# Profile normalization + grounding merge


def normalize_profile(obj: Dict[str, Any], query_seq: Optional[str] = None) -> Dict[str, Any]:
    def _valid_pfam(value: Any) -> str:
        s = str(value or "").strip().upper()
        return s if re.fullmatch(r"PF\d{5}", s) else ""

    def _valid_interpro(value: Any) -> str:
        s = str(value or "").strip().upper()
        return s if re.fullmatch(r"IPR\d{6}", s) else ""

    def _to_list_str(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        s = str(v).strip()
        return [s] if s else []

    def _to_conf(v: Any) -> float:
        try:
            c = float(v)
        except (TypeError, ValueError):
            c = 0.0
        if c < 0.0:
            return 0.0
        if c > 1.0:
            return 1.0
        return c

    profile: Dict[str, Any] = {}
    profile["enzyme_type_normalized"] = str(obj.get("enzyme_type_normalized") or "").strip()
    profile["pfam_id"] = _valid_pfam(obj.get("pfam_id"))
    profile["interpro_id"] = _valid_interpro(obj.get("interpro_id"))
    motif_regex_compat = str(obj.get("motif_regex")).strip() if obj.get("motif_regex") else None
    minimal_patterns = _extract_minimal_conserved_motif_patterns(obj)
    if minimal_patterns:
        for pattern in minimal_patterns:
            _validate_regex_or_raise(pattern, field_name="minimal_conserved_motif_regexes")
        minimal_motif_regex = _compose_required_motif_regex(minimal_patterns)
        if minimal_motif_regex:
            _validate_regex_or_raise(
                minimal_motif_regex,
                field_name="minimal_conserved_motif_regexes",
            )
        seq = _normalize_protein_sequence(query_seq)
        if seq:
            missing_patterns = [
                pattern for pattern in minimal_patterns if not _query_has_motif(seq, pattern)
            ]
            if missing_patterns:
                raise ValueError(
                    "minimal_conserved_motif_regexes do not match the query sequence: "
                    + ", ".join(repr(p) for p in missing_patterns)
                )
            if minimal_motif_regex and not _query_has_motif(seq, minimal_motif_regex):
                raise ValueError(
                    "minimal_conserved_motif_regexes combined regex does not match the query sequence: "
                    f"{minimal_motif_regex!r}"
                )
        profile["motif_regex"] = minimal_motif_regex
        profile["minimal_conserved_motif_patterns"] = minimal_patterns
        source = str(
            obj.get("minimal_conserved_motif_source")
            or obj.get("motif_source")
            or "llm_query_minimal"
        ).strip()
        profile["minimal_conserved_motif_source"] = source or "llm_query_minimal"
        rationale = str(obj.get("minimal_conserved_motif_rationale") or "").strip()
        if rationale:
            profile["minimal_conserved_motif_rationale"] = rationale
        profile["motif_source"] = str(obj.get("motif_source") or "llm_query_minimal").strip()
    else:
        profile["motif_regex"] = motif_regex_compat
        motif_source = str(obj.get("motif_source") or "").strip()
        if motif_source:
            profile["motif_source"] = motif_source
    profile["extra_uniprot_query"] = strip_reviewed_filter(
        str(obj.get("extra_uniprot_query")).strip() if obj.get("extra_uniprot_query") else None
    )
    for key in ("positive_tokens", "negative_tokens", "ec_numbers", "go_terms"):
        profile[key] = _to_list_str(obj.get(key))
    profile["confidence"] = _to_conf(obj.get("confidence"))

    cand = obj.get("candidates")
    out_cands: List[Dict[str, Any]] = []
    if isinstance(cand, list):
        for item in cand:
            if not isinstance(item, dict):
                continue
            pfam_id = _valid_pfam(item.get("pfam_id"))
            interpro_id = _valid_interpro(item.get("interpro_id"))
            if not pfam_id and not interpro_id:
                continue
            out_cands.append(
                {
                    "pfam_id": pfam_id,
                    "interpro_id": interpro_id,
                    "reason": str(item.get("reason") or "").strip(),
                    "confidence": _to_conf(item.get("confidence")),
                }
            )
    if not profile["pfam_id"] and out_cands:
        pfam_cands = sorted(
            (c for c in out_cands if c.get("pfam_id")),
            key=lambda c: float(c.get("confidence") or 0.0),
            reverse=True,
        )
        if pfam_cands:
            profile["pfam_id"] = str(pfam_cands[0].get("pfam_id") or "")
    if not profile["interpro_id"] and out_cands:
        interpro_cands = sorted(
            (c for c in out_cands if c.get("interpro_id")),
            key=lambda c: float(c.get("confidence") or 0.0),
            reverse=True,
        )
        if interpro_cands:
            profile["interpro_id"] = str(interpro_cands[0].get("interpro_id") or "")
    if not out_cands and (profile["pfam_id"] or profile["interpro_id"]):
        out_cands.append(
            {
                "pfam_id": profile["pfam_id"],
                "interpro_id": profile["interpro_id"],
                "reason": "auto-filled from primary mapping",
                "confidence": profile["confidence"],
            }
        )
    profile["candidates"] = out_cands

    rules: List[Dict[str, Any]] = []
    if minimal_patterns:
        rules = _minimal_motif_active_site_rules(minimal_patterns)
    else:
        raw_rules = obj.get("active_site_rules")
    if (not minimal_patterns) and isinstance(raw_rules, list):
        for idx, item in enumerate(raw_rules, start=1):
            norm = _normalize_active_site_rule(item, idx)
            if norm is not None:
                rules.append(norm)
    profile["active_site_rules"] = rules

    policy_raw = (
        "all_required"
        if minimal_patterns
        else str(obj.get("active_site_policy") or "").strip().lower()
    )
    if policy_raw not in ("all_required", "min_required"):
        policy_raw = "all_required"
    profile["active_site_policy"] = policy_raw

    min_hits_raw = None if minimal_patterns else obj.get("active_site_min_hits")
    try:
        min_hits = (
            int(min_hits_raw) if min_hits_raw is not None and str(min_hits_raw).strip() else None
        )
    except (TypeError, ValueError):
        min_hits = None
    if min_hits is not None and min_hits <= 0:
        min_hits = None
    if min_hits is not None and rules:
        min_hits = max(1, min(min_hits, len(rules)))
    if not rules:
        min_hits = None
    profile["active_site_min_hits"] = min_hits

    if query_seq is not None:
        profile = _sanitize_profile_constraints_for_query(profile, query_seq)

    return profile


def apply_grounding_to_profile(
    profile: Dict[str, Any],
    grounding: Dict[str, Any],
    query_seq: Optional[str] = None,
) -> Dict[str, Any]:
    out = dict(profile)
    top_pfams = [x.get("id", "") for x in grounding.get("top_pfams", []) if isinstance(x, dict)]
    top_interpros = [
        x.get("id", "") for x in grounding.get("top_interpros", []) if isinstance(x, dict)
    ]
    top_pairs = [x for x in grounding.get("top_pairs", []) if isinstance(x, dict)]
    top_ecs = [x.get("id", "") for x in grounding.get("top_ec_numbers", []) if isinstance(x, dict)]

    if top_pfams and out.get("pfam_id") not in set(top_pfams):
        out["pfam_id"] = str(top_pfams[0]).upper()
    if top_interpros and out.get("interpro_id") not in set(top_interpros):
        out["interpro_id"] = str(top_interpros[0]).upper()

    if top_pairs:
        best_pair = top_pairs[0]
        bp = str(best_pair.get("pfam_id") or "").upper()
        bi = str(best_pair.get("interpro_id") or "").upper()
        if bp and bi:
            out["pfam_id"] = bp
            out["interpro_id"] = bi

    if (not out.get("ec_numbers")) and top_ecs:
        out["ec_numbers"] = [str(top_ecs[0])]

    added_negative_tokens = _derive_grounding_contamination_negative_tokens(
        enzyme_type=str(out.get("enzyme_type_normalized") or ""),
        grounding=grounding,
    )
    if added_negative_tokens:
        merged_negative_tokens: List[str] = []
        for token in list(out.get("negative_tokens") or []) + added_negative_tokens:
            tok = str(token).strip()
            if tok and tok not in merged_negative_tokens:
                merged_negative_tokens.append(tok)
        out["negative_tokens"] = merged_negative_tokens

    cands = out.get("candidates")
    if not isinstance(cands, list):
        cands = []
    if out.get("pfam_id") and out.get("interpro_id"):
        cands = [
            {
                "pfam_id": out["pfam_id"],
                "interpro_id": out["interpro_id"],
                "reason": "grounded by UniProt reviewed hits",
                "confidence": max(float(out.get("confidence") or 0.0), 0.9),
            }
        ] + [x for x in cands if isinstance(x, dict)]
    out["candidates"] = cands
    out["confidence"] = max(
        float(out.get("confidence") or 0.0),
        0.8 if grounding.get("hit_count", 0) >= 3 else 0.0,
    )
    return normalize_profile(out, query_seq=query_seq)


# Prompt builder + end-to-end profile generator


@lru_cache(maxsize=None)
def _read_prompt_resource(resource_name: str) -> str:
    return resources.files(__package__).joinpath(resource_name).read_text(encoding="utf-8").strip()


def profile_prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    """Return the hash of the exact prompts sent to the profile provider."""

    payload = f"{system_prompt}\n{user_prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def profile_schema_sha256() -> str:
    """Return the hash of the normalized profile response schema."""

    payload = json.dumps(
        PROFILE_JSON_SCHEMA["schema"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_profile_prompts(
    enzyme_type: str,
    query_seq: str,
    grounding: Optional[Dict[str, Any]] = None,
    query_name: Optional[str] = None,
    query_details_text: Optional[str] = None,
) -> Tuple[str, str]:
    sys_prompt = _read_prompt_resource(_PROFILE_SYSTEM_PROMPT_RESOURCE)
    grounding = grounding or {}
    top_pfams = grounding.get("top_pfams") or []
    top_interpros = grounding.get("top_interpros") or []
    top_pairs = grounding.get("top_pairs") or []
    top_ecs = grounding.get("top_ec_numbers") or []
    grounding_block = json.dumps(
        {
            "query": grounding.get("query"),
            "hit_count": grounding.get("hit_count"),
            "top_pfams": top_pfams[:5],
            "top_interpros": top_interpros[:5],
            "top_pairs": top_pairs[:5],
            "top_ec_numbers": top_ecs[:5],
            "sample_entries": grounding.get("sample_entries", [])[:5],
        },
        ensure_ascii=False,
    )
    details = str(query_details_text or "").strip()
    details_block = details if details else "(none provided)"
    qname = str(query_name or "QUERY_INPUT").strip() or "QUERY_INPUT"
    replacements = {
        "__ENZYME_TYPE__": str(enzyme_type),
        "__QUERY_NAME__": qname,
        "__QUERY_SEQUENCE_LENGTH__": str(len(query_seq)),
        "__QUERY_SEQUENCE_PREVIEW__": f"{query_seq[:30]}...{query_seq[-30:]}",
        "__QUERY_SEQUENCE__": query_seq,
        "__QUERY_DETAILS__": details_block,
        "__GROUNDING_BLOCK__": grounding_block,
    }
    user_prompt = _read_prompt_resource(_PROFILE_USER_PROMPT_RESOURCE)
    for marker, value in replacements.items():
        user_prompt = user_prompt.replace(marker, value)
    return sys_prompt, user_prompt


def generate_profile(
    *,
    provider: str,
    base_url: str,
    model: str,
    enzyme_type: str,
    query_seq: str,
    llm_dir: Path,
    query_name: Optional[str] = None,
    query_details_text: Optional[str] = None,
    max_tokens: int = 8192,
    max_attempts: int = 1,
) -> Dict[str, Any]:
    llm_dir.mkdir(parents=True, exist_ok=True)
    grounding = fetch_uniprot_grounding(enzyme_type=enzyme_type, size=25)
    (llm_dir / "profile_grounding_uniprot.json").write_text(
        json.dumps(grounding, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    top_pairs = grounding.get("top_pairs") or []
    _log(
        "uniprot grounding "
        f"hits={grounding.get('hit_count', 0)} "
        f"top_pair={(top_pairs[0] if top_pairs else None)}"
    )

    sys_prompt, user_prompt = build_profile_prompts(
        enzyme_type,
        query_seq,
        grounding=grounding,
        query_name=query_name,
        query_details_text=query_details_text,
    )
    (llm_dir / "profile_prompt.system.txt").write_text(sys_prompt, encoding="utf-8")
    (llm_dir / "profile_prompt.user.txt").write_text(user_prompt, encoding="utf-8")
    prompt_metadata = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "prompt_version": PROFILE_PROMPT_VERSION,
        "response_schema_sha256": profile_schema_sha256(),
        "prompt_sha256": profile_prompt_sha256(sys_prompt, user_prompt),
        "system_prompt_sha256": hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "decoding": {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "retries": 0,
            "max_attempts": max_attempts,
        },
    }
    (llm_dir / "profile_prompt_metadata.json").write_text(
        json.dumps(prompt_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    attempt = 1
    _log(f"profile mapping attempt=1/1 enzyme_type={enzyme_type!r} model={model}")
    raw = llm_chat(
        provider=provider,
        base_url=base_url,
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        json_tool=PROFILE_JSON_TOOL,
        retries=0,
    )
    (llm_dir / f"profile_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
    try:
        profile = _profile_from_llm_raw(raw, grounding=grounding, query_seq=query_seq)
    except Exception as exc:
        err_text = f"{type(exc).__name__}: {exc}"
        (llm_dir / f"profile_attempt_{attempt}.err.txt").write_text(
            err_text,
            encoding="utf-8",
        )
        _log(f"profile mapping parse/validation failed attempt=1/1: {err_text}")
        raise RuntimeError(
            "Failed to generate valid enzyme profile JSON from the single LLM attempt "
            f"(error={err_text})"
        ) from exc

    profile_text = json.dumps(profile, ensure_ascii=False, indent=2)
    (llm_dir / "enzyme_profile.json").write_text(profile_text, encoding="utf-8")
    run_metadata = {
        **prompt_metadata,
        "network_access": True,
        "grounding_sha256": hashlib.sha256(
            (llm_dir / "profile_grounding_uniprot.json").read_bytes()
        ).hexdigest(),
        "raw_response_sha256": hashlib.sha256(
            (llm_dir / f"profile_attempt_{attempt}.txt").read_bytes()
        ).hexdigest(),
        "normalized_profile_sha256": hashlib.sha256(profile_text.encode("utf-8")).hexdigest(),
    }
    (llm_dir / "profile_run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        "profile mapping success "
        f"pfam={profile.get('pfam_id')} interpro={profile.get('interpro_id')} "
        f"confidence={profile.get('confidence')}"
    )
    return profile


def write_mapping_failure_report(
    *,
    llm_dir: Path,
    enzyme_type: str,
    profile: Optional[Dict[str, Any]],
    threshold: float,
    reason: str,
) -> Path:
    """Record profile mapping failure for humans + CLI caller."""

    path = llm_dir / "mapping_failure_report.json"
    payload: Dict[str, Any] = {
        "enzyme_type": enzyme_type,
        "mapping_threshold": float(threshold),
        "reason": reason,
    }
    if isinstance(profile, dict):
        payload["pfam_id"] = profile.get("pfam_id") or ""
        payload["interpro_id"] = profile.get("interpro_id") or ""
        payload["confidence"] = float(profile.get("confidence") or 0.0)
        payload["candidates"] = profile.get("candidates") or []
    llm_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
