#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Importable UniProt/InterPro sequence downloader for profile-guided retrieval.

The profile is supplied as JSON data with ``--profile-json``. New runs execute
this module directly; no Python source is generated or patched at runtime.
Motif and active-site rules are evaluated against the retrieved corpus before
they are admitted as sequence filters.

Output files:
    <outdir>/<output_prefix>.ids.txt
    <outdir>/<output_prefix>.raw.fasta
    <outdir>/<output_prefix>.filtered.fasta
    <outdir>/<output_prefix>.summary.txt

Runtime dependency: ``requests``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from io import StringIO
from typing import List, Tuple, Optional, Dict, Any, Sequence
from urllib.parse import urlsplit, parse_qs

import requests

from enzformer2.profile.profile import minimal_motif_filter_decision, normalize_profile

INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api"
UNIPROT_BASE = "https://rest.uniprot.org"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_accessions_from_ids(path: str, limit: Optional[int] = None) -> List[str]:
    p = os.path.expanduser(str(path or "").strip())
    if not p:
        raise RuntimeError("--input-ids-file is empty")
    if not os.path.exists(p):
        raise RuntimeError(f"--input-ids-file not found: {p}")
    out: List[str] = []
    seen = set()
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not re.fullmatch(r"[A-Za-z0-9]+", line):
                continue
            acc = line.upper()
            if acc in seen:
                continue
            seen.add(acc)
            out.append(acc)
            if limit is not None and len(out) >= int(limit):
                break
    return out


def _parse_cursor_from_url(u: str) -> Optional[str]:
    try:
        vals = parse_qs(urlsplit(u).query).get("cursor")
        return vals[0] if vals else None
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def _extract_results(payload) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("results", "proteins", "items"):
            if isinstance(payload.get(k), list):
                return payload[k]
    return []


def _next_cursor(resp: requests.Response, payload) -> Optional[str]:
    try:
        nxt = resp.links.get("next", {}).get("url")
        if nxt:
            cur = _parse_cursor_from_url(nxt)
            if cur:
                return cur
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        pass
    for h in ("X-Next-Page-Cursor", "X-Next-Page", "X-Next-Cursor"):
        v = resp.headers.get(h)
        if v:
            return v.strip()
    try:
        if isinstance(payload, dict):
            cand = payload.get("next")
            if isinstance(cand, str):
                cur = _parse_cursor_from_url(cand)
                if cur:
                    return cur
            page = payload.get("page") or payload.get("pages") or {}
            if isinstance(page, dict):
                cand = page.get("next") or page.get("nextUrl") or page.get("nextURL")
                if isinstance(cand, str):
                    cur = _parse_cursor_from_url(cand)
                    if cur:
                        return cur
            links = payload.get("links") or {}
            if isinstance(links, dict):
                cand = links.get("next")
                if isinstance(cand, str):
                    cur = _parse_cursor_from_url(cand)
                    if cur:
                        return cur
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        pass
    return None


def interpro_collect_uniprot_accessions(
    entry_db: str,
    entry_id: str,
    page_size: int = 200,
    sleep: float = 0.2,
    limit: Optional[int] = None,
    timeout: int = 60,
    max_retries: int = 5,
    checkpoint_path: Optional[str] = None,
) -> List[str]:
    """InterPro reverse mapping: /api/protein/uniprot/entry/{pfam|interpro}/{ACC}/"""

    base_url = f"{INTERPRO_BASE}/protein/uniprot/entry/{entry_db}/{entry_id}/"
    params = {"page_size": page_size}
    headers = {"Accept": "application/json"}
    accs: List[str] = []
    seen = set()
    cursor: Optional[str] = None
    page = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        if checkpoint.get("entry_db") != entry_db or checkpoint.get("entry_id") != entry_id:
            raise RuntimeError(f"InterPro checkpoint source mismatch: {checkpoint_path}")
        accs = [str(acc).upper() for acc in checkpoint.get("accessions", [])]
        seen = set(accs)
        cursor = checkpoint.get("next_cursor")
        page = int(checkpoint.get("pages_completed", 0))
        if checkpoint.get("complete"):
            log(f"[InterPro] checkpoint complete: {entry_db}:{entry_id} ({len(accs)} accessions)")
            return accs[:limit] if limit else accs
        log(
            f"[InterPro] resuming checkpoint: {entry_db}:{entry_id} "
            f"page={page} total={len(accs)} cursor={cursor!r}"
        )

    while True:
        req = dict(params)
        if cursor:
            req["cursor"] = cursor

        tries = 0
        last_error_status: Optional[int] = None
        while True:
            tries += 1
            try:
                r = requests.get(base_url, params=req, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                last_error_status = None
                log(f"[InterPro] network error: {e} (try {tries}/{max_retries})")
                if tries >= max_retries:
                    raise RuntimeError(
                        f"InterPro GET failed after {max_retries} retries for cursor={cursor!r} "
                        f"(last network error: {e})"
                    )
                time.sleep(sleep)
                continue

            if r.status_code in (408, 429, 500, 502, 503, 504):
                last_error_status = r.status_code
                delay = float(r.headers.get("Retry-After", "1.0"))
                log(f"[InterPro] {r.status_code}; backoff {delay}s (try {tries}/{max_retries})")
                time.sleep(delay)
                if tries < max_retries:
                    continue
                raise RuntimeError(
                    f"InterPro GET failed after {max_retries} retries for cursor={cursor!r}, "
                    f"status={last_error_status}"
                )

            if r.status_code != 200:
                raise RuntimeError(f"InterPro GET failed {r.status_code} {r.url}\n{r.text[:400]}")

            try:
                payload = r.json()
            except ValueError as e:
                raise RuntimeError(f"InterPro JSON parse error on page {page + 1}: {e}")

            collected = 0
            for item in _extract_results(payload):
                acc = None
                if isinstance(item, dict):
                    meta = item.get("metadata") or {}
                    acc = meta.get("accession") or item.get("accession")
                if acc and acc not in seen:
                    seen.add(acc)
                    accs.append(acc)
                    collected += 1
                if limit and len(accs) >= limit:
                    log(f"[InterPro] reached limit={limit}")
                    return accs

            page += 1
            log(
                f"[InterPro] {entry_db}:{entry_id} page {page:>4} -> +{collected:>4} (total {len(accs)})"
            )
            nxt = _next_cursor(r, payload)
            if checkpoint_path:
                _write_json_atomic(
                    checkpoint_path,
                    {
                        "entry_db": entry_db,
                        "entry_id": entry_id,
                        "pages_completed": page,
                        "next_cursor": nxt,
                        "complete": not bool(nxt),
                        "accessions": accs,
                    },
                )
            if not nxt:
                cursor = None
                break
            cursor = nxt
            break

        if not cursor:
            break
        if sleep:
            time.sleep(sleep)

    return accs


def uniprot_search_accessions(
    query: Optional[str],
    page_size: int = 500,
    sleep: float = 0.2,
    limit: Optional[int] = None,
    reviewed_only: bool = False,
    timeout: int = 60,
    max_retries: int = 5,
) -> List[str]:
    """Collect accessions from a UniProt search query.

    This is an accession source. Do not use it as an AND filter when later
    fetching explicit Pfam/InterPro accessions; that would discard broad
    homologs that are valid family members but do not match target-organism
    query hints.
    """

    q = normalize_uniprot_extra_query(query)
    if not q:
        return []
    if reviewed_only:
        q = f"({q}) AND reviewed:true"

    url = f"{UNIPROT_BASE}/uniprotkb/search"
    params = {
        "query": q,
        "fields": "accession",
        "format": "tsv",
        "size": str(max(1, int(page_size))),
    }
    accs: List[str] = []
    seen = set()
    cursor: Optional[str] = None
    page = 0

    while True:
        req = dict(params)
        if cursor:
            req["cursor"] = cursor

        tries = 0
        while True:
            tries += 1
            try:
                r = requests.get(url, params=req, timeout=timeout)
            except requests.RequestException as e:
                log(f"[UniProt search] network error: {e} (try {tries}/{max_retries})")
                if tries >= max_retries:
                    raise RuntimeError(
                        f"UniProt search failed after {max_retries} retries "
                        f"for cursor={cursor!r}: {e}"
                    )
                time.sleep(sleep)
                continue

            if r.status_code in (408, 429, 500, 502, 503, 504):
                delay = float(r.headers.get("Retry-After", "1.0"))
                log(
                    f"[UniProt search] {r.status_code}; backoff {delay}s (try {tries}/{max_retries})"
                )
                time.sleep(delay)
                if tries < max_retries:
                    continue
                raise RuntimeError(
                    f"UniProt search failed after {max_retries} retries "
                    f"for cursor={cursor!r}, status={r.status_code}"
                )

            if r.status_code != 200:
                body = (r.text or "").strip()
                raise RuntimeError(f"UniProt search failed {r.status_code} {r.url}\n{body[:400]}")

            lines = [ln for ln in (r.text or "").splitlines() if ln.strip()]
            collected = 0
            if lines:
                for line in lines[1:]:
                    acc = line.split("\t", 1)[0].strip().upper()
                    if not re.fullmatch(r"[A-Z0-9]+", acc):
                        continue
                    if acc in seen:
                        continue
                    seen.add(acc)
                    accs.append(acc)
                    collected += 1
                    if limit and len(accs) >= limit:
                        log(f"[UniProt search] reached limit={limit}")
                        return accs

            page += 1
            log(f"[UniProt search] page {page:>4} -> +{collected:>4} (total {len(accs)})")
            cursor = _next_cursor(r, {})
            break

        if not cursor:
            break
        if sleep:
            time.sleep(sleep)

    return accs


# Budget for GET URL length before falling back to POST. UniProt's REST gateway
# rejects URIs longer than a few KB with HTTP 414, and even shorter ones can
# trigger upstream proxy limits. 6000 bytes is comfortably under both. Above
# this, the same params are sent via application/x-www-form-urlencoded body.
_UNIPROT_GET_URL_BUDGET = 6000


def _request_length(url: str, params: Dict[str, str]) -> int:
    """Estimate ``GET`` URL length so callers can fall back to ``POST``."""
    from urllib.parse import urlencode

    return len(url) + 1 + len(urlencode(params, doseq=True))


def safe_uniprot_stream_batch(
    accessions: List[str],
    batch_size: int = 80,
    reviewed_only: bool = False,
    extra_query: Optional[str] = None,
    timeout: int = 120,
    sleep: float = 0.2,
    fmt: str = "fasta",
    fields: Optional[str] = None,
    max_retries: int = 5,
) -> str:
    """Retrieve data via UniProt /uniprotkb/stream with backoff & batch-shrinking."""

    if not accessions:
        return ""
    extra_query = normalize_uniprot_extra_query(extra_query)
    url = f"{UNIPROT_BASE}/uniprotkb/stream"
    out: List[str] = []
    i, size = 0, max(1, batch_size)

    while i < len(accessions):
        window = accessions[i : i + size]
        q = " OR ".join(f"accession:{a}" for a in window)
        q = f"({q})"
        if reviewed_only:
            q = f"{q} AND reviewed:true"
        if extra_query:
            q = f"{q} AND ({extra_query})"
        params = {"query": q, "format": fmt, "compressed": "false"}
        if fmt == "tsv" and fields:
            params["fields"] = fields

        tries = 0
        last_error_status: Optional[int] = None

        while True:
            tries += 1
            # Switch to POST when the encoded query is large enough to risk
            # exceeding the UniProt URI length cap (HTTP 414). The /uniprotkb/stream
            # endpoint accepts the same params via application/x-www-form-urlencoded
            # body and that side-steps batch-shrink loops at large batch sizes.
            try:
                if _request_length(url, params) > _UNIPROT_GET_URL_BUDGET:
                    r = requests.post(url, data=params, timeout=timeout)
                else:
                    r = requests.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                # First, retry up to ``max_retries`` at the current batch size with
                # exponential backoff. Transient slowdowns / 120s read-timeouts often
                # recover on the very next call, so shrinking the batch immediately
                # is wasteful — it doubles the number of round-trips and quickly
                # collapses progress to size=1 even when UniProt is fine.
                if tries < max_retries:
                    delay = min(60.0, max(sleep, 1.0) * (2 ** (tries - 1)))
                    log(
                        f"[UniProt] net error; retry {tries}/{max_retries} @index {i}, "
                        f"size {size}: {e}; sleep {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                # Exhausted same-size retries → shrink (and reset retry budget) if
                # batches > 1 are still possible, else give up on this position.
                if size > 1:
                    old = size
                    size = max(1, size // 2)
                    log(
                        f"[UniProt] net error; shrink batch {old} -> {size} @index {i} "
                        f"after {max_retries} retries: {e}"
                    )
                    time.sleep(sleep)
                    break
                raise RuntimeError(
                    f"UniProt stream failed after {max_retries} retries @index {i}, "
                    f"size {size}, last network error: {e}"
                )

            body = (r.text or "").strip()

            # HTTP 200 with an empty body means "no hits in this batch" — valid, not an error.
            if r.status_code == 200:
                if body:
                    out.append(body)
                else:
                    log(f"[UniProt] 200 but empty body @index {i}, size {size} -> 0-hit batch")
                i += size
                if sleep:
                    time.sleep(sleep)
                break

            if r.status_code in (400, 414):
                if size > 1:
                    old = size
                    size = max(1, size // 2)
                    log(f"[UniProt] {r.status_code} (size {old}) -> {size} @index {i}")
                    time.sleep(sleep)
                    break
                raise RuntimeError(
                    f"UniProt stream failed for {accessions[i]} (status {r.status_code}). "
                    f"{r.url}\n{body[:300]}"
                )

            if r.status_code in (408, 429, 500, 502, 503, 504):
                last_error_status = r.status_code
                delay = float(r.headers.get("Retry-After", "1.0"))
                log(
                    f"[UniProt] {r.status_code}; backoff {delay}s @index {i}, size {size} (try {tries}/{max_retries})"
                )
                time.sleep(delay)
                if tries < max_retries:
                    continue
                raise RuntimeError(
                    f"UniProt stream failed after {max_retries} retries @index {i}, "
                    f"size {size}, status={last_error_status}"
                )

            raise RuntimeError(
                f"UniProt stream unexpected {r.status_code} @index {i}, size {size}\n"
                f"{r.url}\n{body[:300]}"
            )

    return "\n".join(out)


def normalize_uniprot_extra_query(extra_query: Optional[str]) -> Optional[str]:
    q = (extra_query or "").strip()
    if not q:
        return None
    # UniProt REST uses organism_name/organism_id; "organism:" is rejected.
    q = re.sub(r"(?i)(?<![A-Za-z0-9_])organism\s*:", "organism_name:", q)
    # LLMs commonly emit NCBI-style ``term[Family]`` syntax, which UniProt
    # rejects with HTTP 400. Preserve the intent using UniProt's protein field.
    family_match = re.fullmatch(r"(?is)\s*(.+?)\s*\[family\]\s*", q)
    if family_match:
        family_term = family_match.group(1).strip().strip('"')
        q = f'protein_name:"{family_term}"'
    q = re.sub(r"(?i)\breviewed\s*:\s*(?:true|false)\b", "", q)
    q = re.sub(r"\(\s*\)", "", q)
    q = re.sub(r"(?i)\b(AND|OR)\s+\)", ")", q)
    q = re.sub(r"(?i)\(\s*(AND|OR)\b", "(", q)
    q = re.sub(r"(?i)\b(AND|OR)\s+(AND|OR)\b", r"\1", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"(?i)^(AND|OR)\b\s*", "", q).strip()
    q = re.sub(r"(?i)\s*\b(AND|OR)$", "", q).strip()
    return q or None


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    recs: List[Tuple[str, str]] = []
    h, buf = None, []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(buf)))
            h = line[1:].strip()
            buf = []
        else:
            stripped = line.strip()
            if stripped == ",":
                continue
            buf.append(stripped)
    if h is not None:
        recs.append((h, "".join(buf)))
    return recs


def write_fasta(path: str, recs: List[Tuple[str, str]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for h, s in recs:
            f.write(f">{h}\n")
            for i in range(0, len(s), 60):
                f.write(s[i : i + 60] + "\n")


def header_has_fragment_flag(h: str) -> bool:
    hl = h.lower()
    return ("fragment" in hl) or ("partial" in hl) or ("truncated" in hl)


def extract_accession_from_header(h: str) -> Optional[str]:
    m = re.match(r"^(?:sp|tr)\|([^|]+)\|", h)
    if m:
        return m.group(1)
    t0 = h.split()[0] if h.split() else ""
    if "|" in t0:
        parts = t0.split("|")
        for p in parts:
            if re.fullmatch(r"[AOPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5}", p):
                return p
    t = h.split()[0] if h.split() else ""
    if re.fullmatch(r"[A-Z0-9]{6,10}", t):
        return t
    return None


_UNIPROT_FIELDS = ",".join(
    [
        "accession",
        "protein_name",
        "ec",
        "go_f",
        "go_id",
        "keyword",
        "length",
        "organism_name",
    ]
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def parse_uniprot_tsv(tsv_text: str) -> Dict[str, Dict[str, str]]:
    if not tsv_text.strip():
        return {}
    reader = csv.reader(StringIO(tsv_text), delimiter="\t")
    rows = list(reader)
    if not rows:
        return {}

    header = rows[0]
    norm_map = {i: _norm(h) for i, h in enumerate(header)}

    def find_col(candidates: List[str]) -> Optional[int]:
        candn = [_norm(c) for c in candidates]
        for i, n in norm_map.items():
            if n in candn:
                return i
        return None

    idx_acc = find_col(["accession", "entry"])
    idx_pn = find_col(["proteinnames", "proteinname"])
    idx_ec = find_col(["ecnumber", "ec"])
    idx_gof = find_col(["geneontologymolecularfunction", "gof"])
    idx_goi = find_col(["geneontologyids", "goid", "goids"])
    idx_kw = find_col(["keyword", "keywords"])
    idx_len = find_col(["length"])
    idx_org = find_col(["organism", "organismname"])

    out: Dict[str, Dict[str, str]] = {}
    for r in rows[1:]:
        if not r or idx_acc is None or idx_acc >= len(r):
            continue
        if _norm(r[idx_acc]) in ("accession", "entry"):
            continue
        acc = r[idx_acc]
        if not acc:
            continue

        def get(idx: Optional[int]) -> str:
            return (r[idx] if (idx is not None and idx < len(r)) else "").strip()

        row = {
            "accession": acc,
            "protein_name": get(idx_pn),
            "ec": get(idx_ec),
            "go_f": get(idx_gof),
            "go_id": get(idx_goi),
            "keywords": get(idx_kw),
            "length": get(idx_len),
            "organism_name": get(idx_org),
        }
        if acc in out:
            for k, v in row.items():
                if not v:
                    continue
                if not out[acc].get(k):
                    out[acc][k] = v
                elif v not in out[acc][k]:
                    out[acc][k] = out[acc][k] + " ; " + v
        else:
            out[acc] = row
    return out


def fetch_uniprot_metadata(
    accessions: List[str],
    batch_size: int = 200,
    reviewed_only: bool = False,
    extra_query: Optional[str] = None,
    timeout: int = 120,
    sleep: float = 0.2,
) -> Dict[str, Dict[str, str]]:
    tsv = safe_uniprot_stream_batch(
        accessions,
        batch_size=batch_size,
        reviewed_only=reviewed_only,
        extra_query=extra_query,
        timeout=timeout,
        sleep=sleep,
        fmt="tsv",
        fields=_UNIPROT_FIELDS,
    )
    return parse_uniprot_tsv(tsv)


# ---------- Profile defaults and loading ----------

_POSITIVE_TOKENS = [
    "enzyme",
]
_NEGATIVE_TOKENS = [
    "regulatory protein",
]
_GO_TERMS = []
_EC_NUMBERS = []

DEFAULT_PROFILE = {
    "positive_tokens": _POSITIVE_TOKENS,
    "negative_tokens": _NEGATIVE_TOKENS,
    "go_terms": _GO_TERMS,
    "ec_numbers": _EC_NUMBERS,
    "motif_regex": None,
    "motif_source": None,
    "minimal_conserved_motif_source": None,
    "minimal_conserved_motif_patterns": [],
    "minimal_conserved_motif_rationale": None,
    "active_site_rules": [],
    "active_site_policy": "all_required",
    "active_site_min_hits": None,
}


def load_profile_json(path: str | os.PathLike[str] | None) -> Dict[str, Any]:
    """Load a normalized profile without changing the downloader source.

    A missing path selects conservative defaults for standalone use. When a
    path is supplied it must contain a JSON object; malformed profile data is
    an error rather than an implicit fallback to uniform filtering.
    """

    if path is None:
        return dict(DEFAULT_PROFILE)
    profile_path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(profile_path):
        raise FileNotFoundError(f"Profile JSON not found: {profile_path}")
    with open(profile_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Profile JSON must contain an object: {profile_path}")
    profile = dict(DEFAULT_PROFILE)
    profile.update(payload)
    for key in ("positive_tokens", "negative_tokens", "go_terms", "ec_numbers"):
        value = profile.get(key)
        if value is None:
            profile[key] = []
        elif isinstance(value, str):
            profile[key] = [value]
        elif isinstance(value, (list, tuple)):
            profile[key] = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise ValueError(f"Profile field {key!r} must be a list or string.")
    rules = profile.get("active_site_rules")
    if rules is None:
        profile["active_site_rules"] = []
    elif not isinstance(rules, list):
        raise ValueError("Profile field 'active_site_rules' must be a list.")
    return normalize_profile(profile)


def has_token(text: str, tokens: List[str]) -> bool:
    tl = (text or "").lower()
    return any((str(tok or "").lower() in tl) for tok in tokens)


def is_positive_metadata(
    m: Dict[str, str],
    positive_tokens: List[str],
    ec_numbers: List[str],
    go_terms: List[str],
) -> bool:
    text = " ".join(
        [
            m.get("protein_name", ""),
            m.get("ec", ""),
            m.get("go_f", ""),
            m.get("go_id", ""),
            m.get("keywords", ""),
        ]
    ).lower()
    return (
        has_token(text, positive_tokens)
        or any(str(ec).lower() in text for ec in ec_numbers)
        or any(str(go).lower() in text for go in go_terms)
    )


def is_negative_metadata(m: Dict[str, str], negative_tokens: List[str]) -> bool:
    text = " ".join(
        [
            m.get("protein_name", ""),
            m.get("go_f", ""),
            m.get("go_id", ""),
            m.get("keywords", ""),
        ]
    ).lower()
    return has_token(text, negative_tokens)


def _normalize_motif_pattern(motif_regex: str) -> str:
    pat = str(motif_regex or "").strip()
    if not pat:
        return ""
    if re.search(r"[.\[\](){}|+*?^$\\]", pat):
        return pat
    return re.escape(pat).replace("X", ".").replace("x", ".")


def has_motif(seq: str, motif_regex: str) -> bool:
    pat = _normalize_motif_pattern(motif_regex)
    if not pat:
        return True
    try:
        return re.search(pat, seq, flags=re.IGNORECASE) is not None
    except re.error as exc:
        raise ValueError(f"Invalid motif regex: {motif_regex!r}") from exc


def _to_output_prefix(raw: str) -> str:
    s = re.sub(r"[^a-z0-9_.-]+", "_", (raw or "").strip().lower()).strip("._-")
    return s or "enzyme_union"


def _parse_active_site_rules_json(raw: Optional[str]) -> List[Dict[str, Any]]:
    txt = (raw or "").strip()
    if not txt:
        return []
    obj = json.loads(txt)
    if not isinstance(obj, list):
        raise ValueError("--active-site-rules-json must be a JSON list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(obj, start=1):
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or f"rule_{i}").strip() or f"rule_{i}"
        regex_raw = item.get("regex")
        regex = str(regex_raw).strip() if regex_raw is not None else ""
        regex = regex or None
        pos_raw = item.get("position_1based")
        try:
            pos = int(pos_raw) if pos_raw is not None and str(pos_raw).strip() else None
        except (TypeError, ValueError):
            pos = None
        if pos is not None and pos <= 0:
            pos = None
        aa_raw = item.get("allowed_aas")
        allowed_aas: List[str] = []
        if isinstance(aa_raw, list):
            for aa in aa_raw:
                tok = str(aa).strip().upper()
                if re.fullmatch(r"[A-Z]", tok) and tok not in allowed_aas:
                    allowed_aas.append(tok)
        tol_raw = item.get("tolerance")
        try:
            tol = int(tol_raw) if tol_raw is not None and str(tol_raw).strip() else 0
        except (TypeError, ValueError):
            tol = 0
        if tol < 0:
            tol = 0
        required = bool(item.get("required", True))
        if regex is None and pos is None:
            continue
        out.append(
            {
                "id": rid,
                "regex": regex,
                "position_1based": pos,
                "allowed_aas": allowed_aas or None,
                "tolerance": tol,
                "required": required,
            }
        )
    return out


def filter_basic(
    recs: List[Tuple[str, str]],
    min_len: Optional[int],
    max_len: Optional[int],
    drop_fragments: bool,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    kept: List[Tuple[str, str]] = []
    cnt = {"total": len(recs), "len_filtered": 0, "fragments_removed": 0, "kept": 0}
    for h, s in recs:
        L = len(s)
        if (min_len is not None and L < min_len) or (max_len is not None and L > max_len):
            cnt["len_filtered"] += 1
            continue
        if drop_fragments and header_has_fragment_flag(h):
            cnt["fragments_removed"] += 1
            continue
        kept.append((h, s))
    cnt["kept"] = len(kept)
    return kept, cnt


def filter_metadata(
    recs: List[Tuple[str, str]],
    metadata: Dict[str, Dict[str, str]],
    positive_tokens: List[str],
    negative_tokens: List[str],
    ec_numbers: List[str],
    go_terms: List[str],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    out: List[Tuple[str, str]] = []
    stats = {
        "total_after_basic_filters": len(recs),
        "meta_positive_true": 0,
        "meta_negative_true": 0,
        "fallback_negative_only_used": 0,
        "kept_after_metadata_filter": 0,
    }
    ann: List[Tuple[Tuple[str, str], bool, bool]] = []
    for h, s in recs:
        acc = extract_accession_from_header(h) or ""
        m = metadata.get(acc, {})
        pos_ok = is_positive_metadata(m, positive_tokens, ec_numbers, go_terms)
        neg_ok = is_negative_metadata(m, negative_tokens)
        if pos_ok:
            stats["meta_positive_true"] += 1
        if neg_ok:
            stats["meta_negative_true"] += 1
        ann.append(((h, s), pos_ok, neg_ok))

    use_fallback = stats["meta_positive_true"] == 0
    if use_fallback:
        stats["fallback_negative_only_used"] = 1

    for (h, s), pos_ok, neg_ok in ann:
        keep = (not neg_ok) if use_fallback else (pos_ok and (not neg_ok))
        if keep:
            out.append((h, s))

    stats["kept_after_metadata_filter"] = len(out)
    return out, stats


def filter_motif(
    recs: List[Tuple[str, str]],
    motif_regex: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    out: List[Tuple[str, str]] = []
    stats = {
        "total_after_metadata_filter": len(recs),
        "motif_ok": 0,
        "motif_reject": 0,
        "kept_after_motif_filter": 0,
    }
    for h, s in recs:
        ok = has_motif(s, motif_regex)
        if ok:
            stats["motif_ok"] += 1
            out.append((h, s))
        else:
            stats["motif_reject"] += 1
    stats["kept_after_motif_filter"] = len(out)
    return out, stats


def _match_position_rule(
    seq: str, pos_1based: int, allowed_aas: Optional[List[str]], tolerance: int
) -> bool:
    L = len(seq)
    center = int(pos_1based)
    tol = max(0, int(tolerance))
    for p in range(center - tol, center + tol + 1):
        idx = p - 1
        if idx < 0 or idx >= L:
            continue
        aa = seq[idx].upper()
        if not allowed_aas:
            return True
        if aa in allowed_aas:
            return True
    return False


def _match_active_site_rule(seq: str, rule: Dict[str, Any]) -> bool:
    regex = rule.get("regex")
    pos = rule.get("position_1based")
    allowed_aas = rule.get("allowed_aas")
    tolerance = int(rule.get("tolerance") or 0)

    regex_ok = True
    if regex:
        try:
            regex_ok = re.search(str(regex), seq, flags=re.IGNORECASE) is not None
        except re.error:
            regex_ok = False

    pos_ok = True
    if pos is not None:
        pos_ok = _match_position_rule(
            seq=seq,
            pos_1based=int(pos),
            allowed_aas=(allowed_aas if isinstance(allowed_aas, list) else None),
            tolerance=tolerance,
        )

    return bool(regex_ok and pos_ok)


def filter_active_site(
    recs: List[Tuple[str, str]],
    rules: List[Dict[str, Any]],
    policy: str,
    min_hits: Optional[int],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    stats = {
        "total_after_motif_filter": len(recs),
        "active_site_rules_total": len(rules),
        "active_site_eval_records": 0,
        "active_site_reject": 0,
        "kept_after_active_site_filter": 0,
        "active_site_passthrough": 0,
    }
    if not rules:
        stats["active_site_passthrough"] = 1
        stats["kept_after_active_site_filter"] = len(recs)
        return list(recs), stats

    required_rules = [r for r in rules if bool(r.get("required", True))]
    eval_rules = required_rules or rules
    if policy not in ("all_required", "min_required"):
        policy = "all_required"
    if policy == "min_required":
        threshold = int(min_hits) if min_hits is not None else 1
        threshold = max(1, min(threshold, len(eval_rules)))
    else:
        threshold = len(eval_rules)

    out: List[Tuple[str, str]] = []
    for h, s in recs:
        stats["active_site_eval_records"] += 1
        hit_count = 0
        for rule in eval_rules:
            if _match_active_site_rule(s, rule):
                hit_count += 1
        if hit_count >= threshold:
            out.append((h, s))
        else:
            stats["active_site_reject"] += 1

    stats["kept_after_active_site_filter"] = len(out)
    return out, stats


def _profile_motif_patterns(profile: Dict[str, Any]) -> List[str]:
    value = profile.get("minimal_conserved_motif_patterns")
    if isinstance(value, list):
        patterns = [str(item).strip() for item in value if str(item).strip()]
        if patterns:
            return patterns
    patterns = [
        str(rule.get("regex") or "").strip()
        for rule in (profile.get("active_site_rules") or [])
        if isinstance(rule, dict) and str(rule.get("regex") or "").strip()
    ]
    if patterns:
        return patterns
    motif_regex = str(profile.get("motif_regex") or "").strip()
    return [motif_regex] if motif_regex else []


def filter_profile_motifs(
    recs: List[Tuple[str, str]],
    *,
    profile: Dict[str, Any],
    force: bool = False,
) -> Tuple[
    List[Tuple[str, str]],
    Dict[str, int],
    Dict[str, int],
    Dict[str, Any],
]:
    candidate_motif, candidate_motif_stats = filter_motif(
        recs,
        motif_regex=str(profile.get("motif_regex") or ""),
    )
    candidate_final, candidate_active_stats = filter_active_site(
        candidate_motif,
        rules=list(profile.get("active_site_rules") or []),
        policy=str(profile.get("active_site_policy") or "all_required"),
        min_hits=profile.get("active_site_min_hits"),
    )
    unique_sequences = len(
        {re.sub(r"\s+", "", sequence).upper() for _, sequence in candidate_final}
    )
    patterns = _profile_motif_patterns(profile)
    decision = minimal_motif_filter_decision(
        patterns,
        unique_sequences_after_filter=unique_sequences,
    )
    decision["candidate_records"] = len(candidate_final)
    if force and (profile.get("motif_regex") or profile.get("active_site_rules")):
        decision["applied"] = True
        decision["reason"] = "explicit"
    if decision["applied"]:
        return candidate_final, candidate_motif_stats, candidate_active_stats, decision
    motif_kept, motif_stats = filter_motif(recs, motif_regex="")
    final_kept, active_site_stats = filter_active_site(
        motif_kept,
        rules=[],
        policy="all_required",
        min_hits=None,
    )
    return final_kept, motif_stats, active_site_stats, decision


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Retrieve and filter enzyme sequences from InterPro/Pfam and UniProt "
            "using an explicit profile and retrieval configuration."
        )
    )
    ap.add_argument("--pfam", default=None, help="Pfam accession (example: PF00000)")
    ap.add_argument("--interpro", default=None, help="InterPro accession (example: IPR000000)")
    ap.add_argument("--skip-pfam", action="store_true", help="Skip fetching from Pfam source")
    ap.add_argument(
        "--skip-interpro", action="store_true", help="Skip fetching from InterPro source"
    )
    ap.add_argument("--outdir", default="out_enzyme_union", help="Output directory")
    ap.add_argument(
        "--output-prefix",
        default="enzyme_union",
        help="Output file prefix (default enzyme_union)",
    )
    ap.add_argument("--page-size", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument(
        "--input-ids-file",
        default=None,
        help="Optional line-based accession list. If set, skip Pfam/InterPro collection.",
    )
    ap.add_argument(
        "--profile-json",
        default=None,
        help="Normalized enzyme profile JSON used for metadata and optional motif filtering.",
    )

    ap.add_argument(
        "--preset",
        choices=["soluble-ecoli", "domain-only", "broad", "none"],
        default="soluble-ecoli",
        help="Length preset: soluble-ecoli=280..650 (default), domain-only=300..480, broad=200..1200, none=explicit",
    )
    ap.add_argument(
        "--min-len", type=int, default=None, help="Min sequence length (override preset)"
    )
    ap.add_argument(
        "--max-len", type=int, default=None, help="Max sequence length (override preset)"
    )
    ap.add_argument(
        "--keep-fragments",
        action="store_true",
        help="Keep headers containing fragment/partial/truncated",
    )

    ap.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Swiss-Prot only (default: include TrEMBL too)",
    )
    ap.add_argument(
        "--extra-query",
        default=None,
        help='Additional UniProt accession search source (e.g. "go:0003824 OR ec:1.1.1.1")',
    )

    ap.add_argument(
        "--enable-active-site-filter",
        action="store_true",
        help=("Apply profile motif/active-site filtering without automatic admission checks."),
    )
    ap.add_argument(
        "--motif-regex",
        default=None,
        help="Explicit regex for motif filtering. Empty string disables motif filtering.",
    )
    ap.add_argument(
        "--active-site-policy",
        choices=["all_required", "min_required"],
        default=None,
        help="Active-site rule decision policy",
    )
    ap.add_argument(
        "--active-site-min-hits",
        type=int,
        default=None,
        help="Minimum required hits when --active-site-policy=min_required",
    )
    ap.add_argument(
        "--active-site-rules-json",
        default=None,
        help="JSON list for active-site rules",
    )
    ap.add_argument(
        "--skip-active-site-filter",
        action="store_true",
        help="Skip profile motif and active-site filtering",
    )

    ap.add_argument("--skip-filtering", action="store_true", help="Skip filtering entirely")

    args = ap.parse_args(list(argv) if argv is not None else None)
    adaptive_profile = load_profile_json(args.profile_json)
    args.extra_query = normalize_uniprot_extra_query(args.extra_query)
    pfam_id = str(args.pfam or "").strip().upper() or None
    interpro_id = str(args.interpro or "").strip().upper() or None

    preset_min, preset_max = None, None
    if args.preset == "soluble-ecoli":
        preset_min, preset_max = 280, 650
    elif args.preset == "domain-only":
        preset_min, preset_max = 300, 480
    elif args.preset == "broad":
        preset_min, preset_max = 200, 1200
    elif args.preset == "none":
        preset_min, preset_max = None, None

    min_len = args.min_len if args.min_len is not None else preset_min
    max_len = args.max_len if args.max_len is not None else preset_max
    reviewed_only = args.reviewed_only
    drop_fragments = not args.keep_fragments

    ensure_dir(args.outdir)

    all_accs: List[str] = []
    per_source_counts: Dict[str, int] = {}
    source_mode = "pfam_interpro"
    ids_input_path = None
    if args.input_ids_file:
        source_mode = "ids_file"
        ids_input_path = os.path.abspath(os.path.expanduser(str(args.input_ids_file)))
        log(f"loading accessions from ids file: {ids_input_path}")
        accs_ids = load_accessions_from_ids(ids_input_path, limit=args.limit)
        per_source_counts["ids_file"] = len(accs_ids)
        all_accs.extend(accs_ids)
    else:
        enabled_sources = 0
        if not args.skip_pfam:
            if not pfam_id:
                raise RuntimeError(
                    "Missing required --pfam (or use --skip-pfam / --input-ids-file)."
                )
            enabled_sources += 1
        if not args.skip_interpro:
            if not interpro_id:
                raise RuntimeError(
                    "Missing required --interpro (or use --skip-interpro / --input-ids-file)."
                )
            enabled_sources += 1
        if enabled_sources == 0:
            raise RuntimeError(
                "No accession source selected. Use --input-ids-file OR enable at least one of "
                "--pfam/--interpro."
            )

        if not args.skip_pfam:
            log(f"collecting accessions from Pfam:{pfam_id} via InterPro")
            accs_pfam = interpro_collect_uniprot_accessions(
                "pfam",
                pfam_id,
                page_size=args.page_size,
                sleep=args.sleep,
                limit=args.limit,
                checkpoint_path=os.path.join(
                    args.outdir, f".interpro_pfam_{pfam_id}.checkpoint.json"
                ),
            )
            per_source_counts[f"pfam_{pfam_id}"] = len(accs_pfam)
            all_accs.extend(accs_pfam)

        if not args.skip_interpro:
            log(f"collecting accessions from InterPro:{interpro_id}")
            accs_ipr = interpro_collect_uniprot_accessions(
                "interpro",
                interpro_id,
                page_size=args.page_size,
                sleep=args.sleep,
                limit=args.limit,
                checkpoint_path=os.path.join(
                    args.outdir, f".interpro_interpro_{interpro_id}.checkpoint.json"
                ),
            )
            per_source_counts[f"interpro_{interpro_id}"] = len(accs_ipr)
            all_accs.extend(accs_ipr)

        if args.extra_query:
            log("collecting accessions from the additional UniProt query")
            accs_extra = uniprot_search_accessions(
                args.extra_query,
                page_size=max(1, args.page_size),
                sleep=args.sleep,
                limit=args.limit,
                reviewed_only=reviewed_only,
            )
            per_source_counts["uniprot_extra_query"] = len(accs_extra)
            all_accs.extend(accs_extra)

    output_prefix = _to_output_prefix(args.output_prefix)
    raw_path = os.path.join(args.outdir, f"{output_prefix}.raw.fasta")
    filt_path = os.path.join(args.outdir, f"{output_prefix}.filtered.fasta")
    ids_path = os.path.join(args.outdir, f"{output_prefix}.ids.txt")
    summary_path = os.path.join(args.outdir, f"{output_prefix}.summary.txt")

    dedup_accs = list(dict.fromkeys(all_accs))
    log(f"Union (deduplicated) accessions: {len(dedup_accs)}")
    with open(ids_path, "w") as f:
        for a in dedup_accs:
            f.write(a + "\n")
    log(f"Wrote accessions -> {ids_path}")

    if not dedup_accs:
        log("No accessions found. Exit.")
        open(raw_path, "w", encoding="utf-8").close()
        open(filt_path, "w", encoding="utf-8").close()
        with open(summary_path, "w", encoding="utf-8") as f:
            for k, v in per_source_counts.items():
                f.write(f"{k}: {v}\n")
            f.write("Union (dedup): 0\n")
            f.write(f"source_mode={source_mode}\n")
            f.write(f"ids_input_path={ids_input_path}\n")
            f.write(f"output_prefix={output_prefix}\n")
            f.write(f"motif_source={adaptive_profile.get('motif_source')}\n")
            f.write(
                f"minimal_conserved_motif_source={adaptive_profile.get('minimal_conserved_motif_source')}\n"
            )
            f.write(
                f"minimal_conserved_motif_patterns={json.dumps(adaptive_profile.get('minimal_conserved_motif_patterns') or [], ensure_ascii=False)}\n"
            )
            f.write(
                f"minimal_conserved_motif_rationale={adaptive_profile.get('minimal_conserved_motif_rationale')}\n"
            )
            f.write("No accessions found.\n")
        return

    log(f"downloading UniProt FASTA (batch={args.batch_size}; reviewed_only={reviewed_only})")
    fasta_txt = safe_uniprot_stream_batch(
        dedup_accs,
        batch_size=args.batch_size,
        reviewed_only=reviewed_only,
        sleep=args.sleep,
        fmt="fasta",
    )
    with open(raw_path, "w") as f:
        f.write(fasta_txt)
    log(f"Wrote raw FASTA -> {raw_path}")

    if args.skip_filtering:
        with open(filt_path, "w", encoding="utf-8") as f:
            f.write(fasta_txt)
        with open(summary_path, "w") as f:
            for k, v in per_source_counts.items():
                f.write(f"{k}: {v}\n")
            f.write(f"Union (dedup): {len(dedup_accs)}\n")
            f.write("Filtering skipped.\n")
            f.write(f"source_mode={source_mode}\n")
            f.write(f"ids_input_path={ids_input_path}\n")
            f.write(f"output_prefix={output_prefix}\n")
            f.write(f"reviewed_only={reviewed_only} extra_query_source={args.extra_query}\n")
            f.write(f"motif_source={adaptive_profile.get('motif_source')}\n")
            f.write(
                f"minimal_conserved_motif_source={adaptive_profile.get('minimal_conserved_motif_source')}\n"
            )
            f.write(
                f"minimal_conserved_motif_patterns={json.dumps(adaptive_profile.get('minimal_conserved_motif_patterns') or [], ensure_ascii=False)}\n"
            )
            f.write(
                f"minimal_conserved_motif_rationale={adaptive_profile.get('minimal_conserved_motif_rationale')}\n"
            )
        log("Filtering skipped. Done.")
        return

    log("fetching UniProt metadata for metadata and motif filtering")
    meta = fetch_uniprot_metadata(
        dedup_accs,
        batch_size=max(100, args.batch_size),
        reviewed_only=reviewed_only,
        timeout=120,
        sleep=args.sleep,
    )
    log(f"Metadata entries fetched: {len(meta)}")

    profile = dict(adaptive_profile)
    profile["positive_tokens"] = list(profile.get("positive_tokens") or [])
    profile["negative_tokens"] = list(profile.get("negative_tokens") or [])
    profile["go_terms"] = [str(x).lower() for x in (profile.get("go_terms") or [])]
    profile["ec_numbers"] = [str(x).lower() for x in (profile.get("ec_numbers") or [])]
    if args.motif_regex is None:
        profile["motif_regex"] = str(profile.get("motif_regex") or "").strip()
    else:
        profile["motif_regex"] = str(args.motif_regex or "").strip()
    profile["active_site_policy"] = str(
        args.active_site_policy or profile.get("active_site_policy") or "all_required"
    )
    profile["active_site_min_hits"] = (
        args.active_site_min_hits
        if args.active_site_min_hits is not None
        else profile.get("active_site_min_hits")
    )
    cli_rules = _parse_active_site_rules_json(args.active_site_rules_json)
    if cli_rules:
        profile["active_site_rules"] = cli_rules
    else:
        profile["active_site_rules"] = list(profile.get("active_site_rules") or [])
    if args.skip_active_site_filter:
        profile["motif_regex"] = ""
        profile["active_site_rules"] = []
        profile["active_site_min_hits"] = None

    recs = parse_fasta(fasta_txt)

    basic_kept, basic_cnt = filter_basic(recs, min_len, max_len, drop_fragments)

    meta_kept, meta_stats = filter_metadata(
        basic_kept,
        metadata=meta,
        positive_tokens=profile["positive_tokens"],
        negative_tokens=profile["negative_tokens"],
        ec_numbers=profile["ec_numbers"],
        go_terms=profile["go_terms"],
    )

    final_kept, motif_stats, active_site_stats, motif_policy = filter_profile_motifs(
        meta_kept,
        profile=profile,
        force=bool(args.enable_active_site_filter),
    )
    log(
        "Motif policy: "
        f"applied={motif_policy['applied']} reason={motif_policy['reason']} "
        f"candidate_records={motif_policy['candidate_records']}"
    )

    write_fasta(filt_path, final_kept)
    log(f"Wrote filtered FASTA -> {filt_path}")

    with open(summary_path, "w") as f:
        for k, v in per_source_counts.items():
            f.write(f"{k}: {v}\n")
        f.write(f"Union (dedup): {len(dedup_accs)}\n")
        f.write(f"source_mode={source_mode}\n")
        f.write(f"ids_input_path={ids_input_path}\n")
        f.write(f"Raw sequences parsed: {len(recs)}\n")
        f.write(f"output_prefix={output_prefix}\n")
        f.write(
            f"Preset={args.preset} Filters: min_len={min_len} max_len={max_len} drop_fragments={drop_fragments}\n"
        )
        f.write(f"reviewed_only={reviewed_only} extra_query_source={args.extra_query}\n")
        f.write(f"positive_tokens={profile['positive_tokens']}\n")
        f.write(f"negative_tokens={profile['negative_tokens']}\n")
        f.write(f"go_terms={profile['go_terms']}\n")
        f.write(f"ec_numbers={profile['ec_numbers']}\n")
        f.write(f"motif_regex={profile['motif_regex']}\n")
        f.write(f"motif_source={profile.get('motif_source')}\n")
        f.write(f"minimal_conserved_motif_source={profile.get('minimal_conserved_motif_source')}\n")
        f.write(
            f"minimal_conserved_motif_patterns={json.dumps(profile.get('minimal_conserved_motif_patterns') or [], ensure_ascii=False)}\n"
        )
        f.write(
            f"minimal_conserved_motif_rationale={profile.get('minimal_conserved_motif_rationale')}\n"
        )
        f.write(f"active_site_policy={profile['active_site_policy']}\n")
        f.write(f"active_site_min_hits={profile['active_site_min_hits']}\n")
        f.write(
            f"active_site_rules={json.dumps(profile['active_site_rules'], ensure_ascii=False)}\n"
        )
        f.write(f"enable_active_site_filter={bool(args.enable_active_site_filter)}\n")
        f.write(f"skip_active_site_filter={bool(args.skip_active_site_filter)}\n")
        for k, v in motif_policy.items():
            f.write(f"motif_policy_{k}={v}\n")
        for k, v in basic_cnt.items():
            f.write(f"basic_{k}: {v}\n")
        for k, v in meta_stats.items():
            f.write(f"metadata_{k}: {v}\n")
        for k, v in motif_stats.items():
            f.write(f"motif_{k}: {v}\n")
        for k, v in active_site_stats.items():
            f.write(f"active_site_{k}: {v}\n")
        f.write("filter_logic=basic -> metadata -> motif_policy -> motif -> active_site\n")

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
