"""Optional OpenAI-compatible runtime for structured profile generation.

Providers are selected explicitly; the profile CLI supplies a default model. API
keys are read from environment variables only and are never written to disk or
included in log output.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


LLM_PROVIDER_OPENAI_COMPAT = "openai-compatible"
LLM_PROVIDER_OPENROUTER = "openrouter"
LLM_PROVIDER_CHOICES = (LLM_PROVIDER_OPENAI_COMPAT, LLM_PROVIDER_OPENROUTER)

DEFAULT_OPENAI_COMPAT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_QUERY_FASTA_LLM_PROVIDER: str | None = None
DEFAULT_QUERY_FASTA_LLM_MODEL: str = "openai/gpt-5.4-pro"

_MODEL_ALIASES = {
    "gpt54pro": "openai/gpt-5.4-pro",
    "gptoss20b": "openai/gpt-oss-20b",
    "gptoss120b": "openai/gpt-oss-120b",
    "gptoss20bopenai": "openai/gpt-oss-20b",
    "gptoss120bopenai": "openai/gpt-oss-120b",
}


@dataclass(frozen=True)
class ResolvedLLMRuntime:
    provider: str
    model: str
    base_url: str


def _alias_key(raw: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())


def _env_float(name: str) -> Optional[float]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _env_int(name: str) -> Optional[int]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def canonicalize_llm_provider(provider: Optional[str]) -> str:
    key = _alias_key(provider)
    if key in {"openrouter", "openrouterai"}:
        return LLM_PROVIDER_OPENROUTER
    if key in {"openaicompat", "openaicompatible", "openai", "trtllm", "trtllmopenai"}:
        return LLM_PROVIDER_OPENAI_COMPAT
    raise ValueError(f"Unsupported llm provider: {provider!r}")


def require_openai_like_provider(provider: Optional[str], field_name: str = "provider") -> str:
    resolved = canonicalize_llm_provider(provider)
    if resolved not in LLM_PROVIDER_CHOICES:
        allowed = " or ".join(repr(item) for item in LLM_PROVIDER_CHOICES)
        raise ValueError(f"{field_name} must be {allowed}, got {provider!r}")
    return resolved


def infer_gpt_oss_variant(model: Optional[str]) -> Optional[str]:
    s = str(model or "").strip().lower()
    if not s:
        return None
    if re.search(r"(^|[^0-9])120b($|[^0-9])", s):
        return "120b"
    if re.search(r"(^|[^0-9])20b($|[^0-9])", s):
        return "20b"
    return None


def canonicalize_llm_model(model: Optional[str], provider: Optional[str]) -> str:
    resolved_provider = canonicalize_llm_provider(provider)
    raw = str(model or "").strip()
    if not raw:
        raise ValueError("A model identifier is required for live profile generation.")
    key = _alias_key(raw)
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    if key == "gptoss20b":
        return "openai/gpt-oss-20b"
    if key == "gptoss120b":
        return "openai/gpt-oss-120b"
    return raw if resolved_provider in LLM_PROVIDER_CHOICES else raw


def normalize_openai_compatible_base_url(base_url: Optional[str]) -> str:
    raw = str(base_url or "").strip() or DEFAULT_OPENAI_COMPAT_BASE_URL
    parsed = urllib.parse.urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid openai-compatible base URL: {base_url!r}")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path = path + "/v1"
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized).rstrip("/")


def is_openrouter_base_url(base_url: Optional[str]) -> bool:
    raw = str(base_url or "").strip()
    if not raw:
        return False
    parsed = urllib.parse.urlparse(raw)
    host = str(parsed.netloc or parsed.path or "").strip().lower()
    return "openrouter.ai" in host


def openai_like_provider_for_base_url(provider: Optional[str], base_url: Optional[str]) -> str:
    resolved_provider = canonicalize_llm_provider(provider)
    if resolved_provider == LLM_PROVIDER_OPENROUTER:
        return resolved_provider
    if is_openrouter_base_url(base_url):
        return LLM_PROVIDER_OPENROUTER
    return LLM_PROVIDER_OPENAI_COMPAT


def default_llm_base_url_for_provider(provider: Optional[str]) -> str:
    resolved_provider = canonicalize_llm_provider(provider)
    if resolved_provider == LLM_PROVIDER_OPENROUTER:
        env_host = (os.environ.get("OPENROUTER_BASE_URL") or "").strip()
        if env_host:
            return normalize_openai_compatible_base_url(env_host)
        return DEFAULT_OPENROUTER_BASE_URL
    env_host = (
        os.environ.get("OPENAI_COMPAT_BASE_URL") or os.environ.get("LLM_BASE_URL") or ""
    ).strip()
    if env_host:
        return normalize_openai_compatible_base_url(env_host)
    return DEFAULT_OPENAI_COMPAT_BASE_URL


def openai_compatible_service_root(base_url: Optional[str]) -> str:
    api_root = normalize_openai_compatible_base_url(base_url)
    parsed = urllib.parse.urlparse(api_root)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    normalized = parsed._replace(path=path or "", params="", query="", fragment="")
    return urllib.parse.urlunparse(normalized).rstrip("/")


def normalize_llm_base_url(provider: Optional[str], base_url: Optional[str]) -> str:
    resolved_provider = canonicalize_llm_provider(provider)
    default_base_url = default_llm_base_url_for_provider(resolved_provider)
    raw = str(base_url or "").strip() or default_base_url
    return normalize_openai_compatible_base_url(raw)


def resolve_openai_like_runtime(
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    *,
    field_name: str = "provider",
) -> ResolvedLLMRuntime:
    resolved_provider = require_openai_like_provider(provider, field_name=field_name)
    resolved_model = canonicalize_llm_model(model, resolved_provider)
    raw_base_url = str(base_url or "").strip()
    resolved_base_url = (
        normalize_llm_base_url(resolved_provider, raw_base_url)
        if raw_base_url
        else default_llm_base_url_for_provider(resolved_provider)
    )
    return ResolvedLLMRuntime(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
    )


def _openrouter_api_key_env_candidates(model: Optional[str]) -> List[str]:
    out: List[str] = []
    variant = infer_gpt_oss_variant(model)
    if variant == "120b":
        out += ["OPENROUTER_API_KEY_120B", "OPENROUTER_API_KEY_GPT_OSS_120B"]
    elif variant == "20b":
        out += ["OPENROUTER_API_KEY_20B", "OPENROUTER_API_KEY_GPT_OSS_20B"]
    out.append("OPENROUTER_API_KEY")
    deduped: List[str] = []
    for name in out:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def _openrouter_api_key_alias_selector_candidates(model: Optional[str]) -> List[str]:
    out: List[str] = []
    variant = infer_gpt_oss_variant(model)
    if variant == "120b":
        out += ["OPENROUTER_API_KEY_ALIAS_120B", "OPENROUTER_API_KEY_ALIAS_GPT_OSS_120B"]
    elif variant == "20b":
        out += ["OPENROUTER_API_KEY_ALIAS_20B", "OPENROUTER_API_KEY_ALIAS_GPT_OSS_20B"]
    out += ["OPENROUTER_API_KEY_ALIAS", "OPENROUTER_API_KEY_NAME"]
    deduped: List[str] = []
    for name in out:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def _resolve_openrouter_api_key_alias(model: Optional[str]) -> Optional[str]:
    for selector_name in _openrouter_api_key_alias_selector_candidates(model):
        selector_value = str(os.environ.get(selector_name) or "").strip()
        if not selector_value:
            continue
        if selector_value.startswith("sk-"):
            return selector_value
        token = str(os.environ.get(selector_value) or "").strip()
        if token:
            return token
        raise RuntimeError(
            "OpenRouter API key alias selector points to an empty env var "
            f"({selector_name}={selector_value!r})"
        )
    return None


def _require_openrouter_api_key(
    provider: Optional[str],
    base_url: Optional[str],
    model: Optional[str] = None,
) -> str:
    for env_name in _openrouter_api_key_env_candidates(model):
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            return token
    alias_token = _resolve_openrouter_api_key_alias(model)
    if alias_token:
        return alias_token
    resolved_provider = openai_like_provider_for_base_url(provider, base_url)
    expected = ", ".join(_openrouter_api_key_env_candidates(model))
    alias_expected = ", ".join(_openrouter_api_key_alias_selector_candidates(model))
    raise RuntimeError(
        "Missing OpenRouter API key for OpenRouter requests "
        f"(provider={resolved_provider!r}, model={str(model or '').strip() or 'DEFAULT'}, "
        f"base_url={str(base_url or '').strip() or 'DEFAULT'}, "
        f"expected_any_of=[{expected}], alias_selectors=[{alias_expected}])"
    )


def build_llm_headers(
    *,
    provider: Optional[str],
    base_url: Optional[str],
    model: Optional[str] = None,
    include_json_content_type: bool = False,
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if include_json_content_type:
        headers["Content-Type"] = "application/json"
    if openai_like_provider_for_base_url(provider, base_url) == LLM_PROVIDER_OPENROUTER:
        headers["Authorization"] = (
            f"Bearer {_require_openrouter_api_key(provider, base_url, model=model)}"
        )
    return headers


def _http_json(
    url: str,
    method: str,
    timeout_sec: float,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url=url,
        data=data,
        headers=dict(headers or {}),
        method=method,
    )
    with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _model_name_matches(requested: str, available: str) -> bool:
    want = str(requested or "").strip()
    have = str(available or "").strip()
    if not want or not have:
        return False
    if want == have:
        return True
    want_tail = want.split("/")[-1]
    have_tail = have.split("/")[-1]
    return bool(want_tail and want_tail == have_tail)


def fetch_openai_compatible_models(
    base_url: Optional[str],
    timeout_sec: float = 5.0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[str]:
    resolved_provider = openai_like_provider_for_base_url(provider, base_url)
    api_root = normalize_llm_base_url(resolved_provider, base_url)
    body = _http_json(
        url=api_root + "/models",
        method="GET",
        timeout_sec=timeout_sec,
        headers=build_llm_headers(provider=resolved_provider, base_url=api_root, model=model),
    )
    data = body.get("data") if isinstance(body, dict) else None
    out: List[str] = []
    if isinstance(data, list):
        for item in data:
            model_id = ""
            if isinstance(item, dict):
                model_id = str(item.get("id") or "").strip()
            else:
                model_id = str(item).strip()
            if model_id and model_id not in out:
                out.append(model_id)
    return out


def check_openai_compatible_server(
    *,
    provider: Optional[str] = None,
    base_url: Optional[str],
    model: str,
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    resolved_provider = openai_like_provider_for_base_url(provider, base_url)
    api_root = normalize_llm_base_url(resolved_provider, base_url)
    service_root = openai_compatible_service_root(api_root)
    action = "reachable"
    if resolved_provider == LLM_PROVIDER_OPENROUTER:
        _require_openrouter_api_key(resolved_provider, api_root, model=model)
        action = "models_reachable"
    models = fetch_openai_compatible_models(
        api_root,
        timeout_sec=timeout_sec,
        provider=resolved_provider,
        model=model,
    )
    if models and not any(_model_name_matches(model, item) for item in models):
        raise RuntimeError(
            f"OpenAI-like model not found: {model} on {api_root}. available={models}"
        )
    return {
        "ok": True,
        "provider": resolved_provider,
        "action": action,
        "host": api_root,
        "service_root": service_root,
        "models_after": models,
        "model": model,
    }


def _extract_openai_message_content(body: Any) -> Optional[str]:
    def _extract_textish(value: Any) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if not isinstance(item, dict):
                    continue
                for key in ("text", "content", "reasoning", "output_text"):
                    candidate = item.get(key)
                    if isinstance(candidate, str):
                        text = candidate.strip()
                        if text:
                            parts.append(text)
                            break
            joined = "".join(parts).strip()
            return joined or None
        return None

    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None
    msg = choice0.get("message")
    if not isinstance(msg, dict):
        return None
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments.strip()
            if isinstance(arguments, dict):
                return json.dumps(arguments, ensure_ascii=False)
    content = msg.get("content")
    content_text = _extract_textish(content)
    if content_text is not None:
        return content_text
    for key in ("reasoning", "text", "output_text"):
        fallback_text = _extract_textish(msg.get(key))
        if fallback_text is not None:
            return fallback_text
    for key in ("output_text", "text"):
        fallback_text = _extract_textish(body.get(key))
        if fallback_text is not None:
            return fallback_text
    return None


def request_chat_text(
    *,
    provider: Optional[str],
    base_url: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    response_format_json: bool = False,
    response_json_schema: Optional[Dict[str, Any]] = None,
    json_tool: Optional[Dict[str, Any]] = None,
    retries: int = 2,
    timeout_sec: float = 600.0,
    backoff: float = 1.5,
) -> str:
    env_timeout = _env_float("QUERY_FASTA_LLM_TIMEOUT_SEC")
    if env_timeout is not None and env_timeout > 0.0:
        timeout_sec = float(env_timeout)
    env_retries = _env_int("QUERY_FASTA_LLM_RETRIES")
    if int(retries) > 0 and env_retries is not None and env_retries >= 0:
        retries = int(env_retries)
    env_backoff = _env_float("QUERY_FASTA_LLM_RETRY_BACKOFF")
    if env_backoff is not None and env_backoff > 0.0:
        backoff = float(env_backoff)

    resolved_provider = canonicalize_llm_provider(provider)
    url = normalize_llm_base_url(resolved_provider, base_url) + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    wants_json = bool(response_format_json or response_json_schema or json_tool)
    if json_tool:
        tool_name = str(json_tool.get("name") or "emit_json").strip() or "emit_json"
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": str(json_tool.get("description") or "Emit structured JSON."),
                    "parameters": dict(json_tool.get("parameters") or {}),
                },
            }
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": tool_name}}
    elif response_json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": response_json_schema,
        }
        payload["structured_outputs"] = True
    elif response_format_json:
        payload["response_format"] = {"type": "json_object"}
    if wants_json:
        payload["verbosity"] = "low"
        if resolved_provider == LLM_PROVIDER_OPENROUTER:
            payload["plugins"] = [{"id": "response-healing"}]

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_llm_headers(
            provider=resolved_provider,
            base_url=url,
            model=model,
            include_json_content_type=True,
        ),
        method="POST",
    )

    last_exc: Optional[Exception] = None
    for attempt in range(int(retries) + 1):
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = _extract_openai_message_content(body)
            if isinstance(content, str):
                return content
            body_excerpt = json.dumps(body, ensure_ascii=False)[:600]
            raise RuntimeError(
                "Unexpected openai-compatible response shape "
                f"(missing choices[0].message.content): body={body_excerpt}"
            )
        except urllib.error.HTTPError as exc:
            body_text = (
                exc.read().decode("utf-8", errors="replace")[:400] if hasattr(exc, "read") else ""
            )
            last_exc = RuntimeError(f"HTTPError {exc.code}: {exc.reason}; body={body_text}")
        except (
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as exc:
            last_exc = exc
        if attempt >= int(retries):
            break
        time.sleep(float(backoff) ** attempt)

    detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc is not None else "unknown"
    raise RuntimeError(
        f"LLM chat failed (provider={resolved_provider}, base_url={base_url}, model={model}): {detail}"
    ) from last_exc
