"""Runtime configuration helpers for Loopwright.

This module owns provider-neutral environment names, defaults, URL validation,
and safe display helpers. Keep it free of native-heavy imports so entrypoints
can import it without weakening native runtime bootstrap ordering.
"""

import ipaddress
import logging
import os
import urllib.parse
from typing import Optional


LOGGER = logging.getLogger(__name__)

FAST_MODE_ENV_VAR = "FAST_MODE"
LLM_BACKEND_ENV_VAR = "LLM_BACKEND"
LLM_MODEL_ENV_VAR = "LLM_MODEL"
MODEL_THINKING_ENV_VAR = "MODEL_THINKING"
MAX_OUTPUT_TOKENS_ENV_VAR = "MAX_OUTPUT_TOKENS"
EMBEDDINGS_MODEL_ENV_VAR = "EMBEDDINGS_MODEL"
OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV_VAR = "OLLAMA_MODEL"
OLLAMA_EMBEDDINGS_MODEL_ENV_VAR = "OLLAMA_EMBED_MODEL"
OLLAMA_THINK_LEVEL_ENV_VAR = "OLLAMA_THINK_LEVEL"
OLLAMA_TIMEOUT_ENV_VAR = "OLLAMA_TIMEOUT"
OPENAI_COMPAT_BASE_URL_ENV_VAR = "OPENAI_COMPAT_BASE_URL"
OPENAI_COMPAT_API_KEY_ENV_VAR = "OPENAI_COMPAT_API_KEY"
OPENAI_COMPAT_MODEL_ENV_VAR = "OPENAI_COMPAT_MODEL"
OPENAI_COMPAT_EMBEDDINGS_MODEL_ENV_VAR = "OPENAI_COMPAT_EMBED_MODEL"
OPENAI_COMPAT_TIMEOUT_ENV_VAR = "OPENAI_COMPAT_TIMEOUT"
EMBEDDINGS_DEVICE_ENV_VAR = "EMBEDDINGS_DEVICE"
WEB_SEARCH_TIMEOUT_ENV_VAR = "WEB_SEARCH_TIMEOUT"
WEB_SEARCH_MAX_RESULTS_ENV_VAR = "WEB_SEARCH_MAX_RESULTS"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "nemotron-3-nano:4b"
DEFAULT_OLLAMA_EMBEDDINGS_MODEL = "embeddinggemma"
DEFAULT_EMBEDDINGS_MODEL = "local-hashing-384"
DEFAULT_WEB_SEARCH_TIMEOUT = 12
DEFAULT_WEB_SEARCH_MAX_RESULTS = 4

SUPPORTED_LLM_BACKENDS = {
    "auto",
    "mock",
    "ollama",
    "openai-compatible",
}
SUPPORTED_EMBEDDINGS_MODELS = {DEFAULT_EMBEDDINGS_MODEL}
SUPPORTED_OLLAMA_THINK_LEVELS = {"low", "medium", "high", "max"}
GPT_OSS_OLLAMA_THINK_LEVELS = {"low", "medium", "high"}
DEFAULT_GPT_OSS_OLLAMA_THINK_LEVEL = "medium"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%r. Using %s.", name, value, default)
        return default


def env_int_range(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = env_int(name, default)
    if value < minimum:
        LOGGER.warning("Invalid %s=%r. Using minimum %s.", name, value, minimum)
        return minimum
    if value > maximum:
        LOGGER.warning("Invalid %s=%r. Using maximum %s.", name, value, maximum)
        return maximum
    return value


def first_env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def is_gpt_oss_model(model: str) -> bool:
    model_name = model.strip().lower().split("/")[-1]
    return model_name.startswith("gpt-oss")


def normalize_ollama_think_level(
    value: Optional[str],
    *,
    model: str,
) -> Optional[str]:
    level = (value or "").strip().lower()
    if level in {"", "auto"}:
        return DEFAULT_GPT_OSS_OLLAMA_THINK_LEVEL if is_gpt_oss_model(model) else None
    if level not in SUPPORTED_OLLAMA_THINK_LEVELS:
        raise RuntimeError(
            f"{OLLAMA_THINK_LEVEL_ENV_VAR} must be one of auto, low, medium, "
            "high, or max."
        )
    if is_gpt_oss_model(model) and level not in GPT_OSS_OLLAMA_THINK_LEVELS:
        raise RuntimeError(
            f"{OLLAMA_THINK_LEVEL_ENV_VAR}=max is not supported for GPT-OSS "
            "models. Use low, medium, or high."
        )
    return level


def is_loopback_host(hostname: str) -> bool:
    host = hostname.rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalize_ollama_base_url(base_url: str) -> str:
    raw_base_url = base_url.strip().rstrip("/")
    if not raw_base_url:
        raise RuntimeError(f"{OLLAMA_BASE_URL_ENV_VAR} must not be empty.")
    parse_failed = False
    try:
        parsed = urllib.parse.urlsplit(raw_base_url)
        _port = parsed.port
    except ValueError:
        parse_failed = True
    if parse_failed:
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must be a valid loopback HTTP(S) URL."
        ) from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must be a loopback HTTP(S) URL."
        )
    if not is_loopback_host(parsed.hostname):
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must point to loopback local Ollama. "
            "Use LLM_BACKEND=openai-compatible for remote gateways."
        )
    if parsed.username or parsed.password:
        raise RuntimeError(f"{OLLAMA_BASE_URL_ENV_VAR} must not include credentials.")
    if parsed.query or parsed.fragment:
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must not include query or fragment data."
        )
    if parsed.path not in {"", "/"}:
        raise RuntimeError(f"{OLLAMA_BASE_URL_ENV_VAR} must not include a path.")

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urllib.parse.urlunsplit((parsed.scheme, netloc, "", "", ""))


def safe_ollama_base_url_for_error(base_url: str) -> str:
    raw_base_url = base_url.strip()
    if not raw_base_url:
        return "<unset>"
    try:
        parsed = urllib.parse.urlsplit(raw_base_url)
    except ValueError:
        return "<invalid>"

    if parsed.hostname:
        netloc = parsed.hostname
        try:
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
        except ValueError:
            pass
        if parsed.scheme in {"http", "https"}:
            return urllib.parse.urlunsplit((parsed.scheme, netloc, "", "", ""))
    return "<invalid>"


def is_loopback_openai_compatible_host(hostname: str) -> bool:
    return is_loopback_host(hostname)


def normalize_openai_compatible_base_url(base_url: str) -> str:
    raw_base_url = base_url.strip().rstrip("/")
    if not raw_base_url:
        raise RuntimeError(
            f"{OPENAI_COMPAT_BASE_URL_ENV_VAR} is required for "
            "LLM_BACKEND=openai-compatible."
        )
    parse_failed = False
    try:
        parsed = urllib.parse.urlsplit(raw_base_url)
        _port = parsed.port
    except ValueError:
        parse_failed = True
    if parse_failed:
        raise RuntimeError(
            f"{OPENAI_COMPAT_BASE_URL_ENV_VAR} must be a valid HTTP(S) URL."
        )
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(
            f"{OPENAI_COMPAT_BASE_URL_ENV_VAR} must be an HTTP(S) URL."
        )
    if parsed.scheme == "http" and not is_loopback_openai_compatible_host(
        parsed.hostname
    ):
        raise RuntimeError(
            f"{OPENAI_COMPAT_BASE_URL_ENV_VAR} must use HTTPS for non-loopback "
            "hosts."
        )
    if parsed.username or parsed.password:
        raise RuntimeError(
            f"{OPENAI_COMPAT_BASE_URL_ENV_VAR} must not include credentials."
        )
    if parsed.query or parsed.fragment:
        raise RuntimeError(
            f"{OPENAI_COMPAT_BASE_URL_ENV_VAR} must not include query strings "
            "or fragments."
        )
    return raw_base_url


def safe_openai_compatible_base_url_for_error(base_url: str) -> str:
    raw_base_url = base_url.strip()
    if not raw_base_url:
        return "<unset>"
    try:
        parsed = urllib.parse.urlsplit(raw_base_url)
    except ValueError:
        return "<invalid>"

    if parsed.hostname:
        netloc = parsed.hostname
        try:
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
        except ValueError:
            pass
        path = "/v1" if parsed.path.rstrip("/") == "/v1" else ""
        if parsed.scheme in {"http", "https"}:
            return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))
    return "<invalid>"
