"""Model and embedding adapters for Loopwright providers."""

import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from .native_runtime import apply_native_runtime_defaults
except ImportError:
    from native_runtime import apply_native_runtime_defaults


apply_native_runtime_defaults()

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict, Field

try:
    from .runtime_config import (
        DEFAULT_OLLAMA_BASE_URL,
        OLLAMA_BASE_URL_ENV_VAR,
        is_loopback_host,
        is_loopback_openai_compatible_host,
        normalize_ollama_base_url,
        normalize_ollama_think_level,
        normalize_openai_compatible_base_url,
        safe_openai_compatible_base_url_for_error,
    )
except ImportError:
    from runtime_config import (
        DEFAULT_OLLAMA_BASE_URL,
        OLLAMA_BASE_URL_ENV_VAR,
        is_loopback_host,
        is_loopback_openai_compatible_host,
        normalize_ollama_base_url,
        normalize_ollama_think_level,
        normalize_openai_compatible_base_url,
        safe_openai_compatible_base_url_for_error,
    )


OLLAMA_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OPENAI_COMPAT_NO_PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({})
)
OLLAMA_LENGTH_DONE_REASONS = {"length", "limit", "max_tokens", "num_predict"}


def _legacy_runtime_override(name: str, current):
    for module_name in (
        "src.DocumentQA",
        "src.ai_loop_runtime",
        "DocumentQA",
        "ai_loop_runtime",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        candidate = getattr(module, name, None)
        if candidate is None or candidate is current:
            continue
        if (
            getattr(candidate, "__name__", None) == name
            and getattr(candidate, "__module__", None)
            in {"src.model_adapters", "model_adapters"}
        ):
            continue
        if candidate is not None:
            return candidate
    return None


def open_ollama_request_no_proxy(request, *, timeout: int):
    override = _legacy_runtime_override(
        "open_ollama_request_no_proxy", open_ollama_request_no_proxy
    )
    if override is not None:
        return override(request, timeout=timeout)

    try:
        parsed = urllib.parse.urlsplit(request.full_url)
        _port = parsed.port
    except ValueError:
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must be a valid loopback HTTP(S) URL."
        ) from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not is_loopback_host(parsed.hostname)
        or parsed.username
        or parsed.password
    ):
        raise RuntimeError(
            f"{OLLAMA_BASE_URL_ENV_VAR} must point to loopback local Ollama. "
            "Use LLM_BACKEND=openai-compatible for remote gateways."
        )
    return OLLAMA_NO_PROXY_OPENER.open(request, timeout=timeout)


def open_openai_compatible_request(request, *, timeout: int):
    override = _legacy_runtime_override(
        "open_openai_compatible_request", open_openai_compatible_request
    )
    if override is not None:
        return override(request, timeout=timeout)

    try:
        parsed = urllib.parse.urlsplit(request.full_url)
    except ValueError:
        parsed = None
    if (
        parsed is not None
        and parsed.hostname
        and is_loopback_openai_compatible_host(parsed.hostname)
    ):
        return OPENAI_COMPAT_NO_PROXY_OPENER.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def _validate_embedding_vectors(
    embeddings: Any, provider_label: str
) -> List[List[float]]:
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(f"{provider_label} embeddings response lacked vectors")

    validated: List[List[float]] = []
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"{provider_label} embeddings response lacked vectors")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in embedding
        ):
            raise RuntimeError(
                f"{provider_label} embeddings response contained non-numeric values"
            )
        vector = [float(value) for value in embedding]
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(
                f"{provider_label} embeddings response contained non-finite values"
            )
        validated.append(vector)

    dimensions = {len(embedding) for embedding in validated}
    if len(dimensions) != 1:
        raise RuntimeError(
            f"{provider_label} embeddings response had inconsistent dimensions"
        )
    return validated


def _validate_single_embedding_vector(
    embeddings: List[List[float]], provider_label: str
) -> List[float]:
    if len(embeddings) != 1:
        raise RuntimeError(f"{provider_label} embeddings response had unexpected length")
    return embeddings[0]


class OllamaEmbeddings(Embeddings):
    """Ollama embedding adapter using the selected local provider runtime."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = normalize_ollama_base_url(base_url)
        self.timeout = timeout

    def _post_embed(self, inputs: str | List[str]) -> List[List[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(
                {
                    "model": self.model,
                    "input": inputs,
                    "truncate": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_ollama_request_no_proxy(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama embeddings API returned HTTP {exc.code} for /api/embed."
            ) from None
        except urllib.error.URLError:
            raise RuntimeError(
                f"Could not connect to Ollama embeddings at {self.base_url}/api/embed."
            ) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama embeddings endpoint returned invalid JSON") from exc

        embeddings = parsed.get("embeddings") if isinstance(parsed, dict) else None
        return _validate_embedding_vectors(embeddings, "Ollama")

    def validate_model_available(self) -> None:
        _validate_single_embedding_vector(self._post_embed("ok"), "Ollama")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._post_embed(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("Ollama embeddings response had unexpected length")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._post_embed(text)
        return _validate_single_embedding_vector(embeddings, "Ollama")


class MockLLM(LLM):
    """Explicit deterministic demo/test LLM."""

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        context_match = re.search(
            r"(?is)(?:^|\n)(?:web search context|context):\s*(.*?)(?:\n\s*Question:|\Z)",
            prompt,
        )
        if context_match:
            context = context_match.group(1)
            citation_id = None
            for line in context.splitlines():
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                citation_match = re.match(r"\[(\d+)\]", stripped_line)
                if citation_match:
                    citation_id = citation_match.group(1)
                    continue
                if citation_id:
                    sentence = re.split(r"(?<=[.!?])\s+", stripped_line, maxsplit=1)[0]
                    return f"{sentence} [{citation_id}]"
        return (
            "This is a mock response. Configure Ollama or an OpenAI-compatible "
            "backend to enable real AI-powered answers."
        )


class OllamaLLM(LLM):
    """Minimal Ollama adapter for deterministic local model inference."""

    model: str
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout: int = 120
    options: Dict[str, object] = Field(default_factory=dict)
    enable_thinking: bool = True
    think_level: Optional[str] = None
    supports_thinking: Optional[bool] = None
    last_thinking: Optional[str] = None

    def model_post_init(self, __context) -> None:
        self.think_level = normalize_ollama_think_level(
            self.think_level,
            model=self.model,
        )

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _api_base_url(self) -> str:
        return normalize_ollama_base_url(self.base_url)

    def _post_json(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
        request = urllib.request.Request(
            f"{self._api_base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_ollama_request_no_proxy(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama API returned HTTP {exc.code} for {path}.") from None
        except urllib.error.URLError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self._api_base_url}{path}."
            ) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON for {path}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Ollama returned unexpected JSON for {path}")
        return parsed

    def validate_model_available(self) -> None:
        response = self._post_json("/api/show", {"model": self.model})
        capabilities = response.get("capabilities")
        self.supports_thinking = (
            any(str(capability).lower() == "thinking" for capability in capabilities)
            if isinstance(capabilities, list)
            else False
        )

    def _clean_model_thinking(self, value: object) -> Optional[str]:
        if not isinstance(value, str):
            return None
        thinking = value.strip()
        return thinking or None

    def _strip_thinking_text(self, text: str) -> str:
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        dangling_think_end = re.search(r"</think>", cleaned, flags=re.IGNORECASE)
        if dangling_think_end:
            cleaned = cleaned[dangling_think_end.end() :]
        dangling_think_start = re.search(r"<think>", cleaned, flags=re.IGNORECASE)
        if dangling_think_start:
            cleaned = cleaned[: dangling_think_start.start()]
        return cleaned.strip()

    def _extract_inline_thinking_and_answer(
        self, text: str
    ) -> tuple[Optional[str], str]:
        snippets = [
            match.group(1).strip()
            for match in re.finditer(
                r"<think>(.*?)</think>",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match.group(1).strip()
        ]
        thinking = "\n\n".join(snippets) if snippets else None
        return thinking, self._strip_thinking_text(text)

    def _thinking_enabled_for_request(self) -> bool:
        return bool(self.enable_thinking and self.supports_thinking is True)

    def _think_value_for_request(self) -> object:
        self.think_level = normalize_ollama_think_level(
            self.think_level,
            model=self.model,
        )
        return self.think_level or True

    def _raise_if_generation_truncated(
        self, response: Dict[str, object], endpoint: str
    ) -> None:
        done_reason = response.get("done_reason")
        if isinstance(done_reason, str):
            normalized_reason = done_reason.strip().lower()
        else:
            normalized_reason = ""
        if response.get("done") is False or normalized_reason in OLLAMA_LENGTH_DONE_REASONS:
            raise RuntimeError(
                "Ollama stopped generation before the answer was complete. "
                f"Increase MAX_OUTPUT_TOKENS, use FAST_MODE=false, disable "
                f"MODEL_THINKING, or choose a model that completes within the "
                f"{endpoint} generation budget."
            )

    def _call_chat_with_thinking(
        self, prompt: str, options: Dict[str, object]
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": self._think_value_for_request(),
            "options": options,
        }
        response = self._post_json("/api/chat", payload)
        self._raise_if_generation_truncated(response, "/api/chat")
        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response did not include a message.")
        generated_text = message.get("content")
        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama chat response did not include generated text.")
        inline_thinking, answer = self._extract_inline_thinking_and_answer(
            generated_text
        )
        self.last_thinking = self._clean_model_thinking(
            message.get("thinking")
        ) or inline_thinking
        return answer

    def _call_generate(
        self, prompt: str, options: Dict[str, object]
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
        response = self._post_json("/api/generate", payload)
        self._raise_if_generation_truncated(response, "/api/generate")
        generated_text = response.get("response")
        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama response did not include generated text.")
        inline_thinking, answer = self._extract_inline_thinking_and_answer(
            generated_text
        )
        self.last_thinking = (
            self._clean_model_thinking(response.get("thinking")) or inline_thinking
            if self.enable_thinking and self.supports_thinking is True
            else None
        )
        return answer

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        self.last_thinking = None
        options = dict(self.options)
        if stop:
            options["stop"] = stop
        if self._thinking_enabled_for_request():
            return self._call_chat_with_thinking(prompt, options)
        return self._call_generate(prompt, options)


class OpenAICompatibleLLM(LLM):
    """Minimal OpenAI-compatible chat-completions adapter."""

    model: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 120
    max_tokens: int = 384
    model_config = ConfigDict(validate_assignment=True)

    def model_post_init(self, __context) -> None:
        self.base_url = normalize_openai_compatible_base_url(self.base_url)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    @property
    def _api_base_url(self) -> str:
        return normalize_openai_compatible_base_url(self.base_url)

    @property
    def _chat_completions_url(self) -> str:
        return f"{self._api_base_url}/chat/completions"

    def _post_chat(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        error_message = None
        try:
            with open_openai_compatible_request(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            error_message = (
                "OpenAI-compatible API returned HTTP "
                f"{status_code} for /chat/completions."
            )
        except urllib.error.URLError:
            safe_base_url = safe_openai_compatible_base_url_for_error(self.base_url)
            error_message = (
                "Could not connect to OpenAI-compatible endpoint at "
                f"{safe_base_url}/chat/completions."
            )
        if error_message:
            raise RuntimeError(error_message) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI-compatible endpoint returned invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI-compatible endpoint returned unexpected JSON")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible response did not include choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("OpenAI-compatible response choice was invalid")
        message = first_choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        if isinstance(first_choice.get("text"), str):
            return first_choice["text"].strip()
        raise RuntimeError("OpenAI-compatible response did not include generated text")

    def validate_model_available(self) -> None:
        self._post_chat("Respond with ok.", max_tokens=1)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._post_chat(prompt, stop=stop)


class OpenAICompatibleEmbeddings(Embeddings):
    """OpenAI-compatible embeddings adapter using /embeddings."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = normalize_openai_compatible_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _embeddings_url(self) -> str:
        return f"{self.base_url}/embeddings"

    def _post_embeddings(self, inputs: str | List[str]) -> List[List[float]]:
        expected_count = 1 if isinstance(inputs, str) else len(inputs)
        payload = {
            "model": self.model,
            "input": inputs,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._embeddings_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        error_message = None
        try:
            with open_openai_compatible_request(
                request, timeout=self.timeout
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_message = (
                "OpenAI-compatible embeddings API returned HTTP "
                f"{exc.code} for /embeddings."
            )
        except urllib.error.URLError:
            safe_base_url = safe_openai_compatible_base_url_for_error(self.base_url)
            error_message = (
                "Could not connect to OpenAI-compatible embeddings endpoint at "
                f"{safe_base_url}/embeddings."
            )
        if error_message:
            raise RuntimeError(error_message) from None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "OpenAI-compatible embeddings endpoint returned invalid JSON"
            ) from exc

        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("OpenAI-compatible embeddings response was invalid")
        if len(data) != expected_count:
            raise RuntimeError(
                "OpenAI-compatible embeddings response had unexpected length"
            )
        ordered_embeddings: List[Any] = [None] * expected_count
        seen_indices = set()
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "OpenAI-compatible embeddings response item was invalid"
                )
            index = item.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= expected_count
                or index in seen_indices
            ):
                raise RuntimeError(
                    "OpenAI-compatible embeddings response had invalid indices"
                )
            seen_indices.add(index)
            ordered_embeddings[index] = item.get("embedding")
        if len(seen_indices) != expected_count or any(
            embedding is None for embedding in ordered_embeddings
        ):
            raise RuntimeError(
                "OpenAI-compatible embeddings response had invalid indices"
            )
        embeddings = ordered_embeddings
        return _validate_embedding_vectors(embeddings, "OpenAI-compatible")

    def validate_model_available(self) -> None:
        _validate_single_embedding_vector(
            self._post_embeddings("ok"), "OpenAI-compatible"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._post_embeddings(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "OpenAI-compatible embeddings response had unexpected length"
            )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._post_embeddings(text)
        return _validate_single_embedding_vector(
            embeddings, "OpenAI-compatible"
        )


__all__ = [
    "MockLLM",
    "OllamaEmbeddings",
    "OllamaLLM",
    "OpenAICompatibleEmbeddings",
    "OpenAICompatibleLLM",
    "OLLAMA_LENGTH_DONE_REASONS",
    "OLLAMA_NO_PROXY_OPENER",
    "OPENAI_COMPAT_NO_PROXY_OPENER",
    "open_ollama_request_no_proxy",
    "open_openai_compatible_request",
]
