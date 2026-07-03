"""Canonical public runtime API for Loopwright.

The runtime implementation lives in ``src.ai_loop_runtime``. The historical
``src.DocumentQA`` module remains as a compatibility shim.
"""

try:
    from .ai_loop_runtime import (
        DEFAULT_OLLAMA_BASE_URL,
        DEFAULT_OLLAMA_MODEL,
        LLM_MODEL_ENV_VAR,
        MAX_DOCUMENT_CHUNKS,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        OLLAMA_THINK_LEVEL_ENV_VAR,
        SELF_CHECK_REFUSAL_ANSWER,
        AILoopEngine,
        DocumentProcessingError,
        DocumentProcessingReport,
        DocumentQA,
        DocumentQAStatus,
        QueryResult,
    )
except ImportError:
    from ai_loop_runtime import (
        DEFAULT_OLLAMA_BASE_URL,
        DEFAULT_OLLAMA_MODEL,
        LLM_MODEL_ENV_VAR,
        MAX_DOCUMENT_CHUNKS,
        OLLAMA_BASE_URL_ENV_VAR,
        OLLAMA_MODEL_ENV_VAR,
        OLLAMA_THINK_LEVEL_ENV_VAR,
        SELF_CHECK_REFUSAL_ANSWER,
        AILoopEngine,
        DocumentProcessingError,
        DocumentProcessingReport,
        DocumentQA,
        DocumentQAStatus,
        QueryResult,
    )


__all__ = [
    "AILoopEngine",
    "DocumentQA",
    "DocumentProcessingError",
    "DocumentProcessingReport",
    "DocumentQAStatus",
    "MAX_DOCUMENT_CHUNKS",
    "QueryResult",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "LLM_MODEL_ENV_VAR",
    "OLLAMA_BASE_URL_ENV_VAR",
    "OLLAMA_MODEL_ENV_VAR",
    "OLLAMA_THINK_LEVEL_ENV_VAR",
    "SELF_CHECK_REFUSAL_ANSWER",
]
