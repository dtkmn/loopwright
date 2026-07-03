"""Loopwright package."""

__all__ = ["AILoopEngine"]


def __getattr__(name):
    if name == "AILoopEngine":
        from .ai_loop_engine import AILoopEngine

        return AILoopEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
