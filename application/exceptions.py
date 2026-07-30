"""Structured errors for the application layer."""

from __future__ import annotations


class PipelineError(Exception):
    """Base error for any pipeline failure."""


class PipelineStageError(PipelineError):
    """Raised when a specific stage fails. Carries stage name for FSM."""

    def __init__(self, stage: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.cause = cause
