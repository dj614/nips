from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

JsonLike = Dict[str, Any]


@dataclass(frozen=True)
class ModeAssignResult:
    """Result of assigning a mode to a single response."""

    mode_id: str
    info: JsonLike


class ModeDetector:
    """Base interface for mode assignment.

    A detector should assign a *stable* `mode_id` so that we can aggregate
    success sets S_k across problems.
    """

    name: str = "base"

    def assign(
        self,
        prompt_messages: Sequence[Dict[str, str]],
        response_text: str,
        extracted_answer: Optional[str] = None,
        meta: Optional[JsonLike] = None,
    ) -> ModeAssignResult:
        raise NotImplementedError


class NullModeDetector(ModeDetector):
    """Placeholder detector: put everything into a single mode."""

    name: str = "null"

    def __init__(self, mode_id: str = "default") -> None:
        self._mode_id = mode_id

    def assign(
        self,
        prompt_messages: Sequence[Dict[str, str]],
        response_text: str,
        extracted_answer: Optional[str] = None,
        meta: Optional[JsonLike] = None,
    ) -> ModeAssignResult:
        return ModeAssignResult(mode_id=self._mode_id, info={"detector": self.name})
