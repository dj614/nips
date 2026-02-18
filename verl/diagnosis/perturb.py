from __future__ import annotations

from typing import Dict, List, Protocol, Sequence


class Perturber(Protocol):
    """Generate prompt variants for robustness/shift diagnosis.

    Step 3 leaves perturbations unimplemented; this is only an interface.
    """

    name: str

    def generate_variants(self, prompt_messages: Sequence[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Return a list of chat-message variants (each itself a list of messages)."""
        ...


class NullPerturber:
    """Placeholder: returns only the original prompt."""

    name = "null"

    def generate_variants(self, prompt_messages: Sequence[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        return [list(prompt_messages)]
