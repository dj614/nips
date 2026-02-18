from __future__ import annotations

"""Natural-mode labeling utilities.

Step-2 metrics (entropy / coverage) need nat samples mapped into the same label
space as reachability modes (the per-problem hint strings).

This module intentionally keeps things simple and deterministic. It is meant as
a *baseline* placeholder until you plug in a real clustering / strong-model
judge.

Current baseline:
  - Extract keywords from each hint string.
  - For a nat response, score each hint by keyword overlap.
  - Choose the best-scoring hint as the response's mode label.
  - If no hint matches, return 'unknown'.

The intent is to make the end-to-end pipeline runnable and to define a clean
interface to swap in better mode detection later.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set


_STOPWORDS: Set[str] = {
    "use",
    "using",
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "via",
    "approach",
    "method",
    "idea",
    "solution",
    "prove",
    "show",
    "apply",
    "inequality",
    "lemma",
    "theorem",
    "formula",
    "rule",
}


def _norm_text(s: str) -> str:
    s = s.lower()
    # keep alnum and common math symbols as separators
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def hint_to_keywords(hint: str, *, min_len: int = 3) -> List[str]:
    """Convert a hint string into a small set of keywords.

    We intentionally keep keywords short and robust to formatting.
    """
    t = _norm_text(hint)
    toks = [w for w in t.split(" ") if len(w) >= min_len and w not in _STOPWORDS]
    # Deduplicate but keep a stable order
    out: List[str] = []
    seen: Set[str] = set()
    for w in toks:
        if w not in seen:
            out.append(w)
            seen.add(w)
    # Only keep a handful to reduce false positives.
    return out[:8]


@dataclass(frozen=True)
class LabeledMode:
    label: str
    score: int


class HeuristicHintLabeler:
    """Assign nat samples to per-problem hint labels by keyword overlap."""

    def __init__(self, *, min_score: int = 1) -> None:
        self.min_score = int(min_score)

    def label_one(self, text: str, hints: Sequence[str]) -> str:
        if not hints:
            return "unknown"
        t = _norm_text(text)
        best: LabeledMode | None = None
        for hint in hints:
            kws = hint_to_keywords(hint)
            if not kws:
                continue
            score = sum(1 for k in kws if k in t)
            if best is None or score > best.score:
                best = LabeledMode(label=str(hint), score=score)
        if best is None or best.score < self.min_score:
            return "unknown"
        return best.label

    def label_many(self, texts: Iterable[str], hints: Sequence[str]) -> List[str]:
        return [self.label_one(t, hints) for t in texts]
