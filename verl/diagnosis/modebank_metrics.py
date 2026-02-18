from __future__ import annotations

"""ModeBank diagnosis metrics (Step 2).

This module computes the core diagnostics described in the workflow:
  1) NAT mode entropy: average H(x) from natural samples.
  2) NAT coverage vs REACH: |M_nat(x)| / |M_reach(x)|.
  3) REACH complementarity: Acc(union) vs Acc(best).

It consumes jsonl records produced by scripts/run_mode_diagnosis.py (Step 1)
without requiring any external tools.

Note: true mode identification for M_nat is an open-ended component.
We provide a deterministic baseline labeler that maps nat samples onto the
per-problem hint labels by keyword overlap.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .nat_mode import HeuristicHintLabeler

JsonLike = Dict[str, Any]


def _entropy_from_labels(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    n = float(len(labels))
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log(p)
    return h


def _iter_sample_texts(record: JsonLike, buckets: Sequence[str]) -> Iterable[str]:
    samples = record.get("samples") or {}
    if not isinstance(samples, dict):
        return []
    for b in buckets:
        bucket = samples.get(b) or {}
        if not isinstance(bucket, dict):
            continue
        for r in bucket.get("responses") or []:
            if isinstance(r, dict):
                t = r.get("text")
                if isinstance(t, str):
                    yield t


def _reach_sets(record: JsonLike) -> Tuple[List[str], List[str]]:
    """Return (hints, solved_hints) for a record."""
    hints = record.get("hints") or []
    if not isinstance(hints, list):
        try:
            hints = list(hints)
        except Exception:
            hints = []
    solved: List[str] = []
    reach = record.get("reach") or {}
    per_hint = reach.get("per_hint") if isinstance(reach, dict) else None
    if isinstance(per_hint, list):
        for it in per_hint:
            if not isinstance(it, dict):
                continue
            hint = it.get("hint")
            if not isinstance(hint, str):
                continue
            if it.get("solved") is True:
                solved.append(hint)
    return [str(h) for h in hints], solved


@dataclass
class ModeBankSummary:
    # counts
    n_records: int = 0
    n_eval: int = 0  # ground_truth != None
    n_reach_eval: int = 0  # evaluable and has at least one hint

    # (1) nat entropy
    nat_entropy_sum: float = 0.0
    nat_entropy_n: int = 0

    # (2) coverage
    coverage_sum: float = 0.0
    coverage_n: int = 0

    # (3) reach complementarity
    reach_union_solved: int = 0

    # per-hint stats (global label space = hint string)
    hint_present: Dict[str, int] = field(default_factory=dict)
    hint_solved: Dict[str, int] = field(default_factory=dict)

    total_hint_pairs: int = 0
    total_hint_pairs_solved: int = 0


class ModeBankEvaluator:
    """Compute Step-2 ModeBank metrics from Step-1 jsonl records."""

    def __init__(
        self,
        *,
        nat_buckets: Sequence[str] = ("high_temp", "low_temp"),
        nat_labeler: Optional[HeuristicHintLabeler] = None,
        ignore_unknown_for_coverage: bool = True,
    ) -> None:
        self.nat_buckets = tuple(nat_buckets)
        self.labeler = nat_labeler or HeuristicHintLabeler(min_score=1)
        self.ignore_unknown_for_coverage = bool(ignore_unknown_for_coverage)
        self.summary = ModeBankSummary()

    def update(self, record: JsonLike) -> None:
        if record.get("type") != "sample":
            return
        self.summary.n_records += 1

        gt = record.get("ground_truth")
        is_eval = gt is not None
        if is_eval:
            self.summary.n_eval += 1

        hints, solved_hints = _reach_sets(record)

        # -------- (1) NAT entropy --------
        nat_texts = list(_iter_sample_texts(record, self.nat_buckets))
        if nat_texts:
            labels = self.labeler.label_many(nat_texts, hints)
            h = _entropy_from_labels(labels)
            self.summary.nat_entropy_sum += h
            self.summary.nat_entropy_n += 1

        # -------- (2) coverage: |M_nat| / |M_reach| --------
        if hints and is_eval:
            # M_reach(x) = {hint: reachable}
            m_reach = set(solved_hints)
            # M_nat(x) is the set of hint-labels that appear in nat outputs.
            m_nat: set[str] = set()
            if nat_texts:
                for lb in self.labeler.label_many(nat_texts, hints):
                    if lb == "unknown" and self.ignore_unknown_for_coverage:
                        continue
                    if lb in hints:
                        m_nat.add(lb)
            denom = max(1, len(m_reach))
            cov = len(m_nat) / float(denom)
            self.summary.coverage_sum += cov
            self.summary.coverage_n += 1

        # -------- (3) reach complementarity (union vs best) --------
        if hints and is_eval:
            self.summary.n_reach_eval += 1

            # union
            if len(solved_hints) > 0:
                self.summary.reach_union_solved += 1

            # per-hint stats
            reach = record.get("reach") or {}
            per_hint = reach.get("per_hint") if isinstance(reach, dict) else None
            # Some datasets may have hints but no per_hint generation (e.g., mid-temp disabled)
            if isinstance(per_hint, list):
                for it in per_hint:
                    if not isinstance(it, dict):
                        continue
                    hint = it.get("hint")
                    if not isinstance(hint, str):
                        continue
                    self.summary.hint_present[hint] = self.summary.hint_present.get(hint, 0) + 1
                    self.summary.total_hint_pairs += 1
                    if it.get("solved") is True:
                        self.summary.hint_solved[hint] = self.summary.hint_solved.get(hint, 0) + 1
                        self.summary.total_hint_pairs_solved += 1
            else:
                # fall back: count presence, but no solved signals
                for hint in hints:
                    self.summary.hint_present[hint] = self.summary.hint_present.get(hint, 0) + 1

    def finalize(self, *, top_hints: int = 20) -> JsonLike:
        s = self.summary
        out: JsonLike = {
            "n_records": s.n_records,
            "n_eval": s.n_eval,
            "n_reach_eval": s.n_reach_eval,
        }

        # (1) nat entropy
        out["nat_entropy"] = {
            "mean": (s.nat_entropy_sum / s.nat_entropy_n) if s.nat_entropy_n > 0 else None,
            "n": s.nat_entropy_n,
        }

        # (2) coverage
        out["coverage"] = {
            "mean": (s.coverage_sum / s.coverage_n) if s.coverage_n > 0 else None,
            "n": s.coverage_n,
            "ignore_unknown": self.ignore_unknown_for_coverage,
        }

        # (3) reach complementarity
        if s.n_reach_eval > 0:
            acc_union = s.reach_union_solved / float(s.n_reach_eval)
            best = 0.0
            best_hint = None
            # Unconditional per-hint acc = solved_count / N (N = n_reach_eval)
            for hint, solved in s.hint_solved.items():
                acc = solved / float(s.n_reach_eval)
                if acc > best:
                    best = acc
                    best_hint = hint
            gap = acc_union - best
            out["reach_complementarity"] = {
                "acc_union": acc_union,
                "acc_best": best,
                "gap": gap,
                "best_hint": best_hint,
                "acc_pair": (s.total_hint_pairs_solved / float(s.total_hint_pairs)) if s.total_hint_pairs > 0 else None,
                "total_hint_pairs": s.total_hint_pairs,
            }
        else:
            out["reach_complementarity"] = None

        # Top hints table (by solved count)
        hints_sorted: List[Tuple[str, int]] = sorted(s.hint_solved.items(), key=lambda kv: (-kv[1], kv[0]))
        hints_out: List[JsonLike] = []
        for hint, solved in hints_sorted[: top_hints]:
            present = s.hint_present.get(hint, 0)
            hints_out.append(
                {
                    "hint": hint,
                    "solved": solved,
                    "present": present,
                    "p_solved_given_present": (solved / float(present)) if present > 0 else None,
                    "p_solved_uncond": (solved / float(s.n_reach_eval)) if s.n_reach_eval > 0 else None,
                }
            )
        out["top_hints"] = hints_out
        return out


def load_jsonl(path: str) -> Iterable[JsonLike]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
