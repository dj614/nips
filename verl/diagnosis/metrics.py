from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

JsonLike = Dict[str, Any]


@dataclass
class ModeStats:
    mode_id: str
    solved: int
    acc: float


class DiagnosisAggregator:
    """Aggregate per-problem records into complementarity metrics.

    This class assumes each *problem* is emitted as one jsonl record with:
      - record['idx']: unique problem id (int)
      - record['ground_truth']: not None when evaluable
      - record['samples'][bucket]['responses']: list of responses with fields
          {'correct': bool|None, 'mode_id': str|None}

    It is robust to missing mode ids (treated as 'unknown').
    """

    def __init__(self) -> None:
        self._seen: Set[int] = set()
        self.n_eval: int = 0
        self.correct_modes_by_q: Dict[int, Set[str]] = {}
        self.solved_by_mode: Dict[str, Set[int]] = defaultdict(set)

    def update(self, record: JsonLike) -> None:
        if record.get("type") != "sample":
            return
        idx = record.get("idx")
        if idx is None:
            return
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except Exception:
                return
        if idx in self._seen:
            return

        gt = record.get("ground_truth")
        if gt is None:
            # Not evaluable, skip from complementarity metrics.
            self._seen.add(idx)
            return

        self._seen.add(idx)
        self.n_eval += 1

        correct_modes: Set[str] = set()
        samples = record.get("samples", {})
        for bucket_name in ("high_temp", "low_temp"):
            bucket = samples.get(bucket_name, {}) if isinstance(samples, dict) else {}
            responses = bucket.get("responses", []) if isinstance(bucket, dict) else []
            for r in responses:
                if not isinstance(r, dict):
                    continue
                if r.get("correct") is True:
                    mid = r.get("mode_id")
                    if mid is None:
                        mid = "unknown"
                    else:
                        mid = str(mid)
                    correct_modes.add(mid)
                    self.solved_by_mode[mid].add(idx)

        self.correct_modes_by_q[idx] = correct_modes

    @staticmethod
    def _jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        u = len(a | b)
        if u == 0:
            return 0.0
        return len(a & b) / u

    def finalize(self, top_modes_for_jaccard: int = 12) -> JsonLike:
        n = self.n_eval
        if n == 0:
            return {
                "n_eval": 0,
                "mode_existence": {"multi_correct_modes": 0, "frac_multi": 0.0},
                "complementarity": {"acc_union": 0.0, "acc_best_single": 0.0, "gap": 0.0},
                "modes": [],
                "jaccard": [],
            }

        mode_sizes: List[Tuple[str, int]] = [(m, len(s)) for m, s in self.solved_by_mode.items()]
        mode_sizes.sort(key=lambda x: (-x[1], x[0]))

        # Union accuracy = any correct under any mode label.
        union_set: Set[int] = set()
        for _, s in self.solved_by_mode.items():
            union_set |= s
        acc_union = len(union_set) / n

        best_single = 0
        best_mode = None
        for m, sz in mode_sizes:
            if sz > best_single:
                best_single = sz
                best_mode = m
        acc_best_single = best_single / n
        gap = acc_union - acc_best_single

        # Mode existence: how many problems have >=2 distinct correct modes.
        multi = sum(1 for ms in self.correct_modes_by_q.values() if len(ms) >= 2)
        frac_multi = multi / n

        modes_out: List[JsonLike] = []
        for m, sz in mode_sizes:
            modes_out.append({"mode_id": m, "solved": sz, "acc": sz / n})

        # Pairwise Jaccard among top modes (by solved count) for readability.
        top_modes = [m for m, _ in mode_sizes[:top_modes_for_jaccard]]
        jaccard_out: List[JsonLike] = []
        for i in range(len(top_modes)):
            for j in range(i + 1, len(top_modes)):
                a = top_modes[i]
                b = top_modes[j]
                sa = self.solved_by_mode.get(a, set())
                sb = self.solved_by_mode.get(b, set())
                jac = self._jaccard(sa, sb)
                jaccard_out.append(
                    {
                        "mode_a": a,
                        "mode_b": b,
                        "jaccard": jac,
                        "intersection": len(sa & sb),
                        "union": len(sa | sb),
                    }
                )

        # Sort Jaccard pairs by lowest overlap first (most complementary).
        jaccard_out.sort(key=lambda d: (d["jaccard"], -d["union"], d["mode_a"], d["mode_b"]))

        return {
            "n_eval": n,
            "mode_existence": {"multi_correct_modes": multi, "frac_multi": frac_multi},
            "complementarity": {
                "acc_union": acc_union,
                "acc_best_single": acc_best_single,
                "gap": gap,
                "best_mode": best_mode,
            },
            "modes": modes_out,
            "jaccard": jaccard_out,
        }
