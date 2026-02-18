#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from verl.diagnosis.modebank_metrics import ModeBankEvaluator, load_jsonl


def _get_group_key(record: Dict[str, Any], group_by: Optional[str]) -> str:
    if not group_by:
        return "all"
    meta = record.get("meta") or {}
    if not isinstance(meta, dict):
        return "all"
    v = meta.get(group_by)
    if v is None:
        return "all"
    return str(v)


def _eval_file(path: str, group_by: Optional[str] = None) -> Dict[str, Any]:
    evaluators: Dict[str, ModeBankEvaluator] = defaultdict(ModeBankEvaluator)

    for rec in load_jsonl(path):
        if rec.get("type") != "sample":
            continue
        k = _get_group_key(rec, group_by)
        evaluators[k].update(rec)

    out: Dict[str, Any] = {}
    for k, ev in evaluators.items():
        out[k] = ev.finalize()
    return out


def _fmt(v: Any, *, nd: int = 4) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _print_summary(tag: str, stats_by_group: Dict[str, Any], group_by: Optional[str]) -> None:
    groups = sorted(stats_by_group.keys())
    if group_by:
        print(f"\n== {tag} (group_by={group_by}) ==")
    else:
        print(f"\n== {tag} ==")

    for g in groups:
        s = stats_by_group[g]
        ne = (s.get("nat_entropy") or {}).get("mean")
        cov = (s.get("coverage") or {}).get("mean")
        rc = s.get("reach_complementarity") or {}
        acc_union = rc.get("acc_union")
        acc_best = rc.get("acc_best")
        gap = rc.get("gap")

        header = f"[{g}]" if group_by else "[all]"
        print(
            f"{header} n_eval={s.get('n_eval')} n_reach_eval={s.get('n_reach_eval')}  "
            f"H_nat={_fmt(ne)}  coverage={_fmt(cov)}  "
            f"Acc(union)={_fmt(acc_union)}  Acc(best)={_fmt(acc_best)}  gap={_fmt(gap)}"
        )


def _delta(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return b - a
    return None


def _print_delta(base_by_group: Dict[str, Any], grpo_by_group: Dict[str, Any], group_by: Optional[str]) -> None:
    groups = sorted(set(base_by_group.keys()) | set(grpo_by_group.keys()))
    print("\n== delta (grpo - base) ==")
    for g in groups:
        b = base_by_group.get(g, {})
        p = grpo_by_group.get(g, {})

        ne_b = (b.get("nat_entropy") or {}).get("mean")
        ne_p = (p.get("nat_entropy") or {}).get("mean")
        cov_b = (b.get("coverage") or {}).get("mean")
        cov_p = (p.get("coverage") or {}).get("mean")

        rc_b = b.get("reach_complementarity") or {}
        rc_p = p.get("reach_complementarity") or {}
        gap_b = rc_b.get("gap")
        gap_p = rc_p.get("gap")

        header = f"[{g}]" if group_by else "[all]"
        print(
            f"{header} dH={_fmt(_delta(ne_b, ne_p))}  "
            f"dCoverage={_fmt(_delta(cov_b, cov_p))}  "
            f"dGap={_fmt(_delta(gap_b, gap_p))}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ModeBank diagnosis outputs (Step 2 metrics).")
    ap.add_argument("--base_jsonl", type=str, required=True, help="Step-1 output jsonl for the base model")
    ap.add_argument("--grpo_jsonl", type=str, required=True, help="Step-1 output jsonl for the GRPO model")
    ap.add_argument(
        "--group_by",
        type=str,
        default=None,
        help="Optional meta field to group metrics by (e.g., ability / problem_type)",
    )
    ap.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save the summarized results as a JSON file",
    )
    args = ap.parse_args()

    base_stats = _eval_file(args.base_jsonl, group_by=args.group_by)
    grpo_stats = _eval_file(args.grpo_jsonl, group_by=args.group_by)

    _print_summary("base", base_stats, args.group_by)
    _print_summary("grpo", grpo_stats, args.group_by)
    _print_delta(base_stats, grpo_stats, args.group_by)

    if args.save_json:
        payload = {"base": base_stats, "grpo": grpo_stats, "group_by": args.group_by}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[summarize_modebank] wrote: {args.save_json}")


if __name__ == "__main__":
    main()
