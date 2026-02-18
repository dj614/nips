#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Tuple

from verl.diagnosis.modebank_metrics import ModeBankEvaluator, load_jsonl

"""Summarize ModeBank diagnosis outputs (Step 2 metrics).

This script aggregates the per-problem jsonl produced by `scripts/run_mode_diagnosis.py`.
It supports three input modes:

1) Manifest (recommended):
   python scripts/summarize_modebank_diagnosis.py \
     --manifest outputs/diagnosis.manifest.json \
     --baseline_tag base \
     --group_by ability \
     --save_json outputs/modebank_summary.json

2) Explicit inputs:
   python scripts/summarize_modebank_diagnosis.py \
     --inputs base=outputs/diagnosis.base.jsonl grpo=outputs/diagnosis.grpo.jsonl \
     --baseline_tag base

3) Legacy (base/grpo):
   python scripts/summarize_modebank_diagnosis.py \
     --base_jsonl outputs/diagnosis.base.jsonl \
     --grpo_jsonl outputs/diagnosis.grpo.jsonl

"""


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


def _extract_core(stats: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Return (H_nat_mean, coverage_mean, gap) from a group stats dict."""
    ne = (stats.get("nat_entropy") or {}).get("mean")
    cov = (stats.get("coverage") or {}).get("mean")
    rc = stats.get("reach_complementarity") or {}
    gap = rc.get("gap")
    return ne, cov, gap


def _print_delta(label: str, base_by_group: Dict[str, Any], other_by_group: Dict[str, Any], group_by: Optional[str]) -> None:
    groups = sorted(set(base_by_group.keys()) | set(other_by_group.keys()))
    print(f"\n== delta ({label}) ==")
    for g in groups:
        b = base_by_group.get(g, {})
        p = other_by_group.get(g, {})

        ne_b, cov_b, gap_b = _extract_core(b)
        ne_p, cov_p, gap_p = _extract_core(p)

        header = f"[{g}]" if group_by else "[all]"
        print(
            f"{header} dH={_fmt(_delta(ne_b, ne_p))}  "
            f"dCoverage={_fmt(_delta(cov_b, cov_p))}  "
            f"dGap={_fmt(_delta(gap_b, gap_p))}"
        )


def _parse_inputs(items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid --inputs item '{it}'. Expected tag=path")
        tag, path = it.split("=", 1)
        tag = tag.strip()
        path = path.strip()
        if not tag or not path:
            raise ValueError(f"Invalid --inputs item '{it}'. Expected tag=path")
        out[tag] = path
    if not out:
        raise ValueError("--inputs is empty")
    return out


def _load_manifest(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    models = obj.get("models") if isinstance(obj, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"Manifest missing 'models' list: {path}")

    out: Dict[str, str] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        tag = m.get("tag")
        out_path = m.get("output_jsonl")
        if tag and out_path:
            out[str(tag)] = str(out_path)
    if not out:
        raise ValueError(f"No models found in manifest: {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ModeBank diagnosis outputs (Step 2 metrics).")

    # New (recommended): manifest or explicit tag=path inputs.
    ap.add_argument("--manifest", type=str, default=None, help="Manifest JSON written by run_mode_diagnosis.py")
    ap.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=None,
        help="Explicit inputs as tag=path pairs (e.g., base=... grpo=... gspo=...)",
    )

    # Legacy: base+grpo only.
    ap.add_argument("--base_jsonl", type=str, default=None, help="(Legacy) Step-1 output jsonl for the base model")
    ap.add_argument("--grpo_jsonl", type=str, default=None, help="(Legacy) Step-1 output jsonl for the GRPO model")

    ap.add_argument(
        "--baseline_tag",
        type=str,
        default="base",
        help="Baseline tag used for delta reports (when available)",
    )
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

    # Resolve inputs.
    tag_to_path: Dict[str, str]
    if args.manifest:
        tag_to_path = _load_manifest(args.manifest)
    elif args.inputs:
        tag_to_path = _parse_inputs(args.inputs)
    else:
        # Legacy fallback.
        if not args.base_jsonl or not args.grpo_jsonl:
            raise SystemExit("Must provide --manifest, --inputs, or both --base_jsonl/--grpo_jsonl")
        tag_to_path = {"base": args.base_jsonl, "grpo": args.grpo_jsonl}

    # Evaluate per model.
    stats_by_model: Dict[str, Dict[str, Any]] = {}
    for tag, path in tag_to_path.items():
        stats_by_model[tag] = _eval_file(path, group_by=args.group_by)

    # Print summaries (baseline first if present).
    order = []
    if args.baseline_tag in stats_by_model:
        order.append(args.baseline_tag)
    order += [t for t in sorted(stats_by_model.keys()) if t != args.baseline_tag]

    for tag in order:
        _print_summary(tag, stats_by_model[tag], args.group_by)

    # Delta vs baseline.
    delta_vs_baseline: Dict[str, Any] = {}
    if args.baseline_tag in stats_by_model and len(stats_by_model) > 1:
        base = stats_by_model[args.baseline_tag]
        for tag in order:
            if tag == args.baseline_tag:
                continue
            other = stats_by_model[tag]
            _print_delta(f"{tag} - {args.baseline_tag}", base, other, args.group_by)

            # Also store deltas in a structured form.
            groups = sorted(set(base.keys()) | set(other.keys()))
            d: Dict[str, Any] = {}
            for g in groups:
                ne_b, cov_b, gap_b = _extract_core(base.get(g, {}))
                ne_p, cov_p, gap_p = _extract_core(other.get(g, {}))
                d[g] = {
                    "dH": _delta(ne_b, ne_p),
                    "dCoverage": _delta(cov_b, cov_p),
                    "dGap": _delta(gap_b, gap_p),
                }
            delta_vs_baseline[tag] = d

    if args.save_json:
        payload = {
            "group_by": args.group_by,
            "baseline_tag": args.baseline_tag,
            "models": stats_by_model,
            "delta_vs_baseline": delta_vs_baseline,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[summarize_modebank] wrote: {args.save_json}")


if __name__ == "__main__":
    main()
