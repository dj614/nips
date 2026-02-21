# scripts/stats_modes_count.py
from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def count_modes(mode_obj: Any) -> int:
    """
    Count number of modes/hints stored in `mode` field.
    Supports:
      - list[str]
      - list[dict] with {"hint":..., "solution":...}
      - dict with {"modes":[...]} or {"hints":[...]}
    """
    if _is_nan(mode_obj):
        return 0

    # Sometimes parquet stores JSON as string; try parse if looks like JSON.
    if isinstance(mode_obj, str):
        s = mode_obj.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            import json
            try:
                mode_obj = json.loads(s)
            except Exception:
                # treat as single hint string
                return 1 if s else 0
        else:
            return 1 if s else 0

    if isinstance(mode_obj, list):
        # list[str] or list[dict]
        cnt = 0
        for it in mode_obj:
            if _is_nan(it):
                continue
            if isinstance(it, str):
                if it.strip():
                    cnt += 1
            elif isinstance(it, dict):
                # accept dict mode if it has hint/solution or any non-empty content
                hint = it.get("hint")
                sol = it.get("solution")
                if isinstance(hint, str) and hint.strip():
                    cnt += 1
                elif isinstance(sol, str) and sol.strip():
                    cnt += 1
                else:
                    # fallback: count non-empty dict as one mode
                    if len(it) > 0:
                        cnt += 1
            else:
                # unknown element type -> count as one if truthy
                if it:
                    cnt += 1
        return cnt

    if isinstance(mode_obj, dict):
        if "modes" in mode_obj and isinstance(mode_obj["modes"], list):
            return count_modes(mode_obj["modes"])
        if "hints" in mode_obj and isinstance(mode_obj["hints"], list):
            return count_modes(mode_obj["hints"])
        # fallback: if dict itself is a single mode
        return 1 if len(mode_obj) > 0 else 0

    # fallback: other scalar types
    return 1 if mode_obj else 0


def summarize_counts(name: str, counts: List[int]) -> None:
    total = len(counts)
    nonzero = sum(1 for c in counts if c > 0)
    zeros = total - nonzero
    avg = sum(counts) / total if total else 0.0
    mn = min(counts) if total else 0
    mx = max(counts) if total else 0

    freq = Counter(counts)
    print(f"\n== {name} ==")
    print(f"problems: {total}")
    print(f"with_mode: {nonzero} ({nonzero/total*100:.2f}%)" if total else "with_mode: 0")
    print(f"no_mode:   {zeros} ({zeros/total*100:.2f}%)" if total else "no_mode: 0")
    print(f"avg_modes_per_problem: {avg:.4f}")
    print(f"min/max: {mn}/{mx}")

    # show small histogram (sorted by count)
    keys = sorted(freq.keys())
    print("histogram (modes_count -> num_problems):")
    for k in keys:
        print(f"  {k:>2} -> {freq[k]}")


def resolve_paths(args) -> List[Tuple[str, str]]:
    """
    Returns list of (name, path)
    """
    out: List[Tuple[str, str]] = []
    if args.paths:
        for p in args.paths:
            out.append((p, p))
        return out

    # template mode
    if not args.template or not args.tasks:
        raise ValueError("Provide either --paths or (--template and --tasks).")
    for t in args.tasks:
        path = args.template.replace("${TASK}", t)
        out.append((t, path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Parquet paths (e.g. /primus_datasets/.../amc23.parquet ...).",
    )
    ap.add_argument(
        "--template",
        type=str,
        default=None,
        help='Path template with ${TASK}, e.g. "/primus_datasets/zmy/NIPS/test_data/${TASK}.parquet"',
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help='Tasks to fill into template, e.g. --tasks amc23 aime24 aime25',
    )
    ap.add_argument(
        "--mode_col",
        type=str,
        default="mode",
        help="Column name storing modes/hints (default: mode).",
    )
    args = ap.parse_args()

    paths = resolve_paths(args)

    all_counts: List[int] = []
    for name, path in paths:
        df = pd.read_parquet(path)
        if args.mode_col not in df.columns:
            counts = [0] * len(df)
        else:
            counts = [count_modes(x) for x in df[args.mode_col].tolist()]
        summarize_counts(name, counts)
        all_counts.extend(counts)

    if len(paths) > 1:
        summarize_counts("ALL", all_counts)


if __name__ == "__main__":
    main()
