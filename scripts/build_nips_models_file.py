#!/usr/bin/env python

"""Build a models file (JSON) for NIPS mode diagnosis.

This is a small convenience utility to create the `--models_file` consumed by
`scripts/run_mode_diagnosis.py`.

It follows the checkpoint layout described in the user's environment:

  RL-trained models:
    /primus_oss_root/repository/quark_sft/${JOB_ID}/save_checkpoint/
      global_step_${STEP}/actor/huggingface

  Base model (prefer local if available, else HF id):
    ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B  (if env exists)
    Qwen/Qwen3-8B                              (fallback)

Default RL models are the ones listed in the request.

Example:

  python scripts/build_nips_models_file.py \
    --output_path outputs/NIPS/models.json

  # Override / extend from CLI:
  python scripts/build_nips_models_file.py \
    --output_path /tmp/models.json \
    --add_model dapo primus... 944 \
    --add_model grpo primus... 1000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PrimusRLModel:
    method: str
    job_id: str
    step: int


DEFAULT_RL_MODELS: List[PrimusRLModel] = [
    PrimusRLModel(method="dapo", job_id="primus2b1d04f0cadadacea83e0c6a4z", step=944),
    PrimusRLModel(method="grpo", job_id="primus0e8c74ef3a3a329dc5021c0a5z", step=1000),
    PrimusRLModel(method="gspo", job_id="primus7c10d4c6d9f8808ea7d141934z", step=1000),
    PrimusRLModel(method="drgrpo", job_id="primusa2ba04567a6d1800561f05c62z", step=1000),
]


def _base_ckpt_path() -> str:
    # Prefer local base checkpoint when the Primus env exists.
    src = os.environ.get("PRIMUS_SOURCE_CHECKPOINT_DIR")
    if src:
        return os.path.join(src, "Qwen3-8B")
    return "Qwen/Qwen3-8B"


def _rl_ckpt_path(job_id: str, step: int) -> str:
    ckpt_dir = f"/primus_oss_root/repository/quark_sft/{job_id}/save_checkpoint"
    return f"{ckpt_dir}/global_step_{int(step)}/actor/huggingface"


def _tag_for(method: str, step: int) -> str:
    return f"qwen_{method}_{int(step)}"


def _parse_add_model(values: List[str]) -> PrimusRLModel:
    if len(values) != 3:
        raise ValueError("--add_model expects 3 args: <method> <JOB_ID> <STEP>")
    method, job_id, step_s = values
    return PrimusRLModel(method=str(method), job_id=str(job_id), step=int(step_s))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build models.json for NIPS mode diagnosis")
    ap.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the models file (JSON).",
    )
    ap.add_argument(
        "--base_tag",
        type=str,
        default="qwen3_8b",
        help="Tag for the base model (default: qwen3_8b)",
    )
    ap.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Optional override for base model path. Default prefers PRIMUS_SOURCE_CHECKPOINT_DIR/Qwen3-8B.",
    )
    ap.add_argument(
        "--add_model",
        type=str,
        nargs=3,
        action="append",
        default=None,
        metavar=("METHOD", "JOB_ID", "STEP"),
        help="Add/override an RL model: METHOD JOB_ID STEP (repeatable).",
    )
    ap.add_argument(
        "--no_defaults",
        action="store_true",
        help="Do not include the default RL models; use only --add_model entries.",
    )
    args = ap.parse_args()

    rl_models: List[PrimusRLModel] = []
    if not args.no_defaults:
        rl_models.extend(DEFAULT_RL_MODELS)
    if args.add_model:
        rl_models.extend(_parse_add_model(v) for v in args.add_model)

    base_path = args.base_path or _base_ckpt_path()

    models: List[Dict[str, Any]] = [
        {
            "tag": str(args.base_tag),
            "path": str(base_path),
            "meta": {"algo": "base"},
        }
    ]

    for m in rl_models:
        models.append(
            {
                "tag": _tag_for(m.method, m.step),
                "path": _rl_ckpt_path(m.job_id, m.step),
                "meta": {"algo": m.method, "job_id": m.job_id, "step": int(m.step)},
            }
        )

    payload: Dict[str, Any] = {"models": models}
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[build_nips_models_file] wrote: {args.output_path}")


if __name__ == "__main__":
    main()
