# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config and CLI arg parsing for offline mode diagnosis.

This module intentionally keeps defaults compatible with the initial single-model
`scripts/run_mode_diagnosis.py` flow. New knobs are added for:
  - comparing multiple models (pre vs post RL)
  - reachability probing via per-problem mode hints
  - configurable exact-match answer extraction
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class DiagnosisConfig:
    """Config for offline response sampling diagnosis."""

    # (1) model name or local path
    model_name: str = "Qwen/Qwen3-8B"

    # Optional: compare multiple models (e.g., base vs GRPO). If provided, this
    # overrides `model_name`.
    model_names: List[str] = field(default_factory=list)
    # Optional: human-readable tags aligned with `model_names`.
    model_tags: List[str] = field(default_factory=list)

    # Optional: load model specs from a YAML/JSON file. If provided, this overrides
    # --model_name/--model_names/--model_tags.
    models_file: Optional[str] = None

    # (2) parquet path (local)
    data_path: str = "dapo_math_17k.parquet"

    # (3)-(6) sampling hyper-params
    high_temp: float = 1.0
    low_temp: float = 0.2
    high_temp_samples: int = 32
    low_temp_samples: int = 8

    # (6.1) reachability probing hyper-params (for hinted prompts)
    mid_temp: float = 0.7
    mid_temp_samples_per_hint: int = 4

    # (6.2) sampling controls
    top_p: float = 0.95
    repetition_penalty: Optional[float] = None

    # (7) response max length in tokens (mapped to max_new_tokens)
    max_resp_length: int = 4096

    # (7.1) exact-match answer extraction (default keeps backward behavior)
    # Supported: gsm8k, boxed, hash_mark, boxed_or_hash_mark
    answer_extract: str = "boxed"
    boxed_cmd: str = "\\boxed"

    # (7.2) hint injection
    hint_template: str = "Use the following approach: {hint}."
    hint_injection: str = "suffix_last_user"  # suffix_last_user|append_user|append_system

    # (8) output jsonl path
    output_path: str = "diagnosis.jsonl"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Offline mode diagnosis sampler: generate multiple responses per prompt "
            "using two temperatures (high/low)."
        )
    )
    p.add_argument("--model_name", type=str, default=DiagnosisConfig.model_name)
    p.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional: multiple model names/paths to compare. Overrides --model_name.",
    )
    p.add_argument(
        "--model_tags",
        type=str,
        nargs="+",
        default=None,
        help="Optional: tags aligned with --model_names (e.g., base grpo).",
    )
    p.add_argument(
        "--models_file",
        type=str,
        default=None,
        help=(
            "Optional: YAML/JSON file specifying a list of models to compare. "
            "If provided, overrides --model_name/--model_names/--model_tags."
        ),
    )
    p.add_argument("--data_path", type=str, default=DiagnosisConfig.data_path)
    p.add_argument("--high_temp", type=float, default=DiagnosisConfig.high_temp)
    p.add_argument("--low_temp", type=float, default=DiagnosisConfig.low_temp)
    p.add_argument("--high_temp_samples", type=int, default=DiagnosisConfig.high_temp_samples)
    p.add_argument("--low_temp_samples", type=int, default=DiagnosisConfig.low_temp_samples)
    p.add_argument("--mid_temp", type=float, default=DiagnosisConfig.mid_temp)
    p.add_argument("--mid_temp_samples_per_hint", type=int, default=DiagnosisConfig.mid_temp_samples_per_hint)
    p.add_argument("--top_p", type=float, default=DiagnosisConfig.top_p)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--max_resp_length", type=int, default=DiagnosisConfig.max_resp_length)
    p.add_argument(
        "--answer_extract",
        type=str,
        default=DiagnosisConfig.answer_extract,
        help="Answer extraction method for exact-match. One of: gsm8k, boxed, hash_mark, boxed_or_hash_mark.",
    )
    p.add_argument(
        "--boxed_cmd",
        type=str,
        default=DiagnosisConfig.boxed_cmd,
        help="Box command for boxed extraction (e.g., \\boxed or \\fbox).",
    )
    p.add_argument("--hint_template", type=str, default=DiagnosisConfig.hint_template)
    p.add_argument(
        "--hint_injection",
        type=str,
        default=DiagnosisConfig.hint_injection,
        help="How to inject hints into the prompt: suffix_last_user|append_user|append_system.",
    )
    p.add_argument("--output_path", type=str, default=DiagnosisConfig.output_path)
    return p


def config_from_args(args: argparse.Namespace) -> DiagnosisConfig:
    model_names: List[str] = []
    model_tags: List[str] = []
    if getattr(args, "model_names", None):
        model_names = list(args.model_names)
        if getattr(args, "model_tags", None):
            model_tags = list(args.model_tags)

    return DiagnosisConfig(
        model_name=args.model_name,
        model_names=model_names,
        model_tags=model_tags,
        models_file=getattr(args, "models_file", None),
        data_path=args.data_path,
        high_temp=args.high_temp,
        low_temp=args.low_temp,
        high_temp_samples=args.high_temp_samples,
        low_temp_samples=args.low_temp_samples,
        mid_temp=args.mid_temp,
        mid_temp_samples_per_hint=args.mid_temp_samples_per_hint,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_resp_length=args.max_resp_length,
        answer_extract=args.answer_extract,
        boxed_cmd=args.boxed_cmd,
        hint_template=args.hint_template,
        hint_injection=args.hint_injection,
        output_path=args.output_path,
    )
