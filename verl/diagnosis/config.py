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

"""Config and CLI arg parsing for offline mode diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosisConfig:
    """Config for offline response sampling diagnosis."""

    # (1) model name or local path
    model_name: str = "Qwen/Qwen3-8B"

    # (2) parquet path (local)
    data_path: str = "dapo_math_17k.parquet"

    # (3)-(6) sampling hyper-params
    high_temp: float = 1.0
    low_temp: float = 0.2
    high_temp_samples: int = 32
    low_temp_samples: int = 8

    # (7) response max length in tokens (mapped to max_new_tokens)
    max_resp_length: int = 4096

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
    p.add_argument("--data_path", type=str, default=DiagnosisConfig.data_path)
    p.add_argument("--high_temp", type=float, default=DiagnosisConfig.high_temp)
    p.add_argument("--low_temp", type=float, default=DiagnosisConfig.low_temp)
    p.add_argument("--high_temp_samples", type=int, default=DiagnosisConfig.high_temp_samples)
    p.add_argument("--low_temp_samples", type=int, default=DiagnosisConfig.low_temp_samples)
    p.add_argument("--max_resp_length", type=int, default=DiagnosisConfig.max_resp_length)
    p.add_argument("--output_path", type=str, default=DiagnosisConfig.output_path)
    return p


def config_from_args(args: argparse.Namespace) -> DiagnosisConfig:
    return DiagnosisConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        high_temp=args.high_temp,
        low_temp=args.low_temp,
        high_temp_samples=args.high_temp_samples,
        low_temp_samples=args.low_temp_samples,
        max_resp_length=args.max_resp_length,
        output_path=args.output_path,
    )
