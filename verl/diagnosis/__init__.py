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

"""Lightweight offline diagnosis utilities.

This package is intentionally self-contained and does NOT depend on
verl's trainer/rollout stack. It is meant for offline experiments such as
multi-sample generation + downstream analysis.
"""

from .config import DiagnosisConfig, build_argparser
from .data import load_parquet_dataset
from .metrics import DiagnosisAggregator
from .mode import ModeAssignResult, ModeDetector, NullModeDetector
from .perturb import NullPerturber, Perturber
from .sampler import load_model_and_tokenizer, sample_n
from .verifier import extract_final_answer, exact_match, verify_response
from .writer import JsonlWriter

__all__ = [
    "DiagnosisConfig",
    "build_argparser",
    "load_parquet_dataset",
    "load_model_and_tokenizer",
    "sample_n",
    "extract_final_answer",
    "exact_match",
    "verify_response",
    "JsonlWriter",
    "ModeAssignResult",
    "ModeDetector",
    "NullModeDetector",
    "Perturber",
    "NullPerturber",
    "DiagnosisAggregator",
]
