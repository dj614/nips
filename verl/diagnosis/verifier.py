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

"""Correctness verification for offline diagnosis.

Step 2 requirement:
  - Extract a final answer from each sampled response.
  - Use exact-match with reward_model.ground_truth.

This module intentionally stays lightweight. It reuses existing answer
extraction utilities when available (e.g., GSM8K-style "####" answers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VerifyResult:
    extracted: Optional[str]
    correct: Optional[bool]
    method: str


def _normalize(s: str) -> str:
    # Keep it minimal and deterministic: strip, remove commas and surrounding whitespace.
    # (Your dataset's ground_truth is typically a compact numeric string.)
    return s.strip().replace(",", "")


def extract_final_answer(text: str) -> Optional[str]:
    """Extract a final answer string from a model response.

    Strategy:
      1) GSM8K strict: look for "#### <number>".
      2) GSM8K flexible: last valid number in the tail.
      3) Fallback: None.

    Notes:
      - We clip internally in the gsm8k extractor to avoid regex slowdowns on long outputs.
    """
    try:
        from verl.utils.reward_score.gsm8k import extract_solution

        ans = extract_solution(solution_str=text, method="strict")
        if ans is not None:
            return ans
        ans = extract_solution(solution_str=text, method="flexible")
        return ans
    except Exception:
        # If reward_score utils are not available for some reason, fail gracefully.
        return None


def exact_match(pred: str, gt: str) -> bool:
    return _normalize(pred) == _normalize(gt)


def verify_response(response_text: str, ground_truth: Optional[str]) -> VerifyResult:
    """Verify a single response against the ground truth.

    Returns:
      VerifyResult(extracted, correct, method)

    If ground_truth is None, correct will be None.
    """
    extracted = extract_final_answer(response_text)
    if ground_truth is None:
        return VerifyResult(extracted=extracted, correct=None, method="no_gt")

    if extracted is None:
        return VerifyResult(extracted=None, correct=False, method="no_extract")

    ok = exact_match(extracted, str(ground_truth))
    return VerifyResult(extracted=extracted, correct=ok, method="exact")
