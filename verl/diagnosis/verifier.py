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


def _extract_gsm8k(text: str) -> Optional[str]:
    """GSM8K-style extraction: "#### ..." with a numeric fallback."""
    try:
        from verl.utils.reward_score.gsm8k import extract_solution

        ans = extract_solution(solution_str=text, method="strict")
        if ans is not None:
            return ans
        return extract_solution(solution_str=text, method="flexible")
    except Exception:
        return None


def _extract_hash_mark(text: str) -> Optional[str]:
    """Extract answer after the last "####" marker."""
    if "####" not in text:
        return None
    tail = text.split("####")[-1]
    # Take the first non-empty line.
    for line in tail.splitlines():
        s = line.strip()
        if s:
            return s
    s = tail.strip()
    return s if s else None


def _last_braced_expression(text: str, key: str) -> Optional[str]:
    """Return the last braced expression starting with `key`, e.g. "\\boxed{".

    Returns the full substring `key...}` including braces.
    """
    idx = text.rfind(key)
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        elif text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def _extract_boxed(text: str, boxed_cmd: str = "\\boxed") -> Optional[str]:
    """Extract the content inside the last \boxed{...} (or other boxed_cmd)."""
    boxed_cmd = str(boxed_cmd).strip()
    if not boxed_cmd:
        boxed_cmd = "\\boxed"
    if not boxed_cmd.startswith("\\"):
        boxed_cmd = "\\" + boxed_cmd

    key = boxed_cmd + "{"
    s = _last_braced_expression(text, key)
    if s is None:
        return None

    # Strip the prefix and trailing brace.
    if not s.startswith(key) or not s.endswith("}"):
        return None
    return s[len(key) : -1].strip()


def extract_final_answer(text: str, method: str = "gsm8k", boxed_cmd: str = "\\boxed") -> Optional[str]:
    """Extract a final answer string from a model response.

    Supported methods:
      - gsm8k: "####"-style GSM8K extraction
      - boxed: last \boxed{...}
      - hash_mark: last "#### ..." line
      - boxed_or_hash_mark: prefer boxed, else hash_mark, else gsm8k
    """
    method = str(method or "gsm8k").strip().lower()

    if method == "boxed":
        return _extract_boxed(text, boxed_cmd=boxed_cmd)
    if method == "hash_mark":
        return _extract_hash_mark(text)
    if method == "boxed_or_hash_mark":
        ans = _extract_boxed(text, boxed_cmd=boxed_cmd)
        if ans is not None:
            return ans
        ans = _extract_hash_mark(text)
        if ans is not None:
            return ans
        return _extract_gsm8k(text)
    if method == "gsm8k":
        return _extract_gsm8k(text)

    # Fallback: try something reasonable.
    ans = _extract_boxed(text, boxed_cmd=boxed_cmd)
    if ans is not None:
        return ans
    ans = _extract_hash_mark(text)
    if ans is not None:
        return ans
    return _extract_gsm8k(text)


def exact_match(pred: str, gt: str) -> bool:
    return _normalize(pred) == _normalize(gt)


def verify_response(
    response_text: str,
    ground_truth: Optional[str],
    *,
    extract_method: str = "gsm8k",
    boxed_cmd: str = "\\boxed",
) -> VerifyResult:
    """Verify a single response against the ground truth.

    Returns:
      VerifyResult(extracted, correct, method)

    If ground_truth is None, correct will be None.
    """
    extracted = extract_final_answer(response_text, method=extract_method, boxed_cmd=boxed_cmd)
    if ground_truth is None:
        return VerifyResult(extracted=extracted, correct=None, method="no_gt")

    if extracted is None:
        return VerifyResult(extracted=None, correct=False, method="no_extract")

    ok = exact_match(extracted, str(ground_truth))
    return VerifyResult(extracted=extracted, correct=ok, method="exact")
