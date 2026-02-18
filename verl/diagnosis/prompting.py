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

"""Prompt utilities for diagnosis.

We keep this module intentionally small and dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def apply_hint(
    prompt_messages: Sequence[Dict[str, Any]],
    hint: str,
    *,
    injection: str = "suffix_last_user",
    template: str = "Use the following approach: {hint}.",
) -> List[Dict[str, Any]]:
    """Return a new prompt with a hint injected.

    Args:
        prompt_messages: Chat messages (list[dict]) compatible with HF chat template.
        hint: A short textual hint (mode instruction).
        injection: One of: suffix_last_user | append_user | append_system.
        template: A string template with `{hint}` placeholder.

    Returns:
        A new list of messages (shallow-copied dicts).
    """
    hint = str(hint)
    hint_text = template.format(hint=hint)

    # Ensure we return a fresh list (no in-place mutation of input).
    msgs: List[Dict[str, Any]] = [dict(m) for m in list(prompt_messages)]

    if injection == "append_user":
        msgs.append({"role": "user", "content": hint_text})
        return msgs

    if injection == "append_system":
        # System messages are most reliable when placed at the beginning.
        return [{"role": "system", "content": hint_text}] + msgs

    # Default: suffix the last user message.
    if injection != "suffix_last_user":
        raise ValueError(f"Unknown hint_injection: {injection}")

    # Find the last user message; if none, append a new user message.
    for i in range(len(msgs) - 1, -1, -1):
        if str(msgs[i].get("role", "")).lower() == "user":
            prev = str(msgs[i].get("content", ""))
            if prev and not prev.endswith("\n"):
                prev += "\n"
            msgs[i]["content"] = prev + "\n" + hint_text
            return msgs

    msgs.append({"role": "user", "content": hint_text})
    return msgs
