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

"""Dataset loader for offline diagnosis.

The expected parquet schema (minimal):
  - prompt: a chat-template compatible list[dict], e.g.
      [{"role":"user","content":"..."}, ...]

Additional fields are passed through as metadata, such as:
  - reward_model: {"ground_truth": "2"}
  - data_source
  - ability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DiagnosisExample:
    """A single prompt example for diagnosis."""

    idx: int
    prompt: List[Dict[str, Any]]
    ground_truth: Optional[str]
    meta: Dict[str, Any]


def _normalize_prompt(obj: Any) -> List[Dict[str, Any]]:
    # parquet may store python objects as numpy arrays; keep it robust.
    if hasattr(obj, "tolist") and callable(obj.tolist):
        obj = obj.tolist()
    if not isinstance(obj, list):
        raise TypeError(f"prompt must be a list of messages, got: {type(obj)}")
    return obj


def load_parquet_dataset(path: str) -> List[DiagnosisExample]:
    """Load a parquet file into a list of DiagnosisExample."""
    import pandas as pd

    df = pd.read_parquet(path)
    if "prompt" not in df.columns:
        raise KeyError(f"Expected column 'prompt' in dataset. Found: {list(df.columns)}")

    examples: List[DiagnosisExample] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        prompt = _normalize_prompt(row.get("prompt"))

        rm = row.get("reward_model")
        gt: Optional[str] = None
        if isinstance(rm, dict):
            # you mentioned reward_model = {'ground_truth': '2'}
            v = rm.get("ground_truth")
            if v is not None:
                gt = str(v)

        meta = {
            "data_source": row.get("data_source"),
            "ability": row.get("ability"),
        }
        # keep the raw row for later extensions; avoid duplicating prompt.
        raw = dict(row)
        raw.pop("prompt", None)
        meta["raw"] = raw

        examples.append(DiagnosisExample(idx=i, prompt=prompt, ground_truth=gt, meta=meta))
    return examples
