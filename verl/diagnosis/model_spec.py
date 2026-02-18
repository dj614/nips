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

"""Model specifications for multi-model offline diagnosis.

We keep this module tiny and dependency-light so it can be used from both:
  - scripts/run_mode_diagnosis.py (step 1 sampler)
  - scripts/summarize_modebank_diagnosis.py (step 2 metrics)

A models file (YAML/JSON) can define an arbitrary number of variants:

models:
  - tag: base
    path: Qwen/Qwen3-8B
  - tag: grpo
    path: /path/to/grpo/ckpt
    meta:
      algo: grpo
      step: 2000

"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


_TAG_SAFE_RE = re.compile(r"[^0-9a-zA-Z._-]+")


def sanitize_tag(tag: str) -> str:
    """Sanitize a model tag to be safe for filenames.

    We keep '.', '_', '-' and alphanumerics; everything else becomes '_'.
    """
    s = str(tag).strip()
    if not s:
        return "model"
    s = _TAG_SAFE_RE.sub("_", s)
    # Avoid leading dots / empty.
    s = s.strip("._") or "model"
    return s


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    path: str
    meta: Dict[str, Any] = field(default_factory=dict)


def _load_structured_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return json.loads(text)

    # Default to YAML for .yml/.yaml/unknown.
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        # Fall back to JSON (useful when yaml isn't installed).
        return json.loads(text)


def load_models_file(path: str) -> List[ModelSpec]:
    """Load a list[ModelSpec] from a YAML/JSON models file.

    Supported schemas:
      1) {"models": [{"tag": ..., "path": ..., "meta": {...}}, ...]}
      2) [{"tag": ..., "path": ...}, ...]

    For backward tolerance, we also accept keys: model_name, model_name_or_path.
    """
    obj = _load_structured_file(path)
    if obj is None:
        raise ValueError(f"Empty models file: {path}")

    items: Sequence[Any]
    if isinstance(obj, dict):
        items = obj.get("models") or []
    elif isinstance(obj, list):
        items = obj
    else:
        raise TypeError(f"Unsupported models file schema: {type(obj)}")

    out: List[ModelSpec] = []
    for it in items:
        if not isinstance(it, dict):
            raise TypeError(f"Each model entry must be a dict, got: {type(it)}")
        tag = it.get("tag") or it.get("name")
        if not tag:
            raise ValueError("Each model entry must have a non-empty 'tag'")

        path_ = it.get("path") or it.get("model_name") or it.get("model_name_or_path")
        if not path_:
            raise ValueError(f"Model '{tag}' is missing 'path'")

        meta = it.get("meta")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise TypeError(f"Model '{tag}' meta must be a dict, got: {type(meta)}")

        out.append(ModelSpec(tag=str(tag), path=str(path_), meta=dict(meta)))

    if not out:
        raise ValueError(f"No models found in models file: {path}")
    return out
