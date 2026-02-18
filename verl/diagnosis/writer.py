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

"""Simple JSONL writer.

We keep this in a small module so later steps (mode clustering, metrics)
can reuse the same writer without duplicating IO logic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TextIO


class JsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fp: Optional[TextIO] = None

    def __enter__(self) -> "JsonlWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def open(self) -> None:
        if self._fp is not None:
            return
        self._fp = open(self.path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        if self._fp is None:
            self.open()
        assert self._fp is not None
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
