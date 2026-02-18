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

"""Offline sampler for multi-mode diagnosis (Step 1).

This script:
  1) Loads a parquet dataset containing chat-style prompts.
  2) Loads a HF causal LM from `--model_name`.
  3) For each prompt, samples responses with:
       - high temperature (diverse)
       - low temperature (stable)
  4) Dumps raw generations to a jsonl file (one record per prompt).

It intentionally does NOT do correctness verification or mode clustering yet.
Those will be implemented in later steps.

Example:

  python scripts/run_mode_diagnosis.py \
    --model_name Qwen/Qwen3-8B \
    --data_path dapo_math_17k.parquet \
    --high_temp 1.0 --high_temp_samples 32 \
    --low_temp 0.2 --low_temp_samples 8 \
    --max_resp_length 4096 \
    --output_path diagnosis.jsonl
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def _write_jsonl(fp, record: Dict[str, Any]) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    # Make local `verl/` importable when running from repo root.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    from verl.diagnosis.config import build_argparser, config_from_args
    from verl.diagnosis.data import load_parquet_dataset
    from verl.diagnosis.sampler import load_model_and_tokenizer, sample_n

    parser = build_argparser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    print("[mode_diagnosis] config:")
    print(cfg)

    print(f"[mode_diagnosis] loading dataset: {cfg.data_path}")
    examples = load_parquet_dataset(cfg.data_path)
    print(f"[mode_diagnosis] #examples: {len(examples)}")

    print(f"[mode_diagnosis] loading model: {cfg.model_name}")
    model, tokenizer = load_model_and_tokenizer(cfg.model_name)

    out_dir = os.path.dirname(cfg.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[mode_diagnosis] writing generations -> {cfg.output_path}")
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            high_responses = sample_n(
                model,
                tokenizer,
                ex.prompt,
                temperature=cfg.high_temp,
                n=cfg.high_temp_samples,
                max_new_tokens=cfg.max_resp_length,
            )
            low_responses = sample_n(
                model,
                tokenizer,
                ex.prompt,
                temperature=cfg.low_temp,
                n=cfg.low_temp_samples,
                max_new_tokens=cfg.max_resp_length,
            )

            record: Dict[str, Any] = {
                "type": "sample",
                "idx": ex.idx,
                "prompt": ex.prompt,
                "ground_truth": ex.ground_truth,
                "meta": ex.meta,
                "samples": {
                    "high_temp": {
                        "temperature": cfg.high_temp,
                        "n": cfg.high_temp_samples,
                        "responses": high_responses,
                    },
                    "low_temp": {
                        "temperature": cfg.low_temp,
                        "n": cfg.low_temp_samples,
                        "responses": low_responses,
                    },
                },
            }
            _write_jsonl(f, record)

            # lightweight progress log
            if (ex.idx + 1) % 10 == 0 or (ex.idx + 1) == len(examples):
                print(f"[mode_diagnosis] processed {ex.idx + 1}/{len(examples)}")

    print("[mode_diagnosis] done")


if __name__ == "__main__":
    main()
