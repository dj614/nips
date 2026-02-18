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

"""Offline sampler for multi-mode diagnosis (Step 2).

This script:
  1) Loads a parquet dataset containing chat-style prompts.
  2) Loads a HF causal LM from `--model_name`.
  3) For each prompt, samples responses with:
       - high temperature (diverse)
       - low temperature (stable)
  4) Extracts final answers and verifies correctness using exact-match with
     `reward_model.ground_truth`.
  5) Streams results to a jsonl file (one record per prompt), and prints a
     small summary to terminal.

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

import os
from typing import Any, Dict, List, Optional


def _is_solved(verifs: List[Dict[str, Any]]) -> Optional[bool]:
    """Return True if any sample is correct.

    If all samples are missing ground truth (correct=None), return None.
    """
    any_gt = any(v.get("correct") is not None for v in verifs)
    if not any_gt:
        return None
    return any(v.get("correct") is True for v in verifs)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    from verl.diagnosis.config import build_argparser, config_from_args
    from verl.diagnosis.data import load_parquet_dataset
    from verl.diagnosis.metrics import DiagnosisAggregator
    from verl.diagnosis.mode import NullModeDetector
    from verl.diagnosis.sampler import load_model_and_tokenizer, sample_n
    from verl.diagnosis.verifier import verify_response
    from verl.diagnosis.writer import JsonlWriter

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

    mode_detector = NullModeDetector()
    agg = DiagnosisAggregator()

    out_dir = os.path.dirname(cfg.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_eval = 0
    high_solved = 0
    low_solved = 0
    any_solved = 0

    high_total = 0
    low_total = 0
    high_correct = 0
    low_correct = 0

    print(f"[mode_diagnosis] writing diagnosis -> {cfg.output_path}")
    with JsonlWriter(cfg.output_path) as writer:
        for ex in examples:
            gt = ex.ground_truth
            if gt is not None:
                n_eval += 1

            high_texts = sample_n(
                model,
                tokenizer,
                ex.prompt,
                temperature=cfg.high_temp,
                n=cfg.high_temp_samples,
                max_new_tokens=cfg.max_resp_length,
            )
            low_texts = sample_n(
                model,
                tokenizer,
                ex.prompt,
                temperature=cfg.low_temp,
                n=cfg.low_temp_samples,
                max_new_tokens=cfg.max_resp_length,
            )

            high_verifs: List[Dict[str, Any]] = []
            for t in high_texts:
                r = verify_response(t, gt)
                m = mode_detector.assign(ex.prompt, t, extracted_answer=r.extracted, meta=ex.meta)
                high_verifs.append(
                    {
                        "text": t,
                        "final_answer": r.extracted,
                        "correct": r.correct,
                        "verify_method": r.method,
                        "mode_id": m.mode_id,
                        "mode_info": m.info,
                    }
                )
            low_verifs: List[Dict[str, Any]] = []
            for t in low_texts:
                r = verify_response(t, gt)
                m = mode_detector.assign(ex.prompt, t, extracted_answer=r.extracted, meta=ex.meta)
                low_verifs.append(
                    {
                        "text": t,
                        "final_answer": r.extracted,
                        "correct": r.correct,
                        "verify_method": r.method,
                        "mode_id": m.mode_id,
                        "mode_info": m.info,
                    }
                )

            high_solved_flag = _is_solved(high_verifs)
            low_solved_flag = _is_solved(low_verifs)
            any_solved_flag: Optional[bool]
            if high_solved_flag is None and low_solved_flag is None:
                any_solved_flag = None
            else:
                any_solved_flag = bool(high_solved_flag) or bool(low_solved_flag)

            if gt is not None:
                if high_solved_flag:
                    high_solved += 1
                if low_solved_flag:
                    low_solved += 1
                if any_solved_flag:
                    any_solved += 1

                # sample-level accuracy
                high_total += len(high_verifs)
                low_total += len(low_verifs)
                high_correct += sum(1 for v in high_verifs if v.get("correct") is True)
                low_correct += sum(1 for v in low_verifs if v.get("correct") is True)

            record: Dict[str, Any] = {
                "type": "sample",
                "idx": ex.idx,
                "prompt": ex.prompt,
                "ground_truth": gt,
                "meta": ex.meta,
                "samples": {
                    "high_temp": {
                        "temperature": cfg.high_temp,
                        "n": cfg.high_temp_samples,
                        "solved": high_solved_flag,
                        "responses": high_verifs,
                    },
                    "low_temp": {
                        "temperature": cfg.low_temp,
                        "n": cfg.low_temp_samples,
                        "solved": low_solved_flag,
                        "responses": low_verifs,
                    },
                    "any_solved": any_solved_flag,
                },
            }
            writer.write(record)
            agg.update(record)

            if (ex.idx + 1) % 10 == 0 or (ex.idx + 1) == len(examples):
                print(f"[mode_diagnosis] processed {ex.idx + 1}/{len(examples)}")

        # Emit a final summary record for downstream analysis.
        summary_mode = agg.finalize()
        summary_record: Dict[str, Any] = {
            "type": "summary",
            "config": {
                "model_name": cfg.model_name,
                "data_path": cfg.data_path,
                "high_temp": cfg.high_temp,
                "low_temp": cfg.low_temp,
                "high_temp_samples": cfg.high_temp_samples,
                "low_temp_samples": cfg.low_temp_samples,
                "max_resp_length": cfg.max_resp_length,
            },
            "solved": {
                "n_eval": n_eval,
                "high_solved": high_solved,
                "low_solved": low_solved,
                "any_solved": any_solved,
                "high_sample_acc": (high_correct / high_total) if high_total > 0 else None,
                "low_sample_acc": (low_correct / low_total) if low_total > 0 else None,
            },
            "mode_detector": getattr(mode_detector, "name", type(mode_detector).__name__),
            "mode_metrics": summary_mode,
        }
        writer.write(summary_record)

    # Summary
    if n_eval == 0:
        print("[mode_diagnosis] done (no ground_truth available, skipped summary)")
        return

    def _pct(a: int, b: int) -> float:
        return (100.0 * a / b) if b > 0 else 0.0

    print("[mode_diagnosis] summary:")
    print(f"  #eval questions     : {n_eval}")
    print(f"  high solved@{cfg.high_temp_samples}: {high_solved}/{n_eval} ({_pct(high_solved, n_eval):.2f}%)")
    print(f"  low  solved@{cfg.low_temp_samples}: {low_solved}/{n_eval} ({_pct(low_solved, n_eval):.2f}%)")
    print(f"  any  solved         : {any_solved}/{n_eval} ({_pct(any_solved, n_eval):.2f}%)")
    print(f"  high sample acc     : {high_correct}/{high_total} ({_pct(high_correct, high_total):.2f}%)")
    print(f"  low  sample acc     : {low_correct}/{low_total} ({_pct(low_correct, low_total):.2f}%)")

    # Step 3: mode complementarity summary (placeholder with NullModeDetector)
    try:
        comp = summary_mode.get("complementarity", {})
        exist = summary_mode.get("mode_existence", {})
        print("[mode_diagnosis] mode metrics:")
        print(f"  acc_union       : {comp.get('acc_union', 0.0):.4f}")
        print(f"  acc_best_single : {comp.get('acc_best_single', 0.0):.4f} (best={comp.get('best_mode')})")
        print(f"  gap             : {comp.get('gap', 0.0):.4f}")
        print(f"  frac multi-modes: {exist.get('frac_multi', 0.0):.4f} ({exist.get('multi_correct_modes', 0)}/{summary_mode.get('n_eval', 0)})")

        jacc = summary_mode.get("jaccard", [])
        if jacc:
            print("  lowest jaccard pairs (top 5):")
            for d in jacc[:5]:
                print(f"    {d['mode_a']} vs {d['mode_b']}: jac={d['jaccard']:.3f} (|∩|={d['intersection']}, |∪|={d['union']})")
    except Exception as e:
        print(f"[mode_diagnosis] warning: failed to print mode metrics: {e}")


if __name__ == "__main__":
    main()
