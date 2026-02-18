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

"""Offline sampler for multi-mode diagnosis.

This script:
  1) Loads a parquet dataset containing chat-style prompts.
  2) Loads one (or multiple) HF causal LM(s) from `--model_name`/`--model_names`.
  3) For each prompt, samples responses with:
       - high temperature (diverse)
       - low temperature (stable)
       - (optional) mid temperature under per-problem mode hints (reachability)
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
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _is_solved(verifs: List[Dict[str, Any]]) -> Optional[bool]:
    """Return True if any sample is correct.

    If all samples are missing ground truth (correct=None), return None.
    """
    any_gt = any(v.get("correct") is not None for v in verifs)
    if not any_gt:
        return None
    return any(v.get("correct") is True for v in verifs)


def _derive_model_tags(model_names: Sequence[str], model_tags: Sequence[str]) -> List[str]:
    if model_tags and len(model_tags) == len(model_names):
        return list(model_tags)
    # Stable default: use basename of local path or repo id tail.
    out: List[str] = []
    for m in model_names:
        s = str(m).rstrip("/")
        out.append(os.path.basename(s) or s.split("/")[-1])
    return out


def _output_path_for_tag(output_path: str, tag: str, multi: bool) -> str:
    if not multi:
        return output_path
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".jsonl"
    return f"{root}.{tag}{ext}"


def _run_one_model(model_name_or_path: str, model_tag: str, cfg, examples) -> Tuple[str, Dict[str, Any]]:
    """Run sampling for a single model and write a jsonl output."""
    from verl.diagnosis.metrics import DiagnosisAggregator
    from verl.diagnosis.mode import NullModeDetector
    from verl.diagnosis.prompting import apply_hint
    from verl.diagnosis.sampler import load_model_and_tokenizer, sample_n
    from verl.diagnosis.verifier import verify_response
    from verl.diagnosis.writer import JsonlWriter

    print(f"[mode_diagnosis] loading model: {model_name_or_path} (tag={model_tag})")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    mode_detector = NullModeDetector()
    agg = DiagnosisAggregator()

    n_eval = 0
    high_solved = 0
    low_solved = 0
    any_solved = 0
    reach_any_solved = 0

    high_total = 0
    low_total = 0
    reach_total = 0
    high_correct = 0
    low_correct = 0
    reach_correct = 0

    out_path = cfg.output_path
    print(f"[mode_diagnosis] writing diagnosis -> {out_path}")
    with JsonlWriter(out_path) as writer:
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
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )
            low_texts = sample_n(
                model,
                tokenizer,
                ex.prompt,
                temperature=cfg.low_temp,
                n=cfg.low_temp_samples,
                max_new_tokens=cfg.max_resp_length,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )

            high_verifs: List[Dict[str, Any]] = []
            for t in high_texts:
                r = verify_response(t, gt, extract_method=cfg.answer_extract, boxed_cmd=cfg.boxed_cmd)
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
                r = verify_response(t, gt, extract_method=cfg.answer_extract, boxed_cmd=cfg.boxed_cmd)
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

            # Reachability probing: append each hint and sample at mid temperature.
            reach_per_hint: List[Dict[str, Any]] = []
            reach_any_solved_flag: Optional[bool] = None
            if getattr(ex, "hints", None):
                hint_solved_flags: List[Optional[bool]] = []
                for hint in ex.hints:
                    prompt_h = apply_hint(
                        ex.prompt,
                        hint,
                        injection=cfg.hint_injection,
                        template=cfg.hint_template,
                    )
                    mid_texts = sample_n(
                        model,
                        tokenizer,
                        prompt_h,
                        temperature=cfg.mid_temp,
                        n=cfg.mid_temp_samples_per_hint,
                        max_new_tokens=cfg.max_resp_length,
                        top_p=cfg.top_p,
                        repetition_penalty=cfg.repetition_penalty,
                    )
                    mid_verifs: List[Dict[str, Any]] = []
                    for t in mid_texts:
                        r = verify_response(t, gt, extract_method=cfg.answer_extract, boxed_cmd=cfg.boxed_cmd)
                        mid_verifs.append(
                            {
                                "text": t,
                                "final_answer": r.extracted,
                                "correct": r.correct,
                                "verify_method": r.method,
                            }
                        )
                    solved_flag = _is_solved(mid_verifs)
                    hint_solved_flags.append(solved_flag)
                    reach_per_hint.append(
                        {
                            "hint": hint,
                            "solved": solved_flag,
                            "responses": mid_verifs,
                        }
                    )

                    if gt is not None:
                        reach_total += len(mid_verifs)
                        reach_correct += sum(1 for v in mid_verifs if v.get("correct") is True)

                if any(s is not None for s in hint_solved_flags):
                    reach_any_solved_flag = any(s is True for s in hint_solved_flags)
                else:
                    reach_any_solved_flag = None

                if gt is not None and reach_any_solved_flag:
                    reach_any_solved += 1

            record: Dict[str, Any] = {
                "type": "sample",
                "idx": ex.idx,
                "model_tag": model_tag,
                "model_name_or_path": model_name_or_path,
                "prompt": ex.prompt,
                "ground_truth": gt,
                "meta": ex.meta,
                "hints": list(getattr(ex, "hints", []) or []),
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
                "reach": {
                    "temperature": cfg.mid_temp,
                    "n_per_hint": cfg.mid_temp_samples_per_hint,
                    "hint_template": cfg.hint_template,
                    "hint_injection": cfg.hint_injection,
                    "any_solved": reach_any_solved_flag,
                    "per_hint": reach_per_hint,
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
                "model_tag": model_tag,
                "model_name_or_path": model_name_or_path,
                "data_path": cfg.data_path,
                "high_temp": cfg.high_temp,
                "low_temp": cfg.low_temp,
                "mid_temp": cfg.mid_temp,
                "high_temp_samples": cfg.high_temp_samples,
                "low_temp_samples": cfg.low_temp_samples,
                "mid_temp_samples_per_hint": cfg.mid_temp_samples_per_hint,
                "top_p": cfg.top_p,
                "repetition_penalty": cfg.repetition_penalty,
                "max_resp_length": cfg.max_resp_length,
                "answer_extract": cfg.answer_extract,
                "boxed_cmd": cfg.boxed_cmd,
                "hint_template": cfg.hint_template,
                "hint_injection": cfg.hint_injection,
            },
            "solved": {
                "n_eval": n_eval,
                "high_solved": high_solved,
                "low_solved": low_solved,
                "any_solved": any_solved,
                "reach_any_solved": reach_any_solved,
                "high_sample_acc": (high_correct / high_total) if high_total > 0 else None,
                "low_sample_acc": (low_correct / low_total) if low_total > 0 else None,
                "reach_sample_acc": (reach_correct / reach_total) if reach_total > 0 else None,
            },
            "mode_detector": getattr(mode_detector, "name", type(mode_detector).__name__),
            "mode_metrics": summary_mode,
        }
        writer.write(summary_record)

    # Print a compact per-model summary (keeps old script behavior).
    if n_eval > 0:
        def _pct(a: int, b: int) -> float:
            return (100.0 * a / b) if b > 0 else 0.0

        print(f"[mode_diagnosis] summary ({model_tag}):")
        print(f"  #eval questions     : {n_eval}")
        print(f"  high solved@{cfg.high_temp_samples}: {high_solved}/{n_eval} ({_pct(high_solved, n_eval):.2f}%)")
        print(f"  low  solved@{cfg.low_temp_samples}: {low_solved}/{n_eval} ({_pct(low_solved, n_eval):.2f}%)")
        print(f"  any  solved         : {any_solved}/{n_eval} ({_pct(any_solved, n_eval):.2f}%)")
        print(f"  reach any solved    : {reach_any_solved}/{n_eval} ({_pct(reach_any_solved, n_eval):.2f}%)")
        print(f"  high sample acc     : {high_correct}/{high_total} ({_pct(high_correct, high_total):.2f}%)")
        print(f"  low  sample acc     : {low_correct}/{low_total} ({_pct(low_correct, low_total):.2f}%)")
        if reach_total > 0:
            print(f"  reach sample acc    : {reach_correct}/{reach_total} ({_pct(reach_correct, reach_total):.2f}%)")

    return out_path, summary_record


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    from verl.diagnosis.config import build_argparser, config_from_args
    from verl.diagnosis.data import load_parquet_dataset

    parser = build_argparser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    print("[mode_diagnosis] config:")
    print(cfg)

    print(f"[mode_diagnosis] loading dataset: {cfg.data_path}")
    examples = load_parquet_dataset(cfg.data_path)
    print(f"[mode_diagnosis] #examples: {len(examples)}")

    model_names = cfg.model_names if cfg.model_names else [cfg.model_name]
    model_tags = _derive_model_tags(model_names, cfg.model_tags)

    multi = len(model_names) > 1
    out_dir = os.path.dirname(cfg.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Run each model into its own jsonl file for clean downstream comparisons.
    summaries: List[Dict[str, Any]] = []
    for model_name_or_path, model_tag in zip(model_names, model_tags):
        cfg_one = cfg
        # Rebind the output path per model when in multi-model mode.
        cfg_one = type(cfg)(**{**cfg.__dict__, "output_path": _output_path_for_tag(cfg.output_path, model_tag, multi)})
        out_path, summary = _run_one_model(model_name_or_path, model_tag, cfg_one, examples)
        summaries.append({"tag": model_tag, "output_path": out_path, "summary": summary})

    # If multiple models, print a short side-by-side comparison of reach/nat solved.
    if summaries:
        print("[mode_diagnosis] done. outputs:")
        for s in summaries:
            solved = (s.get("summary") or {}).get("solved", {})
            print(
                f"  - {s['tag']}: {s['output_path']} "
                f"(any_solved={solved.get('any_solved')}/{solved.get('n_eval')}, "
                f"reach_any_solved={solved.get('reach_any_solved')}/{solved.get('n_eval')})"
            )
    return


if __name__ == "__main__":
    main()
