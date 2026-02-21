#!/usr/bin/env python

"""Generate per-problem mode hints via a vLLM-style /v1/completions API and write back to parquet.

This script prepares datasets for `scripts/run_mode_diagnosis.py`.
`verl/diagnosis/data.py` reads per-problem reachable mode hints from the dataset
column `mode` (expected to be a list[str]).

Example:

  # In-place update (recommended after backing up):
  QWEN_IP=... \
  python scripts/build_mode_hints_to_parquet.py \
    --data_path /primus_datasets/zmy/NIPS/test_data/amc23.parquet \
    --output_path /primus_datasets/zmy/NIPS/test_data/amc23.parquet

  # Or by TASK name and path template:
  QWEN_IP=... \
  python scripts/build_mode_hints_to_parquet.py \
    --task amc23 --data_path_template '/primus_datasets/zmy/NIPS/test_data/${TASK}.parquet'

Notes:
  - Endpoint is resolved from env `QWEN_IP`/`QWEN_IP` by default.
    Set it to either host[:port] (we append `/v1/completions`) or a full URL.
  - The completion API is expected to be compatible with the OpenAI-style
    `/v1/completions` schema returned by vLLM:
      request payload: {"prompt": ..., ...generation_params}
      response: {"choices": [{"text": ..., "finish_reason": ...}, ...]}
  - We write `mode` as list[str] (hints only) to match the existing diagnosis loader.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from verl.diagnosis.verifier import exact_match, extract_final_answer


# -------------------------
# Prompt template
# -------------------------

INSTRUCTION_PROMPT = (
    "You are a math problem solver and a \"mode generator\".\n\n"
    "Task:\n"
    "For EACH math problem provided below, generate 2 to 5 distinct solution approaches (modes).\n"
    "Each mode must include:\n"
    "(1) a one-sentence concise hint,\n"
    "(2) a concise solution,\n"
    "and the final answer MUST appear in LaTeX \\boxed{...} at the end of the solution.\n\n"
    "Requirements:\n"
    "- Produce 2–5 modes per problem.\n"
    "- Modes must be meaningfully different (different key idea/technique), not just cosmetic rewrites.\n"
    "- Keep the hint concise in one sentence.\n"
    "- Keep the solution concise but complete: show the key steps only; avoid long exposition.\n"
    "- Every mode must arrive at the SAME final answer. If you find inconsistency, fix it internally and output only consistent modes.\n"
    "- Put the final answer in \\boxed{...} at the end of EACH mode's solution (e.g., \"Final: \\boxed{42}\").\n"
    "- Do not add any extra commentary outside the requested output format.\n\n"
    "Output format (STRICT JSON only; no markdown, no extra text):\n"
    "Return a JSON array. Each element corresponds to one problem:\n"
    "[\n"
    "  {\n"
    "    \"problem\": \"<original problem text>\",\n"
    "    \"modes\": [\n"
    "      {\n"
    "        \"hint\": \"<short hint>\",\n"
    "        \"solution\": \"<concise solution ending with Final: \\\\boxed{...}>\"\n"
    "      },\n"
    "      ...\n"
    "    ]\n"
    "  },\n"
    "  ...\n"
    "]\n\n"
    "Now generate the modes for the following problems:\n"
    "<PROBLEMS>\n"
    "{problems}\n"
    "</PROBLEMS>\n"
)


# -------------------------
# vLLM-style /v1/completions client
# -------------------------


def send_request(
    prompt: str,
    ip: str = "localhost",
    port: str = "8000",
    url: str = "http://localhost:port/v1/completions",
    param_gen: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 300,
) -> Tuple[str, List[str]]:
    """Send a single /v1/completions request.

    This follows the reference calling pattern used elsewhere in the codebase.
    """

    if param_gen is None:
        param_gen = {}

    url = url.replace("localhost", str(ip))
    url = url.replace("port", str(port))
    payload = {
        "prompt": prompt,
        **param_gen,
    }

    with requests.post(url, json=payload, timeout=timeout_seconds) as response:
        try:
            response.raise_for_status()
        except Exception as e:
            print(f"[vllm_utils:send_request] Error: {repr(e)} - {response}")
            response = repr(e)
            finish_reason = "stop"
            return response, [finish_reason]

        result = response.json()
        text = result["choices"][-1].get("text", "")
        finish_reason = result["choices"][-1].get("finish_reason", "stop")
        return text, [finish_reason]


def get_model_output(
    prompt: str,
    param_gen: Dict[str, Any],
    ip: str = "localhost",
    port: str = "8000",
    url: str = "http://localhost:port/v1/completions",
    timeout_seconds: float = 300,
) -> Tuple[str, List[str]]:
    """Wrapper with a broad try/except to match existing tooling."""

    try:
        res, finish_reason = send_request(
            ip=ip,
            port=port,
            prompt=prompt,
            param_gen=param_gen,
            url=url,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:
        print(f"HTTP exception error: {error}")
        res, finish_reason = "", ["error"]
    return res, finish_reason


@dataclass
class PrimusClient:
    url: str
    timeout: Optional[float] = None
    # Generation params (top-level keys in the /v1/completions payload).
    param_gen: Optional[Dict[str, Any]] = None

    def _convert_openai_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if content is None:
                content = ""
            content = str(content)
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send a single completion request and return the `text` field."""

        prompt = self._convert_openai_messages_to_prompt(messages)

        # Default generation parameters (match the reference style and keep stable behavior).
        param_gen = dict(self.param_gen or {})
        param_gen.setdefault("top_k", 50)
        param_gen.setdefault("top_p", 0.8)
        param_gen.setdefault("temperature", 0.5)
        param_gen.setdefault("max_tokens", 1024)

        text, _finish = get_model_output(
            prompt=prompt,
            param_gen=param_gen,
            url=self.url,
            timeout_seconds=float(self.timeout) if self.timeout is not None else 300,
        )
        return text


# -------------------------
# JSON parsing helpers
# -------------------------


def _extract_json_array(text: str) -> List[Any]:
    """Parse and return the first top-level JSON array in `text`."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Fallback: locate a JSON array substring.
    i = text.find("[")
    j = text.rfind("]")
    if i >= 0 and j > i:
        sub = text[i : j + 1]
        obj = json.loads(sub)
        if isinstance(obj, list):
            return obj
    raise ValueError("response is not a JSON array")


def _hints_from_modes(obj: Any) -> List[str]:
    """Extract list[str] hints from the model JSON output."""
    if not isinstance(obj, list) or not obj:
        return []

    # We call one problem per request; take the first element.
    rec = obj[0]
    if not isinstance(rec, dict):
        return []
    modes = rec.get("modes")
    if not isinstance(modes, list):
        return []

    hints: List[str] = []
    for m in modes:
        if not isinstance(m, dict):
            continue
        h = m.get("hint")
        if h is None:
            continue
        h = str(h).strip()
        if not h:
            continue
        hints.append(h)
    # Stable dedupe while preserving order.
    seen = set()
    out: List[str] = []
    for h in hints:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def _ground_truth_from_row(df: "pd.DataFrame", i: int) -> Optional[str]:
    """Fetch reward_model.ground_truth from a dataframe row (robust to struct/flattened columns)."""
    # Common flattened column name.
    for key in ("reward_model.ground_truth", "ground_truth", "answer", "gt"):
        if key in df.columns:
            v = df.at[i, key]
            if v is None:
                continue
            try:
                if isinstance(v, float) and (v != v):  # NaN
                    continue
            except Exception:
                pass
            s = str(v).strip()
            if s:
                return s

    # Common nested struct column: reward_model is dict-like.
    if "reward_model" in df.columns:
        rm = df.at[i, "reward_model"]
        if hasattr(rm, "tolist") and callable(rm.tolist):
            rm = rm.tolist()
        if isinstance(rm, dict):
            v = rm.get("ground_truth")
            if v is not None:
                s = str(v).strip()
                return s if s else None
        if isinstance(rm, str) and rm.strip().startswith("{"):
            try:
                rmj = json.loads(rm)
                if isinstance(rmj, dict) and rmj.get("ground_truth") is not None:
                    s = str(rmj.get("ground_truth")).strip()
                    return s if s else None
            except Exception:
                pass

    return None


def _boxed_answers_from_modes(obj: Any) -> List[str]:
    """Extract boxed answers from each mode's solution in the model JSON output."""
    if not isinstance(obj, list) or not obj:
        return []
    rec = obj[0]
    if not isinstance(rec, dict):
        return []
    modes = rec.get("modes")
    if not isinstance(modes, list):
        return []
    answers: List[str] = []
    for m in modes:
        if not isinstance(m, dict):
            continue
        sol = m.get("solution")
        if sol is None:
            continue
        ans = extract_final_answer(str(sol), method="boxed", boxed_cmd="\\boxed")
        if ans is None:
            continue
        answers.append(str(ans).strip())
    return answers


def _verify_modes_match_ground_truth(obj: Any, ground_truth: str) -> Tuple[bool, str]:
    """Verify all modes share the same boxed answer and exact-match ground_truth.

    Returns:
      (ok, detail) where detail is a short string for logging.
    """
    gt = str(ground_truth).strip()
    if not gt:
        return False, "empty_gt"

    answers = _boxed_answers_from_modes(obj)
    if not answers:
        return False, "no_boxed"

    # Require all extracted answers to be identical (exact_match after normalization).
    base = answers[0]
    for a in answers[1:]:
        if not exact_match(str(a), str(base)):
            return False, f"inconsistent_boxed: {answers!r}"

    if not exact_match(str(base), gt):
        return False, f"mismatch_boxed: pred={base!r} gt={gt!r}"

    return True, "ok"


# -------------------------
# Dataset helpers
# -------------------------


def _normalize_prompt(obj: Any) -> List[Dict[str, Any]]:
    # parquet may store python objects as numpy arrays; keep it robust.
    if hasattr(obj, "tolist") and callable(obj.tolist):
        obj = obj.tolist()
    if not isinstance(obj, list):
        raise TypeError(f"prompt must be a list of messages, got: {type(obj)}")
    return obj


def _extract_problem_text(prompt: List[Dict[str, Any]]) -> str:
    """Heuristic: use the last user message content as the problem text."""
    for msg in reversed(prompt):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).lower() == "user":
            c = msg.get("content")
            if c is None:
                continue
            return str(c)
    # Fallback: join all content fields.
    parts = []
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("content") is not None:
            parts.append(str(msg.get("content")))
    return "\n".join(parts)


def _has_mode(v: Any) -> bool:
    if v is None:
        return False
    # pandas may store missing values as NaN (float)
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return False
    except Exception:
        pass
    if hasattr(v, "tolist") and callable(v.tolist):
        v = v.tolist()
    if isinstance(v, (list, tuple, set)):
        return len(v) > 0
    if isinstance(v, str):
        return bool(v.strip())
    return False


def _resolve_data_path(task: Optional[str], data_path: Optional[str], template: str) -> str:
    if data_path:
        return data_path
    if not task:
        raise SystemExit("Must provide either --data_path or --task")
    return template.replace("${TASK}", task)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-problem mode hints and write to parquet (column: mode)")

    ap.add_argument("--task", type=str, default=None, help="Optional task name to fill in --data_path_template")
    ap.add_argument("--data_path", type=str, default=None, help="Input parquet path (overrides --task)")
    ap.add_argument(
        "--data_path_template",
        type=str,
        default="/primus_datasets/zmy/NIPS/test_data/${TASK}.parquet",
        help="Template for --task (default matches user's NIPS paths)",
    )
    ap.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output parquet path. Default: overwrite input path.",
    )
    ap.add_argument(
        "--only_missing",
        action="store_true",
        help="Only generate for rows with empty/missing mode (default behavior)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mode even if present.",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap for debugging (process first N rows).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds between API calls to avoid rate limits.",
    )

    # Primus API config.
    ap.add_argument(
        "--primus_url",
        type=str,
        default=None,
        help=(
            "Completion endpoint URL. If omitted, use env QWEN_IP/QWEN_IP and append /v1/completions if needed."
        ),
    )
    ap.add_argument(
        "--model",
        type=str,
        default="qwen3-235b-instruct-trans-sg",
        help="(Deprecated) Kept for backward compatibility; the /v1/completions endpoint selects the served model.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional requests timeout (seconds)",
    )

    args = ap.parse_args()

    # Default behavior: only_missing unless --overwrite is explicitly set.
    only_missing = bool(args.only_missing) or not bool(args.overwrite)

    data_path = _resolve_data_path(args.task, args.data_path, args.data_path_template)
    out_path = args.output_path or data_path

    # Resolve endpoint from CLI or env (QWEN_IP preferred).
    primus_url = None
    if args.primus_url:
        primus_url = str(args.primus_url).strip()
    if not primus_url:
        qip = os.environ.get("QWEN_IP") or os.environ.get("QWEN_IP")
        if qip:
            u = str(qip).strip()
            if u:
                if not (u.startswith("http://") or u.startswith("https://")):
                    u = "http://" + u
                u = u.rstrip("/")
                if "/v1/" not in u:
                    u = u + "/v1/completions"
                primus_url = u

    if not primus_url:
        raise SystemExit(
            "Missing endpoint. Please export QWEN_IP (or QWEN_IP) as host[:port] or full URL, "
            "or pass --primus_url explicitly."
        )

    client = PrimusClient(url=primus_url, timeout=args.timeout)

    print(f"[mode_hints] loading parquet: {data_path}")
    df = pd.read_parquet(data_path).reset_index(drop=True)
    if "prompt" not in df.columns:
        raise SystemExit(f"Expected column 'prompt' in parquet. Found: {list(df.columns)}")
    if "mode" not in df.columns:
        df["mode"] = None

    n_total = len(df)
    if args.max_rows is not None:
        n_total = min(n_total, int(args.max_rows))
        df = df.iloc[:n_total].copy().reset_index(drop=True)

    n_done = 0
    n_skip = 0
    n_fail = 0

    for i in range(n_total):
        v = df.at[i, "mode"] if "mode" in df.columns else None
        if only_missing and _has_mode(v):
            n_skip += 1
            continue

        prompt_obj = _normalize_prompt(df.at[i, "prompt"])
        problem_text = _extract_problem_text(prompt_obj)
        if "{problems}" not in INSTRUCTION_PROMPT:
            raise RuntimeError("INSTRUCTION_PROMPT missing '{problems}' placeholder")
        user_prompt = INSTRUCTION_PROMPT.replace("{problems}", problem_text)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = client.complete(messages)
            arr = _extract_json_array(content)

            gt = _ground_truth_from_row(df, i)
            if gt is None:
                raise ValueError("missing reward_model.ground_truth for verification")

            ok, detail = _verify_modes_match_ground_truth(arr, gt)
            if not ok:
                raise ValueError(f"ground_truth verify failed: {detail}")

            hints = _hints_from_modes(arr)
            if not hints:
                raise ValueError("parsed 0 hints")
            df.at[i, "mode"] = hints
            n_done += 1
        except Exception as e:
            n_fail += 1
            # Keep going; leave the row unchanged for retry.
            print(f"[mode_hints] WARN row={i} failed: {type(e).__name__}: {e}")

        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(
                f"[mode_hints] progress {i + 1}/{n_total} | wrote={n_done} skipped={n_skip} failed={n_fail}",
                flush=True,
            )

    # Atomic write.
    tmp_path = f"{out_path}.tmp.{uuid.uuid4().hex[:8]}"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"[mode_hints] done. wrote={n_done} skipped={n_skip} failed={n_fail} -> {out_path}")


if __name__ == "__main__":
    main()
