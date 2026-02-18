#!/usr/bin/env python

"""Generate per-problem mode hints via Primus Qwen API and write back to parquet.

This script prepares datasets for `scripts/run_mode_diagnosis.py`.
`verl/diagnosis/data.py` reads per-problem reachable mode hints from the dataset
column `mode` (expected to be a list[str]).

Example:

  # In-place update (recommended after backing up):
  PRIMUS_API_KEY=... \
  python scripts/build_mode_hints_to_parquet.py \
    --data_path /primus_datasets/zmy/GARO/test_data/amc23.parquet \
    --output_path /primus_datasets/zmy/GARO/test_data/amc23.parquet

  # Or by TASK name and path template:
  PRIMUS_API_KEY=... \
  python scripts/build_mode_hints_to_parquet.py \
    --task amc23 --data_path_template '/primus_datasets/zmy/GARO/test_data/${TASK}.parquet'

Notes:
  - This script calls an internal Primus completion endpoint that streams via SSE.
    We parse SSE using `requests` only (no extra dependency).
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


# -------------------------
# Prompt template
# -------------------------

INSTRUCTION_PROMPT = (
    "You are a math problem solver and a \"mode generator\".\n\n"
    "Task:\n"
    "For EACH math problem provided below, generate 2 to 5 distinct solution approaches (modes).\n"
    "Each mode must include:\n"
    "(1) a short hint (like: \"use Markov inequality\"),\n"
    "(2) a concise solution,\n"
    "and the final answer MUST appear in LaTeX \\boxed{...} at the end of the solution.\n\n"
    "Requirements:\n"
    "- Produce 2–5 modes per problem.\n"
    "- Modes must be meaningfully different (different key idea/technique), not just cosmetic rewrites.\n"
    "- Keep the hint very short (<= 10 words). No punctuation-heavy or multi-sentence hints.\n"
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
# Primus Qwen SSE client
# -------------------------


@dataclass
class PrimusClient:
    url: str
    api_key: str
    model: str
    timeout: Optional[float] = None

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

    def _build_request_data(self, prompt: str, model: str) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        session_id = uuid.uuid4().hex
        return {
            "session_id": session_id,
            "request_id": request_id,
            "model": model,
            "prompt": prompt,
            "extra_args": {
                "top_k": 50,
                "top_p": 0.8,
                "temperature": 0.5,
                # keep generous defaults to match existing Primus usage
                "max_prefill_tokens": 65536,
                "context_length": 65536,
                "max_new_tokens": 8192,
            },
        }

    def _iter_sse(self, resp: requests.Response) -> Iterable[Tuple[Optional[str], str]]:
        """Yield (event_name, data) from an SSE response."""
        event: Optional[str] = None
        data_lines: List[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip("\r")
            if not line:
                if data_lines:
                    yield event, "\n".join(data_lines)
                event = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip() or None
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
                continue
        if data_lines:
            yield event, "\n".join(data_lines)

    def complete(self, messages: List[Dict[str, str]], *, model: Optional[str] = None) -> str:
        model = model or self.model
        prompt = self._convert_openai_messages_to_prompt(messages)
        payload = self._build_request_data(prompt, model)

        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = requests.post(self.url, json=payload, headers=headers, stream=True, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(
                f"primus request failed: {type(e).__name__}: {e}; url={self.url}; model={model}; request_id={payload.get('request_id')}"
            )

        if not resp.ok:
            body_preview = (resp.text or "")[:800]
            raise RuntimeError(
                f"primus HTTP {resp.status_code}; url={self.url}; model={model}; request_id={payload.get('request_id')}; body[:800]={body_preview!r}"
            )

        last_content = ""
        for ev, data in self._iter_sse(resp):
            if not data:
                continue
            try:
                raw = json.loads(data)
            except Exception:
                # Some servers may send non-JSON keep-alives; ignore.
                continue

            # Expected schema: raw["choices"][0]["message"]["content"]
            try:
                choices = raw.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or ""
                    if isinstance(content, str):
                        last_content = content
            except Exception:
                pass

            if ev == "complete":
                break

        return last_content


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
        default="/primus_datasets/zmy/GARO/test_data/${TASK}.parquet",
        help="Template for --task (default matches user's GARO paths)",
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
        default="http://shenma-ai-pub.alibaba-inc.com/v1/completions",
        help="Primus completion endpoint URL",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="qwen3-235b-instruct-trans-sg",
        help="Primus model name",
    )
    ap.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Primus API key. Prefer env PRIMUS_API_KEY.",
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

    api_key = args.api_key or os.environ.get("PRIMUS_API_KEY") or os.environ.get("QWEN_PRIMUS_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set env PRIMUS_API_KEY (recommended) or pass --api_key")

    client = PrimusClient(url=args.primus_url, api_key=api_key, model=args.model, timeout=args.timeout)

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
        user_prompt = INSTRUCTION_PROMPT.format(problems=problem_text)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = client.complete(messages)
            arr = _extract_json_array(content)
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
