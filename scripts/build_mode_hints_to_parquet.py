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
    If you provide only a host (e.g., `10.2.5.30`), we will default to port 8000.
    Set it to either host[:port] (we append `/v1/completions`) or a full URL.
  - The completion API is expected to be compatible with the OpenAI-style
    `/v1/completions` schema returned by vLLM:
      request payload: {"prompt": ..., ...generation_params}
      response: {"choices": [{"text": ..., "finish_reason": ...}, ...]}
  - We write `mode` as list[str] (hints only) to match the existing diagnosis loader.
  - By default, we DO NOT overwrite the input parquet. Instead, we write to a new file
    with suffix `.with_mode.parquet` (to avoid clobbering data on slow network filesystems).
    Use `--inplace` to overwrite input explicitly.
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
# -------------------------
# OpenAI-compatible client (/v1/completions or /v1/chat/completions)
# -------------------------


class APIRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.body = body


def _endpoint_kind(url: str) -> str:
    u = (url or "").lower()
    # Must check chat first because it also contains "completions".
    if "/chat/completions" in u:
        return "chat"
    if "/completions" in u:
        return "completions"
    return "unknown"


def _swap_completions_endpoint(url: str) -> str:
    u = str(url).rstrip("/")
    if u.endswith("/v1/completions"):
        return u[: -len("/v1/completions")] + "/v1/chat/completions"
    if u.endswith("/v1/chat/completions"):
        return u[: -len("/v1/chat/completions")] + "/v1/completions"
    # If user passed a base URL (or something else), be conservative.
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def _post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_seconds: float,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds, headers=headers or {})
    except requests.RequestException as e:
        raise APIRequestError(f"request failed: {e}", url=url) from e

    if not resp.ok:
        # Keep error body short to avoid spamming logs.
        body = (resp.text or "").strip()
        if len(body) > 800:
            body = body[:800] + " ... (truncated)"
        raise APIRequestError(
            f"http error: {resp.status_code}",
            url=url,
            status_code=int(resp.status_code),
            body=body,
        )

    try:
        out = resp.json()
    except Exception as e:
        body = (resp.text or "").strip()
        if len(body) > 800:
            body = body[:800] + " ... (truncated)"
        raise APIRequestError("non-json response", url=url, body=body) from e

    if not isinstance(out, dict):
        raise APIRequestError(f"unexpected response type: {type(out)}", url=url)
    return out


def _extract_text_from_openai_result(result: Dict[str, Any]) -> str:
    # vLLM / OpenAI-like responses generally have `choices`.
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        c = choices[-1]
        if isinstance(c, dict):
            if c.get("text") is not None:
                return str(c.get("text"))
            msg = c.get("message")
            if isinstance(msg, dict) and msg.get("content") is not None:
                return str(msg.get("content"))

    # Surface API error if present.
    err = result.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("type") or "unknown_error"
        raise APIRequestError(f"api error: {msg}")

    raise APIRequestError("unexpected response schema (missing choices/text/message.content)")


def _call_openai_compatible(
    *,
    url: str,
    model: Optional[str],
    messages: List[Dict[str, str]],
    prompt: Optional[str],
    param_gen: Dict[str, Any],
    timeout_seconds: float,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    """Call either /v1/completions or /v1/chat/completions.

    Returns:
      (text, used_url, used_kind)
    """

    def _payload_for(k: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(param_gen or {})
        if model:
            payload["model"] = str(model)
        if k == "chat":
            payload["messages"] = messages
        else:
            payload["prompt"] = prompt if prompt is not None else ""
        return payload

    kind = _endpoint_kind(url)

    def _try(k: str, u: str) -> str:
        payload = _payload_for(k)
        result = _post_json(u, payload, timeout_seconds=timeout_seconds, headers=headers)
        return _extract_text_from_openai_result(result)

    # If the URL already points at a known endpoint, try it first, then fallback to the sibling.
    if kind in ("chat", "completions"):
        try:
            return _try(kind, url), url, kind
        except APIRequestError as e:
            # Many clusters only expose one of the endpoints; gracefully fallback.
            if e.status_code in (404, 405) or "unexpected response schema" in str(e):
                alt_url = _swap_completions_endpoint(url)
                alt_kind = "chat" if kind == "completions" else "completions"
                return _try(alt_kind, alt_url), alt_url, alt_kind
            raise

    # Unknown URL: try completions first (backward-compatible), then chat.
    last_err: Optional[APIRequestError] = None
    for k_try, u_try in (("completions", url), ("chat", _swap_completions_endpoint(url))):
        try:
            return _try(k_try, u_try), u_try, k_try
        except APIRequestError as e:
            last_err = e
            if e.status_code in (404, 405):
                continue
            break

    raise last_err or APIRequestError("failed to resolve a working completion endpoint", url=url)


@dataclass
class PrimusClient:
    url: str
    timeout: Optional[float] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    # Generation params (top-level keys in the OpenAI-compatible payload).
    param_gen: Optional[Dict[str, Any]] = None

    def _default_headers(self) -> Dict[str, str]:
        key = (self.api_key or "").strip()
        if not key:
            return {}
        return {"Authorization": f"Bearer {key}"}

    def _convert_openai_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Qwen / chatml-style text prompt (works with many /v1/completions servers).
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
        """Send a single request and return assistant text."""

        # Default generation parameters (keep stable behavior).
        param_gen = dict(self.param_gen or {})
        param_gen.setdefault("top_k", 50)
        param_gen.setdefault("top_p", 0.8)
        param_gen.setdefault("temperature", 0.5)
        param_gen.setdefault("max_tokens", 1024)

        timeout_seconds = float(self.timeout) if self.timeout is not None else 300.0

        # Build both prompt and messages; backend decides which one it uses.
        prompt = self._convert_openai_messages_to_prompt(messages)

        try:
            text, used_url, _used_kind = _call_openai_compatible(
                url=self.url,
                model=self.model,
                messages=messages,
                prompt=prompt,
                param_gen=param_gen,
                timeout_seconds=timeout_seconds,
                headers=self._default_headers(),
            )
        except APIRequestError as e:
            extra = []
            if e.url:
                extra.append(f"url={e.url}")
            if e.status_code is not None:
                extra.append(f"status={e.status_code}")
            if e.body:
                extra.append(f"body={e.body}")
            detail = (" | " + " ".join(extra)) if extra else ""
            raise RuntimeError(f"LLM API call failed: {e}{detail}") from e

        # Auto-heal: keep using the working URL in later calls.
        self.url = used_url
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


def _default_output_path(data_path: str) -> str:
    """Return a safe default output path that does not overwrite the input."""
    p = str(data_path)
    if p.endswith(".parquet"):
        return p[: -len(".parquet")] + ".with_mode.parquet"
    return p + ".with_mode.parquet"


def _wait_until_parquet_readable(path: str, max_retries: int = 8, base_sleep: float = 1.0) -> None:
    """Best-effort: wait until a parquet becomes readable.

    This mitigates eventual-consistency delays on network / cloud-backed storage.
    """

    last_err: Optional[BaseException] = None
    for r in range(max_retries):
        try:
            # Read a tiny subset to validate metadata + footer availability.
            _ = pd.read_parquet(path, columns=["prompt", "mode"]).head(1)
            return
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2**r))

    raise RuntimeError(f"output parquet not readable after retries: {path} ({last_err})")


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
        help=(
            "Output parquet path. Default: write to a new file with suffix '\.with_mode.parquet' "
            "(does NOT overwrite input)."
        ),
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input parquet in place (DANGEROUS on slow network filesystems).",
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
            "OpenAI-compatible completion endpoint URL. If omitted, use env QWEN_IP and append /v1/completions if needed. Both /v1/completions and /v1/chat/completions are supported (auto-fallback)."
        ),
    )
    ap.add_argument(
        "--model",
        type=str,
        default="qwen3-235b-instruct-trans-sg",
        help="Model name (sent as `model` in the OpenAI-compatible payload). Some servers require this field.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional requests timeout (seconds)",
    )



    ap.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional API key for OpenAI-compatible servers (also read from env OPENAI_API_KEY / QWEN_API_KEY).",
    )

    args = ap.parse_args()

    # Default behavior: only_missing unless --overwrite is explicitly set.
    only_missing = bool(args.only_missing) or not bool(args.overwrite)

    data_path = _resolve_data_path(args.task, args.data_path, args.data_path_template)
    if args.output_path:
        out_path = str(args.output_path)
    else:
        out_path = data_path if args.inplace else _default_output_path(data_path)

    if (not args.inplace) and (out_path == data_path):
        raise SystemExit(
            "Refusing to overwrite input parquet without --inplace. Set --output_path or pass --inplace."
        )

    # Resolve endpoint from CLI or env (QWEN_IP preferred).
    primus_url = None
    if args.primus_url:
        primus_url = str(args.primus_url).strip()
    if not primus_url:
        qip = None
        qip_src = None
        v = os.environ.get("QWEN_IP")
        qip = v
        qip_src = "QWEN_IP"
        if qip:
            u = str(qip).strip()
            if u:
                if not (u.startswith("http://") or u.startswith("https://")):
                    u = "http://" + u
                u = u.rstrip("/")
                # Hard-coded default: our vLLM/OpenAI-compatible service listens on 8000.
                # Only apply to the QWEN_* envs; do not rewrite generic OpenAI base URLs.
                if qip_src == "QWEN_IP":
                    # Insert default port if the URL netloc doesn't already contain one.
                    # Examples:
                    #   http://10.2.5.30          -> http://10.2.5.30:8000
                    #   http://10.2.5.30/v1/...   -> http://10.2.5.30:8000/v1/...
                    try:
                        scheme, rest = u.split("://", 1)
                        netloc, sep, tail = rest.partition("/")
                        auth, at, hostport = netloc.rpartition("@")
                        if at:
                            host = hostport
                        else:
                            auth = ""
                            host = netloc
                        # Avoid mishandling IPv6 literals like "[::1]".
                        if host and not host.startswith("[") and ":" not in host:
                            host = f"{host}:8000"
                            netloc = f"{auth}@{host}" if auth else host
                            rest = netloc + (sep + tail if sep else "")
                            u = f"{scheme}://{rest}"
                    except Exception:
                        # Be conservative on parsing errors.
                        pass
                if "/v1/" not in u:
                    u = u + "/v1/completions"
                primus_url = u

    if not primus_url:
        raise SystemExit(
            "Missing endpoint. Please export QWEN_IP (or QWEN_IP) as host[:port] or full URL, "
            "or pass --primus_url explicitly."
        )

    api_key = (
        (args.api_key.strip() if args.api_key else None)
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("API_KEY")
    )

    client = PrimusClient(
        url=primus_url,
        timeout=args.timeout,
        model=(str(args.model).strip() if args.model else None),
        api_key=api_key,
    )

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
    _wait_until_parquet_readable(out_path)
    print(f"[mode_hints] done. wrote={n_done} skipped={n_skip} failed={n_fail} -> {out_path}")


if __name__ == "__main__":
    main()
