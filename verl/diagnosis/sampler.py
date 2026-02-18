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

"""HuggingFace sampling helpers for offline diagnosis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from verl.utils import hf_tokenizer


def _pick_input_device(model: torch.nn.Module) -> torch.device:
    # Works for both normal and sharded (device_map) models.
    try:
        p = next(model.parameters())
        if p is not None and p.device is not None:
            return p.device
    except StopIteration:
        pass
    # fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    """Load a causal LM model + tokenizer for offline generation.

    The function intentionally stays lightweight and does not depend on verl's
    rollout workers.
    """
    from transformers import AutoModelForCausalLM

    tokenizer = hf_tokenizer(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def _encode_prompt(
    tokenizer: Any,
    prompt_messages: Any,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    # prompt_messages is expected to be chat-template messages: list[dict]
    # but keep it robust.
    if hasattr(prompt_messages, "tolist") and callable(prompt_messages.tolist):
        prompt_messages = prompt_messages.tolist()

    if isinstance(prompt_messages, str):
        enc = tokenizer(
            prompt_messages,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_dict=True,
        )
    else:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template, but prompt is chat messages.")
        enc = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
    return {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}


@torch.inference_mode()
def sample_n(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_messages: Any,
    *,
    temperature: float,
    n: int,
    max_new_tokens: int,
    top_p: float = 0.95,
    repetition_penalty: Optional[float] = None,
) -> List[str]:
    """Sample n responses for a single prompt."""
    if n <= 0:
        return []

    device = _pick_input_device(model)
    inputs = _encode_prompt(tokenizer, prompt_messages, device=device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    gen_kwargs: Dict[str, Any] = dict(
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        num_return_sequences=int(n),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)

    # generate returns [n, prompt+response]
    out = model.generate(input_ids=input_ids, **gen_kwargs)

    responses: List[str] = []
    prompt_len = input_ids.shape[-1]
    for seq in out:
        # slice out the generated part
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(text)
    return responses
