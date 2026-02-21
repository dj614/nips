import os
import json
from typing import Dict, List, Union, Optional, Tuple, Any
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
)


def load_hf_dataset(
    hf_name: str,
    split: Union[str, None],
    subset: Union[str, None],
):
    """
    通用加载 HuggingFace 数据集：
    - 支持有 / 没有 subset(name)
    - 支持有 / 没有 split（无 split 时自动合并所有 split）
    """
    load_kwargs = {}

    if subset not in (None, "", "none", "None"):
        load_kwargs["name"] = subset

    if split not in (None, "", "none", "None"):
        load_kwargs["split"] = split

    ds_raw = load_dataset(hf_name, **load_kwargs)

    if isinstance(ds_raw, DatasetDict):
        # 没指定 split 时，合并所有 split
        ds = concatenate_datasets(list(ds_raw.values()))
    else:
        ds = ds_raw

    return ds


# (hf_name, data_name, split, subset, prompt_key, groundtruth_key)
DatasetConfig = Tuple[str, str, Optional[str], Optional[str], str, str]


def _is_chat_messages(x: Any) -> bool:
    """
    判断 x 是否已经是 HF chat template 风格：
    [
      {"role": "...", "content": "..."},
      ...
    ]
    """
    if not isinstance(x, list) or len(x) == 0:
        return False
    for m in x:
        if not isinstance(m, dict):
            return False
        if "role" not in m or "content" not in m:
            return False
    return True


def _to_prompt_str(val: Any) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _normalize_prompt_to_messages(val: Any) -> List[Dict[str, str]]:
    """
    统一把 prompt 规范成 chat messages。
    - 若原本就是 messages，直接返回
    - 否则转成字符串后用 user 包一层
    """
    if _is_chat_messages(val):
        # 保留原结构
        # 严格转成 role/content 字符串
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in val]

    prompt_str = _to_prompt_str(val).strip()
    return [{"role": "user", "content": prompt_str}]


def process_datasets(configs: List[DatasetConfig]):
    """
    每个 config:
      (hf_name, data_name, split, subset, prompt_key, groundtruth_key)

    例如：
      ("Idavidrein/gpqa", "gpqa_diamond", "train", "gpqa_diamond", "Question", "Correct Answer")
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    for hf_name, data_name, split, subset, prompt_key, groundtruth_key in configs:
        print(f"Loading dataset: {hf_name}, split={split}, subset={subset}")
        ds = load_hf_dataset(hf_name, split, subset)

        records = []
        for ex in ds:
            # ====== gpqa 特殊处理：改 prompt + groundtruth ======
            if hf_name == "Idavidrein/gpqa":
                # 问题
                q_val = ex.get(prompt_key, "")
                question = _to_prompt_str(q_val)

                # 选项：来自原始字段
                opt_A = str(ex.get("Correct Answer", ""))
                opt_B = str(ex.get("Incorrect Answer 1", ""))
                opt_C = str(ex.get("Incorrect Answer 2", ""))
                opt_D = str(ex.get("Incorrect Answer 3", ""))

                # 拼接新的 prompt
                prompt_str = (
                    question
                    + " Please choose one of the following options as the answer: "
                    + f"(A) {opt_A}, (B) {opt_B}, (C) {opt_C}, (D) {opt_D}. "
                    + "Reason step by step and output your final answer as only one letter "
                      "(A, B, C, or D) inside \\boxed{}."
                )

                prompt_messages = [{"role": "user", "content": prompt_str}]

                # groundtruth 改成选项字母（正确答案放在 A）
                gt_str = "A"

            else:
                # ====== 其它数据集：保持原逻辑，但 prompt 输出为 chat messages ======
                p_val = ex.get(prompt_key, "")
                prompt_messages = _normalize_prompt_to_messages(p_val)

                g_val = ex.get(groundtruth_key, "")
                gt_str = _to_prompt_str(g_val)

            records.append(
                {
                    "prompt": prompt_messages,
                    "reward_model": {"ground_truth": gt_str},
                    "data_source": data_name,
                    "ability": "math",
                }
            )

        jsonl_path = os.path.join(SAVE_DIR, f"{data_name}.jsonl")
        parquet_path = os.path.join(SAVE_DIR, f"{data_name}.parquet")

        # 写 JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 写 Parquet（自动推断 schema；prompt 会变成 list<struct<role, content>>）
        table = pa.Table.from_pylist(records)
        pq.write_table(table, parquet_path)

        print(
            f"Saved {len(records)} rows for {hf_name} "
            f"to {jsonl_path} and {parquet_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/primus_datasets/zmy/NIPS/test_data/",
        help=f"保存 JSONL/Parquet 的目录",
    )
    args = parser.parse_args()

    # 覆盖全局 SAVE_DIR，后面 process_datasets 会用到
    SAVE_DIR = args.save_dir

    # (hf_name, data_name, split, subset, prompt_key, groundtruth_key)
    example_configs: List[DatasetConfig] = [
        # for math & dapo_math
        ("HuggingFaceH4/aime_2024", "aime24", "train", None, "problem", "answer"),
        ("math-ai/aime25", "aime25", "test", None, "problem", "answer"),
        ("math-ai/amc23", "amc23", "test", None, "question", "answer"),
    ]

    process_datasets(example_configs)