# Mode Diagnosis (verl) — 使用说明

这套代码用于对一个 **HF causal LM** 在数学题数据集上做 **高温/低温多样采样**、**exact-match 验证**、以及 **mode 互补性诊断指标**（目前 mode 判别为占位的 `NullModeDetector`，后续可替换成你自己的聚类/骨架解析）。

> 入口脚本：`scripts/run_mode_diagnosis.py`  
> 代码模块：`verl/diagnosis/`

---

## 1. 前置条件

### 1.1 Python / 依赖
确保你已经能跑通 `verl` repo 原本的环境（通常包含）：

- `torch`
- `transformers`
- `pandas`
- `pyarrow`（用于 `read_parquet`）
- （可选）`accelerate`（如果你希望更方便地使用 `device_map="auto"`）

> 注意：本诊断脚本不会动训练/rollout/worker，只是离线 generate + verify。

### 1.2 数据格式（Parquet）
`data_path` 指向一个 parquet 文件，至少包含字段：

- `prompt`: `list[dict]`（chat messages），例如：
  ```json
  [{"role":"user","content":"Solve ..."}]
  ```
- `reward_model`: `dict`，至少包含：
  - `ground_truth`: `str`（exact-match 的目标答案）

可选字段（会原样写入输出 meta）：

- `data_source`
- `ability`
- 其它你想保留的 meta

---

## 2. 一行跑起来

### 2.1 基本用法（默认参数）
在 repo 根目录执行：

```bash
python scripts/run_mode_diagnosis.py
```

默认参数等价于：

- `--model_name "Qwen/Qwen3-8B"`
- `--data_path "dapo_math_17k.parquet"`
- `--high_temp 1.0 --high_temp_samples 32`
- `--low_temp 0.2 --low_temp_samples 8`
- `--max_resp_length 4096`
- `--output_path "diagnosis.jsonl"`

### 2.2 指定本地数据与输出
```bash
python scripts/run_mode_diagnosis.py \
  --model_name Qwen/Qwen3-8B \
  --data_path /path/to/dapo_math_17k.parquet \
  --output_path /path/to/diagnosis.jsonl
```

### 2.3 控制采样强度
```bash
python scripts/run_mode_diagnosis.py \
  --high_temp 1.1 --high_temp_samples 64 \
  --low_temp 0.3 --low_temp_samples 16 \
  --max_resp_length 4096
```

---

## 3. 脚本做了什么（Step1–3 总览）

对每道题（一个 `prompt`）：

1. **加载模型与 tokenizer**（HF `AutoModelForCausalLM` + `AutoTokenizer`）
2. 两种策略采样：
   - 高温：`temperature=high_temp`，采样 `high_temp_samples` 条
   - 低温：`temperature=low_temp`，采样 `low_temp_samples` 条
3. **verifier（exact-match）**：
   - 从 response 抽取 `final_answer`
   - 与 `reward_model.ground_truth` 做 exact-match
4. **写入 JSONL**（每题一行）
5. **聚合统计指标**（最后追加一行 `type=summary` 的汇总，并打印到终端）

---

## 4. 输出文件：diagnosis.jsonl 格式

### 4.1 每题一条记录（type=problem）
每一行是一道题的完整结果，结构大致如下（字段可能略有不同，取决于你数据里的 meta）：

```json
{
  "type": "problem",
  "id": 123,
  "prompt": [...],
  "ground_truth": "2",
  "meta": {
    "data_source": "dapo_math_17k",
    "ability": "math"
  },
  "high": {
    "temperature": 1.0,
    "samples": [
      {
        "text": "...",
        "final_answer": "2",
        "correct": true,
        "verify_method": "hash_mark"
      }
    ]
  },
  "low": {
    "temperature": 0.2,
    "samples": [
      {
        "text": "...",
        "final_answer": "3",
        "correct": false,
        "verify_method": "fallback"
      }
    ]
  },
  "mode": {
    "detector": "null",
    "note": "mode assignment placeholder"
  }
}
```

### 4.2 汇总一条记录（type=summary）
文件最后会追加一行：

```json
{
  "type": "summary",
  "num_problems": 17000,
  "high": {"solved_at_least_one": 0.72, "sample_acc": 0.31},
  "low":  {"solved_at_least_one": 0.65, "sample_acc": 0.45},
  "union": {"solved_any": 0.78},
  "mode_metrics": {
    "acc_union": 0.78,
    "acc_max_single_mode": 0.78,
    "gap_union_minus_best": 0.0,
    "pairwise_jaccard": {}
  }
}
```

> 由于当前 `ModeDetector` 是 `NullModeDetector`，所以 mode 相关指标会退化（gap≈0，jaccard 为空或单一模式）。

---

## 5. 终端会打印哪些指标？

脚本会打印：

- 进度（每隔若干题）
- 高温、低温：
  - `solved@N`: 该题在 N 次采样中是否至少有 1 个 correct
  - `sample_acc`: 采样级别正确率（correct samples / total samples）
- overall union：
  - `solved_any`: 高温或低温任意一个样本 correct 即 solved
- mode 指标（当前退化）：
  - `Acc_union`, `Acc_max_single`, `gap`, `Jaccard`（若有）

---

## 6. 常见问题排查

### 6.1 `apply_chat_template` 报错
你的 tokenizer 可能不是 chat 模型或没有 chat template。

处理方式：
- 换一个 chat-instruct 模型
- 或在 `verl/diagnosis/sampler.py` 里改为直接拼 prompt（不推荐，先保持最小侵入）

### 6.2 显存不够 / OOM
- 降低 `--max_resp_length`
- 降低 `--high_temp_samples` / `--low_temp_samples`
- 用更小的模型
- （如果你愿意改一点点）在 `sampler.py` 里做 batch generate / 分批 decode

### 6.3 ground_truth 不存在
确认 parquet 中 `reward_model` 是 dict 且包含 `ground_truth`。  
如果你的字段路径不同，改 `verl/diagnosis/data.py` 里的解析逻辑即可。

---

## 7. 下一步：如何替换 ModeDetector（你后续的 3.1/3.2）

当前 `verl/diagnosis/mode.py` 里是：

- `ModeDetector` 抽象接口
- `NullModeDetector` 占位实现

你可以新增例如：

- `KeywordFeatureModeDetector`
- `SkeletonOpSeqModeDetector`

然后在 `scripts/run_mode_diagnosis.py` 把：

```python
mode_detector = NullModeDetector()
```

替换成你的 detector 即可。  
聚合器会自动计算 `S_k`、union/best、Jaccard 等互补性指标。

---

## 8. 最小可复现实验建议（sanity check）

先用很小的数据子集（比如 20–100 题）跑通：

```bash
python scripts/run_mode_diagnosis.py \
  --data_path /path/to/small.parquet \
  --high_temp_samples 8 --low_temp_samples 2 \
  --max_resp_length 512 \
  --output_path /tmp/diagnosis.small.jsonl
```

确认：
- jsonl 每行都是合法 JSON
- summary 行存在
- 终端指标合理

---

如果你想我继续：我可以把 **mode 判别（关键词特征 + HDBSCAN/KMeans）** 的第一版也按同样“最小侵入”方式加进去，并确保输出 schema 不破坏当前格式。

---

## 4. Step2：ModeBank 指标汇总（entropy / coverage / reach complementarity）

如果你的 parquet 里还额外包含字段：

- `mode`: `list[str]`（每道题的一组 hint，例如 `"use Markov inequality"`）

那么 Step1 会对每个 hint 做中温采样并记录 `reach.per_hint[*].solved`。

当你分别对 **base model** 与 **GRPO model** 跑完 Step1（会得到两个 jsonl）后，
可以使用下面脚本计算 Step2 指标并打印对比：

```bash
python scripts/summarize_modebank_diagnosis.py \
  --base_jsonl outputs/diagnosis.base.jsonl \
  --grpo_jsonl outputs/diagnosis.grpo.jsonl
```

可选：按 `meta` 里的字段分组（例如 `ability` / `problem_type`）：

```bash
python scripts/summarize_modebank_diagnosis.py \
  --base_jsonl outputs/diagnosis.base.jsonl \
  --grpo_jsonl outputs/diagnosis.grpo.jsonl \
  --group_by ability
```

脚本会输出：
- NAT mode entropy（由自然采样文本按 hint-keyword overlap 的基线规则标注后计算）
- NAT coverage vs REACH（|M_nat(x)| / |M_reach(x)|）
- REACH complementarity（Acc(union) vs Acc(best)）
