#!/usr/bin/env bash
set -euo pipefail

# A thin wrapper for running ModeBank diagnosis (step-1 sampling + step-2 summary).
#
# Recommended usage (most general):
#   bash scripts/run_mode_diagnosis.sh \
#     --models_file models.yaml \
#     --data_path /path/to/dataset.parquet \
#     --output_path outputs/diagnosis.jsonl \
#     --group_by ability
#
# models.yaml format:
#   models:
#     - tag: base
#       path: Qwen/Qwen3-8B
#     - tag: grpo
#       path: /path/to/ckpt
#       meta:
#         algo: grpo
#
# Legacy Primus usage is preserved:
#   bash scripts/run_mode_diagnosis.sh <JOB_ID> <STEP> [extra_tag=/path/to/ckpt ...]
# This will auto-build a temp models file containing base+grpo (+ extras).

OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-outputs}
DATA_DIR=${DATA_DIR:-"/primus_datasets/zmy"}
DEFAULT_TESTSETS=(amc23 aime24 aime25)
GROUP_BY=${GROUP_BY:-ability}

MODELS_FILE=""
DATA_PATH=""
DATA_PATH_USER_SPECIFIED="false"
OUTPUT_PATH="${OUTPUT_DIR}/NIPS/diagnosis.jsonl"
SAVE_JSON="${OUTPUT_DIR}/NIPS/modebank_summary.json"

# Parse optional flags.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models_file)
      MODELS_FILE="$2"; shift 2;;
    --data_path)
      DATA_PATH="$2"; DATA_PATH_USER_SPECIFIED="true"; shift 2;;
    --output_path)
      OUTPUT_PATH="$2"; shift 2;;
    --save_json)
      SAVE_JSON="$2"; shift 2;;
    --group_by)
      GROUP_BY="$2"; shift 2;;
    --)
      shift; break;;
    *)
      break;;
  esac
done

# If no --models_file is given, fall back to legacy positional args: JOB_ID STEP.
if [[ -z "$MODELS_FILE" ]]; then
  JOB_ID="${1:-}";
  STEP="${2:-}";
  if [[ -z "$JOB_ID" || -z "$STEP" ]]; then
    echo "ERROR: either provide --models_file, or legacy args: <JOB_ID> <STEP>" >&2
    exit 1
  fi
  shift 2 || true

  CHECKPOINT_DIR="/primus_oss_root/repository/quark_sft/${JOB_ID}/save_checkpoint"
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/global_step_${STEP}/actor/huggingface"

  # Prefer a local base checkpoint (Primus env) when available; otherwise fall back
  # to the public HF repo id.
  BASE_PATH="Qwen/Qwen3-8B"
  if [[ -n "${PRIMUS_SOURCE_CHECKPOINT_DIR:-}" ]]; then
    BASE_PATH="${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B"
  fi

  TMP_MODELS_FILE=$(mktemp /tmp/mode_diagnosis_models.XXXX.yaml)
  cat > "$TMP_MODELS_FILE" <<EOF_MODELS
models:
  - tag: base
    path: ${BASE_PATH}
  - tag: grpo
    path: ${CHECKPOINT_PATH}
EOF_MODELS

  # Optional additional models: tag=/path/to/ckpt
  for spec in "$@"; do
    if [[ "$spec" != *"="* ]]; then
      echo "WARN: skip invalid model spec '$spec' (expected tag=/path/to/ckpt)" >&2
      continue
    fi
    tag="${spec%%=*}"
    path="${spec#*=}"
    cat >> "$TMP_MODELS_FILE" <<EOF_EXTRA
  - tag: ${tag}
    path: ${path}
EOF_EXTRA
  done

  MODELS_FILE="$TMP_MODELS_FILE"
fi

# Ensure output directories exist.
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$(dirname "$SAVE_JSON")"

DATA_PATHS=()
DATA_TAGS=()
if [[ "$DATA_PATH_USER_SPECIFIED" == "true" ]]; then
  DATA_PATHS+=("$DATA_PATH")
  DATA_TAGS+=("custom")
else
  for t in "${DEFAULT_TESTSETS[@]}"; do
    DATA_PATHS+=("${DATA_DIR}/NIPS/test_data/${t}.parquet")
    DATA_TAGS+=("$t")
  done
fi

for i in "${!DATA_PATHS[@]}"; do
  dp="${DATA_PATHS[$i]}"
  tag="${DATA_TAGS[$i]}"

  out="$OUTPUT_PATH"
  save="$SAVE_JSON"
  if [[ "$DATA_PATH_USER_SPECIFIED" == "false" ]]; then
    out="${OUTPUT_PATH%.*}_${tag}.jsonl"
    save="${SAVE_JSON%.*}_${tag}.json"
  fi

  mkdir -p "$(dirname "$out")"
  mkdir -p "$(dirname "$save")"

  python scripts/run_mode_diagnosis.py \
    --models_file "$MODELS_FILE" \
    --data_path "$dp" \
    --output_path "$out" \
    --answer_extract boxed --boxed_cmd "\\boxed" \
    --high_temp 1.0 --high_temp_samples 32 \
    --low_temp 0.2 --low_temp_samples 8 \
    --mid_temp 0.6 --mid_temp_samples_per_hint 4

  MANIFEST_PATH="${out%.*}.manifest.json"

  python scripts/summarize_modebank_diagnosis.py \
    --manifest "$MANIFEST_PATH" \
    --baseline_tag base \
    --save_json "$save" \
    --group_by "$GROUP_BY"
done