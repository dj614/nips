#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for GARO mode diagnosis.
#
# It performs:
#   (1) mode hint generation by Primus API: writes `mode` (list[str]) into each parquet.
#   (2) offline sampling + ModeBank metrics for base + RL-trained models.

OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-outputs}
DATA_ROOT=${GARO_DATA_ROOT:-"/primus_datasets/zmy/GARO/test_data"}
TASKS_CSV=${TASKS_CSV:-"amc23,aime24,aime25"}
GROUP_BY=${GROUP_BY:-ability}

# Sampling settings (kept consistent with scripts/run_mode_diagnosis.sh defaults).
HIGH_TEMP=${HIGH_TEMP:-1.0}
HIGH_SAMPLES=${HIGH_SAMPLES:-32}
LOW_TEMP=${LOW_TEMP:-0.2}
LOW_SAMPLES=${LOW_SAMPLES:-8}
MID_TEMP=${MID_TEMP:-0.6}
MID_SAMPLES_PER_HINT=${MID_SAMPLES_PER_HINT:-4}

MODELS_FILE=${MODELS_FILE:-"${OUTPUT_DIR}/GARO/models.json"}
SKIP_MODE_GEN=0
ONLY_MODE_GEN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/run_garo_mode_diagnosis.sh [--skip_mode_gen] [--only_mode_gen] \
    [--tasks "amc23,aime24,aime25"] [--data_root /path/to/test_data] \
    [--models_file /path/to/models.json] [--output_dir outputs/GARO]

Environment overrides:
  GARO_DATA_ROOT, TASKS_CSV, PRIMUS_OUTPUT_DIR, GROUP_BY,
  HIGH_TEMP/HIGH_SAMPLES, LOW_TEMP/LOW_SAMPLES, MID_TEMP/MID_SAMPLES_PER_HINT

Notes:
  - Step (1) requires PRIMUS_API_KEY in the environment.
  - Mode hints are written in-place to the parquet files by default.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip_mode_gen)
      SKIP_MODE_GEN=1; shift 1;;
    --only_mode_gen)
      ONLY_MODE_GEN=1; shift 1;;
    --tasks)
      TASKS_CSV="$2"; shift 2;;
    --data_root)
      DATA_ROOT="$2"; shift 2;;
    --models_file)
      MODELS_FILE="$2"; shift 2;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      usage; exit 2;;
  esac
done

mkdir -p "${OUTPUT_DIR}/GARO"

# Build default models file if missing (JSON -> no PyYAML dependency).
if [[ ! -f "${MODELS_FILE}" ]]; then
  python scripts/build_garo_models_file.py --output_path "${MODELS_FILE}"
fi

IFS="," read -r -a TASKS <<< "${TASKS_CSV}"

for t in "${TASKS[@]}"; do
  t=$(echo "$t" | xargs)
  if [[ -z "$t" ]]; then
    continue
  fi
  DATA_PATH="${DATA_ROOT}/${t}.parquet"
  OUT_DIR="${OUTPUT_DIR}/GARO/${t}"
  mkdir -p "${OUT_DIR}"

  if [[ ${SKIP_MODE_GEN} -eq 0 ]]; then
    echo "[garo_mode] (1) build mode hints -> ${DATA_PATH}"
    python scripts/build_mode_hints_to_parquet.py \
      --data_path "${DATA_PATH}" \
      --output_path "${DATA_PATH}" \
      --only_missing
  fi

  if [[ ${ONLY_MODE_GEN} -eq 1 ]]; then
    continue
  fi

  echo "[garo_mode] (2) run diagnosis on ${t}"
  OUTPUT_JSONL="${OUT_DIR}/diagnosis.jsonl"

  python scripts/run_mode_diagnosis.py \
    --models_file "${MODELS_FILE}" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_JSONL}" \
    --answer_extract boxed --boxed_cmd "\\boxed" \
    --high_temp "${HIGH_TEMP}" --high_temp_samples "${HIGH_SAMPLES}" \
    --low_temp "${LOW_TEMP}" --low_temp_samples "${LOW_SAMPLES}" \
    --mid_temp "${MID_TEMP}" --mid_temp_samples_per_hint "${MID_SAMPLES_PER_HINT}"

  MANIFEST_PATH="${OUTPUT_JSONL%.*}.manifest.json"
  python scripts/summarize_modebank_diagnosis.py \
    --manifest "${MANIFEST_PATH}" \
    --baseline_tag qwen3_8b \
    --group_by "${GROUP_BY}" \
    --save_json "${OUT_DIR}/modebank_summary.json"
done

echo "[garo_mode] done. outputs under: ${OUTPUT_DIR}/GARO"
