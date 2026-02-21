#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for NIPS mode diagnosis.
#
# It performs:
#   (1) mode hint generation by a vLLM-style /v1/completions API: writes `mode` (list[str]) into each parquet.
#   (2) offline sampling + ModeBank metrics for base + RL-trained models.

OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-outputs}
DATA_ROOT="/primus_datasets/zmy/NIPS/test_data"
TASKS_CSV=${TASKS_CSV:-"amc23,aime24,aime25"}
GROUP_BY=${GROUP_BY:-ability}

# Sampling settings (kept consistent with scripts/run_mode_diagnosis.sh defaults).
HIGH_TEMP=${HIGH_TEMP:-1.0}
HIGH_SAMPLES=${HIGH_SAMPLES:-32}
LOW_TEMP=${LOW_TEMP:-0.2}
LOW_SAMPLES=${LOW_SAMPLES:-8}
MID_TEMP=${MID_TEMP:-0.6}
MID_SAMPLES_PER_HINT=${MID_SAMPLES_PER_HINT:-4}

MODELS_FILE="scripts/qwen_models_file.yaml"
SKIP_MODE_GEN=0
ONLY_MODE_GEN=0

wait_parquet_readable() {
  local p="$1"
  # Best-effort guard for eventually-consistent filesystems.
  python - "$p" <<'PY'
import sys
import time

path = sys.argv[1]
last_err = None
for r in range(8):
    try:
        import pandas as pd
        _ = pd.read_parquet(path).head(1)
        sys.exit(0)
    except Exception as e:
        last_err = e
        time.sleep(1.0 * (2 ** r))
raise SystemExit(f"parquet not readable after retries: {path} ({last_err})")
PY
}

usage() {
  cat <<EOF
Usage:
  bash scripts/run_nips_mode_diagnosis.sh [--skip_mode_gen] [--only_mode_gen] \
    [--tasks "amc23,aime24,aime25"] [--data_root /path/to/test_data] \
    [--models_file /path/to/models.yaml] [--output_dir outputs/NIPS]

Environment overrides:
  TASKS_CSV, PRIMUS_OUTPUT_DIR, GROUP_BY,
  HIGH_TEMP/HIGH_SAMPLES, LOW_TEMP/LOW_SAMPLES, MID_TEMP/MID_SAMPLES_PER_HINT

Notes:
  - Step (1) uses env QWEN_IP (or QWEN_IP) to route to the Qwen completion endpoint.
  - Mode hints are written out-of-place by default:
      <task>.parquet -> <task>.with_mode.parquet
    Use scripts/build_mode_hints_to_parquet.py --inplace if you truly want in-place updates.
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

mkdir -p "${OUTPUT_DIR}/NIPS"

if [[ ! -f "${MODELS_FILE}" ]]; then
  echo "ERROR: models file not found: ${MODELS_FILE}" >&2
  echo "       (Tip) use scripts/build_nips_models_file.py to generate a JSON models file, or pass --models_file." >&2
  exit 1
fi

IFS="," read -r -a TASKS <<< "${TASKS_CSV}"

for t in "${TASKS[@]}"; do
  t=$(echo "$t" | xargs)
  if [[ -z "$t" ]]; then
    continue
  fi
  DATA_PATH="${DATA_ROOT}/${t}.parquet"
  MODE_DATA_PATH="${DATA_PATH%.parquet}.with_mode.parquet"
  OUT_DIR="${OUTPUT_DIR}/NIPS/${t}"
  mkdir -p "${OUT_DIR}"

  if [[ ${SKIP_MODE_GEN} -eq 0 ]]; then
    echo "[nips_mode] (1) build mode hints -> ${MODE_DATA_PATH}"
    python scripts/build_mode_hints_to_parquet.py \
      --data_path "${DATA_PATH}" \
      --output_path "${MODE_DATA_PATH}" \
      --only_missing
  fi

  if [[ ${ONLY_MODE_GEN} -eq 1 ]]; then
    continue
  fi

  DIAG_DATA_PATH="${MODE_DATA_PATH}"
  if [[ ! -f "${DIAG_DATA_PATH}" ]]; then
    # Backward-compatible fallback: allow in-place mode files.
    DIAG_DATA_PATH="${DATA_PATH}"
  fi

  wait_parquet_readable "${DIAG_DATA_PATH}"

  echo "[nips_mode] (2) run diagnosis on ${t} (data=${DIAG_DATA_PATH})"
  OUTPUT_JSONL="${OUT_DIR}/diagnosis.jsonl"

  python scripts/run_mode_diagnosis.py \
    --models_file "${MODELS_FILE}" \
    --data_path "${DIAG_DATA_PATH}" \
    --output_path "${OUTPUT_JSONL}" \
    --answer_extract boxed --boxed_cmd "\\boxed" \
    --high_temp "${HIGH_TEMP}" --high_temp_samples "${HIGH_SAMPLES}" \
    --low_temp "${LOW_TEMP}" --low_temp_samples "${LOW_SAMPLES}" \
    --mid_temp "${MID_TEMP}" --mid_temp_samples_per_hint "${MID_SAMPLES_PER_HINT}"

  MANIFEST_PATH="${OUTPUT_JSONL%.*}.manifest.json"
  python scripts/summarize_modebank_diagnosis.py \
    --manifest "${MANIFEST_PATH}" \
    --baseline_tag qwen_base \
    --group_by "${GROUP_BY}" \
    --save_json "${OUT_DIR}/modebank_summary.json"
done

echo "[nips_mode] done. outputs under: ${OUTPUT_DIR}/NIPS"
