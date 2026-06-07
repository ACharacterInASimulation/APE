#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/scratchpad_multihop.yaml"
INPUT_JSONL="data/scratchpad_multihop/eval.jsonl"
LITM_DIR="data/litm_nq"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/eval_suites/scratchpad_${RUN_ID}"
BASE_MODEL=""
CHECKPOINT=""
TRAIN_JSONL=""
TRAIN_EVAL_JSONL=""
TRAIN_MAX_STEPS=""
TRAIN_BATCH_SIZE=""
TRAIN_GRAD_ACCUM_STEPS=""
TRAIN_WARMUP_STEPS=""
TRAIN_LOGGING_STEPS=""
TRAIN_MAX_EXAMPLES=""
TRAIN_MAX_EVAL_EXAMPLES=""
TRAIN_LEARNING_RATE=""
TRAIN_SCRATCHPAD_LEARNING_RATE=""
TRAIN_SCRATCHPAD_INIT_TEXT=""
TRAIN_SPARSE_BACKEND=""
TRAIN_POSITION_STRATEGY=""
TRAIN_QUESTION_POSITION_GAP=""
MAX_EXAMPLES=""
LIMIT_PER_LITM_FILE=""
LITM_DOC_COUNTS="10,20,30"
LITM_POSITIONS="start,middle,end"
PARALLEL_LITM_POSITIONS="start"
MAX_NEW_TOKENS=""
MAX_CONTEXT_TOKENS=""
APE_TEMPERATURE=""
APE_SCALE=""
CUDA_DEVICE="${CUDA_DEVICE:-3}"
SKIP_TRAIN=0
FORCE_TRAIN=0
DRY_RUN=0

config_value() {
  python - "$CONFIG" "$1" "$2" <<'PY'
import sys
from pathlib import Path

config_path, dotted, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import yaml
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
except Exception:
    data = {}
cur = data
for part in dotted.split("."):
    if not isinstance(cur, dict) or part not in cur:
        print(fallback)
        raise SystemExit
    cur = cur[part]
print(cur if cur is not None else fallback)
PY
}

litm_gold_index() {
  case "$1:$2" in
    10:start) echo 0 ;;
    10:middle) echo 4 ;;
    10:end) echo 9 ;;
    20:start) echo 0 ;;
    20:middle) echo 9 ;;
    20:end) echo 19 ;;
    30:start) echo 0 ;;
    30:middle) echo 14 ;;
    30:end) echo 29 ;;
    *) echo "Unsupported LITM doc-count/position: $1/$2" >&2; return 2 ;;
  esac
}

check_litm_files() {
  local missing=0
  local count position gold_index path
  IFS=',' read -r -a counts <<< "$LITM_DOC_COUNTS"
  IFS=',' read -r -a positions <<< "$LITM_POSITIONS"
  for count in "${counts[@]}"; do
    count="${count//[[:space:]]/}"
    [[ -z "$count" ]] && continue
    for position in "${positions[@]}"; do
      position="${position//[[:space:]]/}"
      [[ -z "$position" ]] && continue
      gold_index="$(litm_gold_index "$count" "$position")" || return $?
      path="$LITM_DIR/${count}_total_documents/nq-open-${count}_total_documents_gold_at_${gold_index}.jsonl.gz"
      if [[ ! -f "$path" ]]; then
        echo "Missing LITM file: $path" >&2
        missing=1
      fi
    done
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Run scripts/download_litm_nq.py --output-dir \"$LITM_DIR\" --positions \"$LITM_POSITIONS\" before eval." >&2
    return 1
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/run_scratchpad_eval_suite.sh [options]

Trains the scratchpad model when the checkpoint is missing, then evaluates it.

Evaluates trained scratchpad checkpoint variants:
  - scratchpad_noscale
  - scratchpad_scaled
  - scratchpad_scaled_pos512

Each method runs multi-hop as-is and the representative LITM position selected
by --parallel-litm-positions.

Options:
  --checkpoint PATH              default: config output_dir
  --skip-train                   require checkpoint and only run eval
  --force-train                  train even if checkpoint already exists
  --train-jsonl PATH
  --train-eval-jsonl PATH
  --train-max-steps N
  --train-batch-size N
  --train-grad-accum-steps N
  --train-warmup-steps N
  --train-logging-steps N
  --train-max-examples N
  --train-max-eval-examples N
  --train-learning-rate FLOAT
  --train-scratchpad-learning-rate FLOAT
  --train-scratchpad-init-text TEXT
  --train-sparse-backend NAME    flash_block, sdpa_mask, eager_block, dense
  --train-position-strategy NAME
  --train-question-position-gap N
  --config PATH
  --base-model MODEL_OR_PATH
  --input-jsonl PATH
  --litm-dir DIR
  --output-dir DIR
  --max-examples N
  --limit-per-litm-file N
  --litm-doc-counts CSV          default: 10,20,30
  --litm-positions CSV           default: start,middle,end
  --parallel-litm-positions CSV  default: start
  --max-new-tokens N
  --max-context-tokens N
  --ape-temperature FLOAT
  --ape-scale FLOAT
  --cuda-device N               default: 3
  --dry-run
EOF
}

CHECKPOINT="${CHECKPOINT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --force-train) FORCE_TRAIN=1; shift ;;
    --train-jsonl) TRAIN_JSONL="$2"; shift 2 ;;
    --train-eval-jsonl) TRAIN_EVAL_JSONL="$2"; shift 2 ;;
    --train-max-steps) TRAIN_MAX_STEPS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --train-grad-accum-steps) TRAIN_GRAD_ACCUM_STEPS="$2"; shift 2 ;;
    --train-warmup-steps) TRAIN_WARMUP_STEPS="$2"; shift 2 ;;
    --train-logging-steps) TRAIN_LOGGING_STEPS="$2"; shift 2 ;;
    --train-max-examples) TRAIN_MAX_EXAMPLES="$2"; shift 2 ;;
    --train-max-eval-examples) TRAIN_MAX_EVAL_EXAMPLES="$2"; shift 2 ;;
    --train-learning-rate) TRAIN_LEARNING_RATE="$2"; shift 2 ;;
    --train-scratchpad-learning-rate) TRAIN_SCRATCHPAD_LEARNING_RATE="$2"; shift 2 ;;
    --train-scratchpad-init-text) TRAIN_SCRATCHPAD_INIT_TEXT="$2"; shift 2 ;;
    --train-sparse-backend) TRAIN_SPARSE_BACKEND="$2"; shift 2 ;;
    --train-position-strategy) TRAIN_POSITION_STRATEGY="$2"; shift 2 ;;
    --train-question-position-gap) TRAIN_QUESTION_POSITION_GAP="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --input-jsonl) INPUT_JSONL="$2"; shift 2 ;;
    --litm-dir) LITM_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --limit-per-litm-file) LIMIT_PER_LITM_FILE="$2"; shift 2 ;;
    --litm-doc-counts) LITM_DOC_COUNTS="$2"; shift 2 ;;
    --litm-positions) LITM_POSITIONS="$2"; shift 2 ;;
    --parallel-litm-positions) PARALLEL_LITM_POSITIONS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --max-context-tokens) MAX_CONTEXT_TOKENS="$2"; shift 2 ;;
    --ape-temperature) APE_TEMPERATURE="$2"; shift 2 ;;
    --ape-scale) APE_SCALE="$2"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(config_value output_dir outputs/scratchpad_multihop_qwen3_1_7b)"
fi
if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Missing multi-hop eval JSONL: $INPUT_JSONL" >&2
  echo "Run scripts/prepare_scratchpad_data.py first, or pass --input-jsonl." >&2
  exit 1
fi
if [[ ! -d "$LITM_DIR" ]]; then
  echo "Missing LITM directory: $LITM_DIR" >&2
  echo "Run scripts/download_litm_nq.py first, or pass --litm-dir." >&2
  exit 1
fi
check_litm_files

should_train=0
if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  if [[ ! -e "$CHECKPOINT" ]]; then
    echo "Scratchpad checkpoint does not exist and --skip-train was set: $CHECKPOINT" >&2
    exit 1
  fi
elif [[ "$FORCE_TRAIN" -eq 1 || ! -e "$CHECKPOINT" ]]; then
  should_train=1
else
  echo "Using existing scratchpad checkpoint: $CHECKPOINT" >&2
  echo "Pass --force-train to retrain before eval." >&2
fi

mkdir -p "$OUTPUT_DIR"

train_cmd=(
  python scripts/train_scratchpad.py
  --config "$CONFIG"
  --output-dir "$CHECKPOINT"
)

[[ -n "$BASE_MODEL" ]] && train_cmd+=(--model "$BASE_MODEL")
[[ -n "$TRAIN_JSONL" ]] && train_cmd+=(--train-jsonl "$TRAIN_JSONL")
[[ -n "$TRAIN_EVAL_JSONL" ]] && train_cmd+=(--eval-jsonl "$TRAIN_EVAL_JSONL")
[[ -n "$TRAIN_MAX_STEPS" ]] && train_cmd+=(--max-steps "$TRAIN_MAX_STEPS")
[[ -n "$TRAIN_BATCH_SIZE" ]] && train_cmd+=(--batch-size "$TRAIN_BATCH_SIZE")
[[ -n "$TRAIN_GRAD_ACCUM_STEPS" ]] && train_cmd+=(--grad-accum-steps "$TRAIN_GRAD_ACCUM_STEPS")
[[ -n "$TRAIN_WARMUP_STEPS" ]] && train_cmd+=(--warmup-steps "$TRAIN_WARMUP_STEPS")
[[ -n "$TRAIN_LOGGING_STEPS" ]] && train_cmd+=(--logging-steps "$TRAIN_LOGGING_STEPS")
[[ -n "$TRAIN_MAX_EXAMPLES" ]] && train_cmd+=(--max-train-examples "$TRAIN_MAX_EXAMPLES")
[[ -n "$TRAIN_MAX_EVAL_EXAMPLES" ]] && train_cmd+=(--max-eval-examples "$TRAIN_MAX_EVAL_EXAMPLES")
[[ -n "$TRAIN_LEARNING_RATE" ]] && train_cmd+=(--learning-rate "$TRAIN_LEARNING_RATE")
[[ -n "$TRAIN_SCRATCHPAD_LEARNING_RATE" ]] && train_cmd+=(--scratchpad-learning-rate "$TRAIN_SCRATCHPAD_LEARNING_RATE")
[[ -n "$TRAIN_SCRATCHPAD_INIT_TEXT" ]] && train_cmd+=(--scratchpad-init-text "$TRAIN_SCRATCHPAD_INIT_TEXT")
[[ -n "$TRAIN_SPARSE_BACKEND" ]] && train_cmd+=(--sparse-attention-backend "$TRAIN_SPARSE_BACKEND")
[[ -n "$TRAIN_POSITION_STRATEGY" ]] && train_cmd+=(--position-strategy "$TRAIN_POSITION_STRATEGY")
[[ -n "$TRAIN_QUESTION_POSITION_GAP" ]] && train_cmd+=(--question-position-gap "$TRAIN_QUESTION_POSITION_GAP")

eval_cmd=(
  python scripts/eval_scratchpad.py
  --config "$CONFIG"
  --methods "scratchpad_noscale,scratchpad_scaled,scratchpad_scaled_pos512"
  --checkpoint "$CHECKPOINT"
  --input-jsonl "$INPUT_JSONL"
  --litm-dir "$LITM_DIR"
  --output-jsonl "$OUTPUT_DIR/predictions.jsonl"
  --metrics-json "$OUTPUT_DIR/metrics.json"
  --litm-doc-counts "$LITM_DOC_COUNTS"
  --litm-positions "$LITM_POSITIONS"
  --parallel-litm-positions "$PARALLEL_LITM_POSITIONS"
  --order-variants "as_is"
)

[[ -n "$BASE_MODEL" ]] && eval_cmd+=(--base-model "$BASE_MODEL")
[[ -n "$MAX_EXAMPLES" ]] && eval_cmd+=(--max-examples "$MAX_EXAMPLES")
[[ -n "$LIMIT_PER_LITM_FILE" ]] && eval_cmd+=(--limit-per-litm-file "$LIMIT_PER_LITM_FILE")
[[ -n "$MAX_NEW_TOKENS" ]] && eval_cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$MAX_CONTEXT_TOKENS" ]] && eval_cmd+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -n "$APE_TEMPERATURE" ]] && eval_cmd+=(--ape-temperature "$APE_TEMPERATURE")
[[ -n "$APE_SCALE" ]] && eval_cmd+=(--ape-scale "$APE_SCALE")

if [[ "$should_train" -eq 1 ]]; then
  printf 'Training: CUDA_VISIBLE_DEVICES=%q' "$CUDA_DEVICE"
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
fi
printf 'Evaluating: CUDA_VISIBLE_DEVICES=%q' "$CUDA_DEVICE"
printf ' %q' "${eval_cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

if [[ "$should_train" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${train_cmd[@]}"
fi
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${eval_cmd[@]}"
