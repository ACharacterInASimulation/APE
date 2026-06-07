#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/scratchpad_multihop.yaml"
INPUT_JSONL="data/scratchpad_multihop/eval.jsonl"
LITM_DIR="data/litm_nq"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/eval_suites/ape_decoder_${RUN_ID}"
BASE_MODEL=""
DECODER_CHECKPOINT=""
MAX_EXAMPLES=""
LIMIT_PER_LITM_FILE="1000"
LITM_DOC_COUNTS="20"
LITM_POSITIONS="start,middle,end"
PARALLEL_LITM_POSITIONS="start"
ORDER_VARIANTS="as_is"
MAX_NEW_TOKENS=""
MAX_CONTEXT_TOKENS=""
APE_TEMPERATURE=""
APE_SCALE=""
CUDA_DEVICE="${CUDA_DEVICE:-2}"
DRY_RUN=0

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
Usage: scripts/run_ape_eval_suite.sh [options]

Runs:
  - ape_scaled on LITM representative position + multi-hop as-is
  - ape_scaled_pos64 on LITM representative position + multi-hop as-is
  - ape_scaled_pos128 on LITM representative position + multi-hop as-is
  - ape_scaled_pos512 on LITM representative position + multi-hop as-is
  - decoder on LITM start/middle/end + multi-hop as-is

Options:
  --config PATH
  --base-model MODEL_OR_PATH
  --decoder-checkpoint PATH
  --input-jsonl PATH
  --litm-dir DIR
  --output-dir DIR
  --max-examples N
  --limit-per-litm-file N        default: 1000
  --litm-doc-counts CSV          default: 20
  --litm-positions CSV           default: start,middle,end
  --parallel-litm-positions CSV  default: start
  --order-variants CSV           default: as_is
  --max-new-tokens N
  --max-context-tokens N
  --ape-temperature FLOAT
  --ape-scale FLOAT
  --cuda-device N               default: 2
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --decoder-checkpoint) DECODER_CHECKPOINT="$2"; shift 2 ;;
    --input-jsonl) INPUT_JSONL="$2"; shift 2 ;;
    --litm-dir) LITM_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    --limit-per-litm-file) LIMIT_PER_LITM_FILE="$2"; shift 2 ;;
    --litm-doc-counts) LITM_DOC_COUNTS="$2"; shift 2 ;;
    --litm-positions) LITM_POSITIONS="$2"; shift 2 ;;
    --parallel-litm-positions) PARALLEL_LITM_POSITIONS="$2"; shift 2 ;;
    --order-variants) ORDER_VARIANTS="$2"; shift 2 ;;
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

mkdir -p "$OUTPUT_DIR"

common_args=(
  python scripts/eval_scratchpad.py
  --config "$CONFIG"
  --input-jsonl "$INPUT_JSONL"
  --litm-dir "$LITM_DIR"
  --litm-doc-counts "$LITM_DOC_COUNTS"
  --litm-positions "$LITM_POSITIONS"
  --parallel-litm-positions "$PARALLEL_LITM_POSITIONS"
  --order-variants "$ORDER_VARIANTS"
)

[[ -n "$BASE_MODEL" ]] && common_args+=(--base-model "$BASE_MODEL")
[[ -n "$DECODER_CHECKPOINT" ]] && common_args+=(--decoder-checkpoint "$DECODER_CHECKPOINT")
[[ -n "$MAX_EXAMPLES" ]] && common_args+=(--max-examples "$MAX_EXAMPLES")
[[ -n "$LIMIT_PER_LITM_FILE" ]] && common_args+=(--limit-per-litm-file "$LIMIT_PER_LITM_FILE")
[[ -n "$MAX_NEW_TOKENS" ]] && common_args+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$MAX_CONTEXT_TOKENS" ]] && common_args+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -n "$APE_TEMPERATURE" ]] && common_args+=(--ape-temperature "$APE_TEMPERATURE")
[[ -n "$APE_SCALE" ]] && common_args+=(--ape-scale "$APE_SCALE")

methods=(ape_scaled ape_scaled_pos64 ape_scaled_pos128 ape_scaled_pos512 decoder)
for method in "${methods[@]}"; do
  method_cmd=(
    "${common_args[@]}"
    --methods "$method"
    --output-jsonl "$OUTPUT_DIR/${method}.predictions.jsonl"
    --metrics-json "$OUTPUT_DIR/${method}.metrics.json"
  )
  printf 'Planned: CUDA_VISIBLE_DEVICES=%q' "$CUDA_DEVICE"
  printf ' %q' "${method_cmd[@]}"
  printf '\n'
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

pids=()
for method in "${methods[@]}"; do
  method_cmd=(
    "${common_args[@]}"
    --methods "$method"
    --output-jsonl "$OUTPUT_DIR/${method}.predictions.jsonl"
    --metrics-json "$OUTPUT_DIR/${method}.metrics.json"
  )
  printf 'Launching %s on CUDA_VISIBLE_DEVICES=%q\n' "$method" "$CUDA_DEVICE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${method_cmd[@]}" > "$OUTPUT_DIR/${method}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Method failed: ${methods[$index]} (see $OUTPUT_DIR/${methods[$index]}.log)" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

from scripts.eval_scratchpad import summarize

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.predictions.jsonl")):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
    json.dump(summarize(rows), handle, indent=2)
print(f"combined predictions {out_dir / 'predictions.jsonl'}")
print(f"combined metrics {out_dir / 'metrics.json'}")
PY
