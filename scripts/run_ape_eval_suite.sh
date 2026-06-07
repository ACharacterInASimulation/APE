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
LIMIT_PER_LITM_FILE=""
LITM_DOC_COUNTS="10,20,30"
LITM_POSITIONS="start,middle,end"
PARALLEL_LITM_POSITIONS="start"
ORDER_VARIANTS="as_is,gold_start,gold_middle,gold_end"
MAX_NEW_TOKENS=""
MAX_CONTEXT_TOKENS=""
APE_TEMPERATURE=""
APE_SCALE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_ape_eval_suite.sh [options]

Runs:
  - ape_scaled on LITM representative position + multi-hop as-is
  - ape_scaled_pos64 on LITM representative position + multi-hop as-is
  - ape_scaled_pos128 on LITM representative position + multi-hop as-is
  - ape_scaled_pos512 on LITM representative position + multi-hop as-is
  - decoder on LITM start/middle/end + multi-hop order variants

Options:
  --config PATH
  --base-model MODEL_OR_PATH
  --decoder-checkpoint PATH
  --input-jsonl PATH
  --litm-dir DIR
  --output-dir DIR
  --max-examples N
  --limit-per-litm-file N
  --litm-doc-counts CSV          default: 10,20,30
  --litm-positions CSV           default: start,middle,end
  --parallel-litm-positions CSV  default: start
  --order-variants CSV           default: as_is,gold_start,gold_middle,gold_end
  --max-new-tokens N
  --max-context-tokens N
  --ape-temperature FLOAT
  --ape-scale FLOAT
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

mkdir -p "$OUTPUT_DIR"

cmd=(
  python scripts/eval_scratchpad.py
  --config "$CONFIG"
  --methods "ape_scaled,ape_scaled_pos64,ape_scaled_pos128,ape_scaled_pos512,decoder"
  --input-jsonl "$INPUT_JSONL"
  --litm-dir "$LITM_DIR"
  --output-jsonl "$OUTPUT_DIR/predictions.jsonl"
  --metrics-json "$OUTPUT_DIR/metrics.json"
  --litm-doc-counts "$LITM_DOC_COUNTS"
  --litm-positions "$LITM_POSITIONS"
  --parallel-litm-positions "$PARALLEL_LITM_POSITIONS"
  --order-variants "$ORDER_VARIANTS"
)

[[ -n "$BASE_MODEL" ]] && cmd+=(--base-model "$BASE_MODEL")
[[ -n "$DECODER_CHECKPOINT" ]] && cmd+=(--decoder-checkpoint "$DECODER_CHECKPOINT")
[[ -n "$MAX_EXAMPLES" ]] && cmd+=(--max-examples "$MAX_EXAMPLES")
[[ -n "$LIMIT_PER_LITM_FILE" ]] && cmd+=(--limit-per-litm-file "$LIMIT_PER_LITM_FILE")
[[ -n "$MAX_NEW_TOKENS" ]] && cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "$MAX_CONTEXT_TOKENS" ]] && cmd+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -n "$APE_TEMPERATURE" ]] && cmd+=(--ape-temperature "$APE_TEMPERATURE")
[[ -n "$APE_SCALE" ]] && cmd+=(--ape-scale "$APE_SCALE")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${cmd[@]}"
