#!/usr/bin/env bash
# Build the evaluation prompt grid, then generate the target's greedy outputs
# (these trajectories are the held-out replay set for the offline checks).
# Usage: ./scripts/03_generate.sh [max-new]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
$PY worlddraft/build_prompts.py
$PY_ENGINE worlddraft/gen_vllm.py --model "$TARGET" --prompts "$PROMPTS" \
    --max-new "${1:-0}" --out "$WORK/q8/gen_0.jsonl"
$PY worlddraft/build_clean_prompts.py q8      # writes prompts_eval_clean_q8.jsonl + clean_ids_q8.json
