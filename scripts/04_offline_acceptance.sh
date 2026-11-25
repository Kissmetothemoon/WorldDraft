#!/usr/bin/env bash
# Step 4: offline held-out acceptance (tau) with and without the prior,
# on the cleaned replay prompts. Writes one pkl + log per shard.
#   work/pilot_lex/lex_<shard>.pkl
# Usage: ./scripts/04_offline_acceptance.sh [nshard] [lam]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NSHARD="${1:-2}"
LAM="${2:-0.5}"
for i in $(seq 0 $((NSHARD - 1))); do
  $PY worlddraft/sd_lex.py --target "$TARGET" --draft "$DRAFT_PUB" \
      --lex "$WORK/wiki/wiki_*.jsonl" --lam "$LAM" --off-langs en,de,fr,es,zh \
      --prompts "$WORK/prompts_eval_clean_q8.jsonl" --clean-ids "$CLEAN_IDS" \
      --per-group 12 --temps 0 --shard "$i" --nshard "$NSHARD" \
      --out "$WORK/pilot_lex/sd_$i.pkl" > "$WORK/pilot_lex/sd_$i.log" 2>&1
done
echo "done: $WORK/pilot_lex/sd_*.pkl"
