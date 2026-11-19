#!/usr/bin/env bash
# Step 5: candidate study / weight selection on the replay set.
# pilot_lex.py evaluates tau@15 for every lambda and expansion variant at once;
# lex_select.py then picks lambda per language on one half of the prompts and
# reports the held-out half (this is how the paper's lambda_l were chosen).
# Usage: ./scripts/05_select_lambda.sh [nshard]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NSHARD="${1:-4}"
for i in $(seq 0 $((NSHARD - 1))); do
  $PY research/pilot_lex.py --target "$TARGET" --draft "$DRAFT_PUB" \
      --lex "$WORK/wiki/wiki_*.jsonl" --eval "$WORK/prompts_eval_clean_q8.jsonl" \
      --clean-ids "$CLEAN_IDS" --per-group 80 --shard "$i" --nshard "$NSHARD" \
      --out "$WORK/pilot_lex/pilot_$i.pkl" > "$WORK/pilot_lex/pilot_$i.log" 2>&1
done
$PY research/lex_select.py "$WORK/pilot_lex/selection.md" "$WORK/pilot_lex/pilot_*.pkl"
echo "done: $WORK/pilot_lex/selection.md"
