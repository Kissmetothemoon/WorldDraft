#!/usr/bin/env bash
# Step 6: vLLM throughput with LexDraft inside the DFlash drafter.
# --lams is a per-language lambda list ("te:0.5,am:1.5,..."); missing languages run with the prior off.
# Repeat with --lex-dir "" and the same --method for the no-prior reference (that is how
# results/main_table.md was produced: ar / dpub / dpub_lex / dml / dml_lex at conc 1, 8, 32).
# Usage: ./scripts/06_engine_bench.sh <lams> <out-tag>
# The paper selected (dev half):
#   multilingual drafter: de:0.75,fr:0.5,es:0.75,ru:0.5,zh:0.25,ja:0.5,ko:0.75,ar:0.5,th:1.0,sw:1.0,hi:1.0,bn:0.75,te:0.5,am:1.5,sr-Cyrl:1.0,sr-Latn:1.0
#   public drafter:      same for Latin/Cyrillic, with ko:1.0,th:1.5,hi:1.5,bn:1.5,te:1.5,am:1.5,sr-Cyrl:1.5,sr-Latn:1.5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LAMS="${1:?e.g. te:0.5,am:1.5,bn:0.75}"
TAG="${2:?output tag, e.g. dml_lex}"
$PY_ENGINE lexdraft/engine_lex.py --model "$TARGET" --method dflash \
    --spec-model "$DRAFT_ML" --k 15 --verify-sched "1-4:15,5-16:7,17-64:3" \
    --lex-dir "$PRIOR_DIR" --lams "$LAMS" \
    --prompts "$WORK/prompts_eval_clean_q8.jsonl" \
    --conc 1,8,32 --per-group 32 --out "$WORK/engine/$TAG.pkl" \
    > "$WORK/engine/$TAG.log" 2>&1
echo "done: $WORK/engine/$TAG.pkl"
