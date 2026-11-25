#!/usr/bin/env bash
# Step 2: tokenised text -> GPU-ready count tables, one <lang>.npz each.
#   work/lextab/<lang>.npz   (ctx_keys, ctx_end, ctx_floor, ctx_top, pair_keys, pair_lp)
# Usage: ./scripts/02_build_prior.sh ["$WORK/wiki/wiki_*.jsonl"] [max-tokens-per-lang]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GLOB="${1:-$WORK/wiki/wiki_*.jsonl}"
MAXT="${2:-1e7}"
$PY worlddraft/lex_tables.py "$TARGET" "$GLOB" "$PRIOR_DIR" "$MAXT"
