#!/usr/bin/env bash
# Step 1: monolingual Wikipedia text -> tokenised jsonl
#   work/wiki/wiki_<lang>.jsonl
# Usage: ./scripts/01_wiki_text.sh ["te,am,bn"] [tokens-per-language]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LANGS="${1:-te,am,bn,th,hi}"
TOKENS="${2:-1e7}"
$PY worlddraft/wiki_lex_prep.py "$TARGET" "$WORK/wiki" "$TOKENS" "$LANGS"
