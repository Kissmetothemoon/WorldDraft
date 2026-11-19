#!/usr/bin/env bash
# Step 3: sanity-check the Triton walk against the Python reference on real
# DFlash logits, and time it (expect kernel==ref 1.0000, plus ~0.1-1 ms per call).
# Needs a gen_vllm.py output file with prompt_ids/output_ids (scripts/03_generate.sh).
# Usage: ./scripts/03_check_kernel.sh <gen.jsonl> [langs]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GEN="${1:?path to a gen_*.jsonl file}"
LANGS="${2:-te,am,en}"
$PY research/test_lex_triton.py "$TARGET" "$DRAFT_PUB" "$PRIOR_DIR" "$GEN" "$LANGS" 0.75
