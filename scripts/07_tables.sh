#!/usr/bin/env bash
# Step 7: main table + losslessness report from the engine pkl files.
# main_summary.py expects one pkl per (config, shard) named <config>_<shard>.pkl:
#   ar, draft, eagle3, dpub, dpub_lex, dml, dml_lex
# Output columns: speedup vs AR in the same engine, engine tau, and the share of
# greedy requests whose output is identical with and without the prior.
# Usage: ./scripts/07_tables.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
$PY worlddraft/main_summary.py "$WORK/engine" "$WORK/main_table.md"
$PY worlddraft/main_quality.py "$WORK/engine" "$TARGET" "$WORK/main_quality.md"
echo "done: $WORK/main_table.md, $WORK/main_quality.md"
