#!/usr/bin/env bash
# Shared environment for the entrypoints. Override any of these before calling a script.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORLDDRAFT_ROOT="${WORLDDRAFT_ROOT:-${LEXDRAFT_ROOT:-$(dirname "$REPO")}}"
export PYTHONPATH="$REPO/worlddraft${PYTHONPATH:+:$PYTHONPATH}"

# Python for the offline / HF pipeline (torch, transformers, triton).
PY="${WORLDDRAFT_PY:-${LEXDRAFT_PY:-python3}}"
# Python for the vLLM engine scripts (a separate environment, see requirements.txt).
PY_ENGINE="${WORLDDRAFT_ENGINE_PY:-${LEXDRAFT_ENGINE_PY:-$PY}}"

# Models (default names match the paper's configurations).
TARGET="${TARGET:-$WORLDDRAFT_ROOT/models/Qwen3-8B}"
DRAFT_PUB="${DRAFT_PUB:-$WORLDDRAFT_ROOT/models/Qwen3-8B-DFlash-b16}"
DRAFT_ML="${DRAFT_ML:-$WORLDDRAFT_ROOT/models/ft_ml_q8}"        # our 50K multilingual distillation
PRIOR_DIR="${PRIOR_DIR:-$WORLDDRAFT_ROOT/work/lextab}"
WORK="${WORLDDRAFT_ROOT}/work"
PROMPTS="${PROMPTS:-$WORK/prompts_eval.jsonl}"
CLEAN_IDS="${CLEAN_IDS:-$WORK/clean_ids_q8.json}"

mkdir -p "$WORK"
