# Data and checkpoints

Nothing large is committed. Everything below is downloaded or regenerated with
`WORLDDRAFT_ROOT` pointing at a working directory (default: the parent of this
repository, so the layout is `$WORLDDRAFT_ROOT/{code,work,data,models,state}` — the
scripts write to `$WORLDDRAFT_ROOT/work` and `$WORLDDRAFT_ROOT/state`).

```bash
export WORLDDRAFT_ROOT=/path/to/workspace   # must contain data/ and models/ (or symlinks)
```

## Evaluation data (`$WORLDDRAFT_ROOT/data/`)

| path | source | licence |
|---|---|---|
| `mgsm/<code>/test-00000-of-00001.parquet` | [MGSM](https://huggingface.co/datasets/juletxara/mgsm) (`shisa-ai/mgsm`-style export), used for the 11 native math conditions and as the translation source | CC BY-SA 4.0 (per MGSM's release) |
| `aya_dataset/**/*.parquet` | [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset), `train` split only; used as held-out dialogue prompts | Apache-2.0 |
| `ultrachat_200k/data/train_sft-*.parquet` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT |
| `gsm8k_train.parquet` (optional) | GSM8K train split, used when building distillation prompts | MIT |
| `openai_humaneval/` (optional) | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval), official-protocol prompt builder | MIT |

`build_prompts.py` reads exactly these paths. `prompts_eval_clean_q8.jsonl` and
`prompts_eval_clean_ll8.jsonl` in `results/prompts/` are the filtered prompt lists
the paper used (`build_clean_prompts.py` produces them from `prompts_eval.jsonl`).

Wikipedia text for the prior comes from `wikimedia/wikipedia`, `20231101.<lang>`,
streamed:
```bash
python worlddraft/wiki_lex_prep.py <target-tokenizer-or-model> $WORLDDRAFT_ROOT/work/wiki 10000000 te,am,bn,th,hi
# writes work/wiki/wiki_<lang>.jsonl, then:
python worlddraft/lex_tables.py <target-tokenizer-or-model> "$WORLDDRAFT_ROOT/work/wiki/wiki_*.jsonl" $WORLDDRAFT_ROOT/work/lextab 1e7
```
`lex_tables.py` produces `work/lextab/<lang>.npz` (~440 MB for 18 languages at 10M
tokens each; Japanese is the largest). These were **not** shipped — rebuild them.

## Model checkpoints (`$WORLDDRAFT_ROOT/models/`)

| role | checkpoint | notes |
|---|---|---|
| target | `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct` | Llama needs a gated HF token |
| public block drafter | `z-lab/Qwen3-8B-DFlash-b16`, `z-lab/Llama-3.1-8B-Instruct-DFlash-UltraChat` (block size 10) | `trust_remote_code=True`; MIT weights, code from [z-lab/dflash](https://github.com/z-lab/dflash) |
| multilingual block drafter ("DFlash ml") | distilled by us on 50K target responses | reproduce with `research/train_dflash.py` |
| AR drafter | `Qwen/Qwen3-0.6B` distilled by us; `meta-llama/Llama-3.2-1B-Instruct` distilled by us | reproduce with `research/train_ar_drafter.py` |
| EAGLE-3 baseline | `AngelSlim/Qwen3-4B_eagle3` and the matching released EAGLE-3 checkpoints | non-Latin draft-vocabulary coverage is discussed in the paper appendix |

Checkpoints are fetched with the Hugging Face hub client (a mirror is optional):

```bash
export HF_ENDPOINT=https://hf-mirror.com   # optional; unset to use huggingface.co
python - <<'PY'
import os
from huggingface_hub import snapshot_download
root = os.environ["WORLDDRAFT_ROOT"]
for repo in ["Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16"]:
    snapshot_download(repo, local_dir=f"{root}/models/{repo.split('/')[-1]}")
PY
```
