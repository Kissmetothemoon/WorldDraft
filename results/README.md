# Exported result data

Produced by our engines; nothing here was re-typed by hand except where noted.

| file | contents |
|---|---|
| `engine_speedup.csv` | long-format vLLM speedups parsed programmatically from `main_table.md` (324 rows = 18 languages × 3 concurrency levels × 6 methods). Columns: `concurrency,lang,method,speedup,tau,ar_tok_s`. `speedup` is relative to plain target decoding in the same engine; `tau` is the engine's mean acceptance length (draft + bonus); `ar_tok_s` is the baseline's aggregate tokens/s. `method` ∈ `ar_draft, eagle3, dflash_pub, dflash_pub_lex, dflash_ml, dflash_ml_lex`, where `*_lex` adds LexDraft and `dflash_ml` is the 50K multilingual distillation. English and the other Latin/Romanized conditions run with the prior off, so their `*_lex` rows are the same underlying run. |
| `main_table.md` | the human-readable version of the same table, plus the greedy output-identity matrix (share of requests identical with and without the prior). Header: vLLM nightly, Qwen3-8B, one NVIDIA L20, clean prompts, greedy. |
| `main_quality.md` | MGSM accuracy / translation chrF per configuration (losslessness check on the target's task quality). |
| `r9_pub_sel.md`, `r9_ftml_sel.md` | held-out lambda selection on the public and multilingual Qwen drafter: tau@15 on the test half for top-1, a fixed weight, the dev-selected global weight, and the per-language dev choice. |
| `r9_pub_selxw.md`, `r9_ftml_selxw.md` | the same with the cross-word bigram extension (the word-boundary ablation). |
| `prompts/prompts_eval_clean_q8.jsonl` | the filtered Qwen3-8B evaluation prompts (`id, task, lang, text`). |
| `prompts/prompts_eval_clean_ll8.jsonl` | the filtered Llama-3.1-8B evaluation prompts. |
| `prompts/clean_ids_{q8,ll8}.json` | the prompt ids kept after filtering (same list the replay runs used). |

Regenerate the CSV from a new `main_table.md` with, roughly:

```python
import re, csv
txt = open("main_table.md").read()
for conc_block in re.split(r"^## concurrency (\d+)$", txt, flags=re.M)[1:]:
    ...
```
`scripts/07_tables.sh` regenerates the markdown itself from the engine pkl files.

Raw per-request logs and pkl files are not shipped (they contain host paths).
The prompt text is redistributed under the licences listed in `../data/README.md`.
