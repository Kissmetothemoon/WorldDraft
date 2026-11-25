<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/worlddraft-dark.svg">
  <img src="assets/brand/worlddraft.svg" alt="WorldDraft" width="340">
</picture>

Official code for *What Slows Parallel Diffusion Drafters Across Languages?*

[Explore measured speedups and inspect recorded draft blocks](https://kissmetothemoon.github.io/worlddraft-page/).

In **WorldDraft**, an inserted lowercase **l** splits *word* and spells *world*. The dashed letter marks the break between token pieces and points to the unequal cost of fragmentation across languages.

Parallel (block-diffusion) drafters such as DFlash propose a whole block of tokens
in one pass, but their speedup collapses in languages whose words are split into
many tokens: on Qwen3-8B the public DFlash checkpoint gives 2.80x in English and
1.08x in Telugu. An autoregressive drafter sees the subword pieces it has already
proposed; independent block positions see only the accepted prefix. WorldDraft
closes that *within-word visibility gap* without touching the drafter or the
target: a per-language table of word continuations, counted once from monolingual
text, reranks and expands the block candidate pool so that the pieces of a word
are chosen consistently with each other.

- **Training-free.** No gradients, no drafter retraining. One extra Triton kernel.
- **Cheap.** The table is built from 10M Wikipedia tokens per language; 2M already
  recovers most of the gain. The walk costs 0.13 ms per drafting step at batch 1
  (1.0 ms at batch 32) against 30–40 ms engine steps.
- **Lossless.** The target still verifies every token. Under greedy decoding,
  outputs are identical (2,106/2,106 paired batch-1 requests in the paper).
- **Portable.** Works with the public DFlash checkpoint and with a multilingually
  distilled one, on Qwen3-8B and Llama-3.1-8B.
- **Effective where it matters.** +50% speedup in Telugu and +44% in Amharic on
  the public drafter in vLLM at concurrency 1; gains shrink for English (below 1%).

## Method

For context `c` — up to the last three subword pieces of the *current word*,
reset at every word start — the table stores a smoothed continuation
distribution `p(x | c)` and a word-end probability `p(END | c)`, backed off to
shorter within-word contexts with an add-½ floor. At block position `k`:

1. Candidate pool = the drafter's top-16 tokens ∪ the prior's top-4 continuations.
2. Score each candidate

   ```
   s_k(x) = log q_k(x) + λ_l · ℓ_k(x),
   ℓ_k(x) = log p(x | c_k)        if x continues the word,
            log p(END | c_k)       if x starts a new word,
   ```

   where `q_k` is the drafter's block distribution and `λ_l ≥ 0` weights the
   prior for output language `l`.
3. Pick the argmax, update `c_k`, continue to the next block position (left to
   right, one Triton program per request).
4. The target verifies as usual. `λ_l = 0` reproduces the original drafter.
   Because normalized `exp s_k` is an explicit proposal distribution, the
   standard speculative acceptance / residual-resampling rule keeps sampling
   lossless.

The prior models a word's *continuation and boundary* and leaves the choice of the
next word to the neural drafter. A cross-word test adding a bigram at word starts
was worse in 60 of 72 comparisons (median −2.0 points), which is why the table
stops at the word boundary.

## Install

The paper used two environments.

```bash
git clone https://github.com/Kissmetothemoon/WorldDraft && cd WorldDraft
export WORLDDRAFT_ROOT=/path/to/workspace   # holds data/ and models/, see data/README.md

# (a) offline pipeline: HF speculative loop, tables, distillation, analysis
pip install -r requirements.txt           # torch 2.8.0, transformers 4.57.3, triton 3.4.0, ...

# (b) serving engine: a vLLM build that ships vllm/v1/spec_decode/dflash.py
pip install --pre "vllm==0.31.1rc1.dev253+g7d0b4e57a.cu129" \
    --index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129
```

Then point the entrypoints at both interpreters:

```bash
export WORLDDRAFT_PY=$VIRTUAL_ENV/bin/python          # env (a)
export WORLDDRAFT_ENGINE_PY=/path/to/env-b/bin/python # env (b)
```

`scripts/env.sh` also sets `PYTHONPATH=worlddraft`, so the flat scripts can
`from common import ...` as in the original codebase.

## Quickstart

```bash
./scripts/01_wiki_text.sh te,am,bn 1e7      # work/wiki/wiki_<lang>.jsonl
./scripts/02_build_prior.sh                 # work/lextab/<lang>.npz per language
./scripts/03_generate.sh                    # target trajectories -> replay set
./scripts/03_check_kernel.sh work/q8/gen_0.jsonl te,am,en
```

`03_check_kernel.sh` runs the Triton walk against the Python reference on real
DFlash logits and prints, for each language, the fraction of positions where the
kernel and the reference agree (expected 1.0000) and the per-call latency at
batch 1 / 8 / 32.

Then measure acceptance and speed:

```bash
./scripts/04_offline_acceptance.sh 2 0.5    # tau with/without the prior, held-out prompts
./scripts/05_select_lambda.sh 4             # dev-half lambda selection -> selection.md
./scripts/06_engine_bench.sh "te:0.5,am:1.5,bn:0.75" dml_lex
./scripts/07_tables.sh                      # main table + losslessness report
```

For the full grid in the paper, run `06_engine_bench.sh` once per configuration
(`ar`, `dpub`, `dpub_lex`, `dml`, `dml_lex`) with the per-language weights listed
in that script's header, and shard each run into `<tag>_<i>.pkl` files.
`results/main_table.md` is the expected output format.

## Repository map

```
worlddraft/     the method and the main pipeline
  common.py         language table, chat templates, token classification
  wiki_lex_prep.py  monolingual Wikipedia text -> tokenised jsonl
  lex_tables.py     counted tables -> GPU-ready .npz per language
  lex_triton.py     Triton walk (LexTables, lex_walk, lex_walk_ref)
  sd_lex.py         HF batch-1 speculative loop, greedy + sampling, lossless
  sd_lib.py         greedy spec loops used for wall-clock (DFlash, AR-draft)
  engine_lex.py     vLLM benchmark with WorldDraft inside the DFlash proposer
  engine_bench.py   stock vLLM benchmark (baselines)
  engine_profile.py per-step engine cost decomposition
  gen_vllm.py       target trajectories with vLLM
  build_*.py        prompt grids and filtered prompt lists
  acc_eval.py, main_summary.py, main_quality.py   task quality, main table
research/     the paper's analysis, mechanism study, pilots, distillation
              (see research/README.md for a script -> section table)
scripts/      numbered end-to-end entrypoints (env.sh sets the paths)
results/      exported result data: main table, quality, held-out selection,
              engine_speedup.csv, the cleaned prompt lists
data/         how to obtain MGSM / Aya / Wikipedia / checkpoints
```

## Results

`results/engine_speedup.csv` has one row per
`concurrency, lang, method, speedup, tau, ar_tok_s` cell, parsed from
`results/main_table.md` (vLLM nightly, Qwen3-8B, one NVIDIA L20, greedy, clean
prompts; speedup relative to plain target decoding in the same engine). Methods:
`ar_draft`, `eagle3`, `dflash_pub`, `dflash_pub_lex`, `dflash_ml`, `dflash_ml_lex`.
Concurrency 1, selected cells:

| language | AR draft | DFlash public | + WorldDraft | DFlash ml | + WorldDraft |
|---|---|---|---|---|---|
| en | 1.90x | 2.80x | 2.79x | 2.85x | 2.83x |
| th | 2.14x | 1.34x | 1.54x | 1.75x | 1.97x |
| bn | 2.23x | 1.39x | 1.83x | 1.92x | 2.34x |
| te | 2.38x | 1.08x | 1.62x | 2.24x | 2.62x |
| am | 1.98x | 1.32x | 1.89x | 2.01x | 2.66x |

Losslessness: in the batch-1 loop all 2,106 paired requests are token-identical
with and without WorldDraft. In vLLM, bf16 near-tie flips make some requests differ
under different batch/verification shapes — an engine effect that also appears
between plain target decoding and multilingual DFlash — while MGSM accuracy and
chrF stay within noise (`results/main_quality.md`).

## Limitations

Taken from the paper. vLLM speedups cover Qwen3-8B on one L20; Llama uses replay
and a speculative loop. The 18 conditions omit many scripts and domains. Serving
is greedy; sampling uses loop and target-NLL checks. A 512-token speed cap
truncates some MGSM answers; accuracy uses 2048. Romanized Hindi lacks matching
prior text; word boundaries depend on the tokenizer and segmenter.

## Acknowledgements

The DFlash drafter checkpoints and their remote code (`modeling_dflash.py`,
`dflash.py`, `utils.py`) come from [z-lab/dflash](https://github.com/z-lab/dflash)
(MIT); `sd_lib.py` adapts its `spec_generate` greedy loop. Engine runs use
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0). Targets: Qwen3 (Apache-2.0)
and Llama 3.1 (Llama 3.1 Community License). Evaluation data: MGSM, Aya, UltraChat,
GSM8K, HumanEval — see `data/README.md`.

## Citation

```bibtex
@article{feng2026worlddraft,
  title   = {What Slows Parallel Diffusion Drafters Across Languages?},
  author  = {Yuannuo Feng and Wenshuai Yao and Yizhe Chen and Ngai Wong and Wenyong Zhou and Wang Kang},
  year    = {2026},
  note    = {Under review. Code: \url{https://github.com/Kissmetothemoon/WorldDraft}},
  url     = {https://kissmetothemoon.github.io/worlddraft-page/}
}
```

Released under the MIT License (see `LICENSE`).
