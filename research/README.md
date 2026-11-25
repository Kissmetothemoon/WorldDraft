# Research scripts

Everything here is the analysis, mechanism-study, and baseline code behind the
paper's figures and tables. None of it is needed to *use* WorldDraft — see
`../worlddraft/` and `../scripts/` for the method itself. They read/write under
`$WORLDDRAFT_ROOT/work/<tag>/` and share `common.py` through `PYTHONPATH=worlddraft`
(honoured by `../scripts/env.sh`).

```bash
export WORLDDRAFT_ROOT=/path/to/workspace PYTHONPATH=$PWD/worlddraft
```

Scripts take positional arguments (see each docstring); the table below points at
the paper part each one produces.

## Study setup and data

| script | what it does | paper |
|---|---|---|
| `build_prompts.py` *(in `worlddraft/`)* | language x task prompt grid from MGSM / Aya | §3 |
| `build_train_prompts.py` | content-matched English vs multilingual distillation prompts | App. protocol |
| `build_extra_prompts.py` | extra dialogue prompts for cells with few samples | — |
| `build_official_prompts.py` | GSM8K / MATH-500 / HumanEval in the official DFlash format | App. protocol |
| `content_matched.py` | translate-task view on one shared 250-question source | §5 task dependence |
| `check_alignment.py` | baseline-alignment gate against published checkpoints | App. protocol |
| `official_summary.py` | official-protocol reproduction summary | App. protocol |
| `token_type_stats.py` | per-token-type statistics (raw, not entropy-standardised) | §4 |

## The reversal and the within-word gap

| script | what it does | paper |
|---|---|---|
| `compare_drafters.py` | AR vs block drafter on identical trajectories; fertility/entropy regression | §4, Fig. reversal |
| `analyze.py` | aggregate tf/cand/eagle/speed pkls into the stage report | §4–5 |
| `tf_dflash.py` | teacher-forced DFlash analysis (acceptance, deviations, invalid UTF-8) | §4–5 |
| `deficit_decomp.py` | where the block drafter's excess error sits | §5 |
| `mi_probe.py`, `mi_analysis.py` | within-block mutual-information probe and its effect on excess error | §5, App. MI |
| `cand_vllm.py` | suffix-conflict / length-divergence measures over candidates | §5 |
| `reveal_summary.py` | word-start reveal intervention (Δ_start) | §5, Fig. mechanism (b) |
| `ar_offline.py` | offline greedy acceptance of an independent AR drafter | §4 |
| `align_ar_sd.py` | our spec loop vs HF assisted generation on the same models | App. protocol |
| `eagle_vllm.py` | EAGLE-3 acceptance per condition in vLLM | §4, Table 1 |
| `speed_bench.py` | batch-1 wall clock: target, DFlash, AR-draft spec decoding | Table 1 |

## Candidate selection, data dose, and layout pilots

| script | what it does | paper |
|---|---|---|
| `pilot_lex.py` | **the prior study**: tau@15 for every lambda / expansion variant at once | §6, App. ablation |
| `lex_select.py` | dev-half weight selection + held-out report | §6 |
| `pilot_summary.py` | merge variant tables | §6 |
| `pilot_heads.py` | learned adjacent-pair selector (the "Sel" in the ablation) | §6, Fig. selector |
| `pilot_chunk.py` | joint in-word chunk prediction on a frozen drafter (route C') | — |
| `pilot_pard.py` | parallel drafter from a small multilingual LM (route M3) | — |
| `pilot_fusion.py` | per-position fusion ceiling (route D) | — |
| `pilot_cpu.py`, `pilot_compare.py` | CPU pilots / paired drafter comparison on stored records | — |
| `solve_summary.py` | per-language ceilings of block-level fixes | App. headroom |
| `eval_unit.py`, `eval_unit_summary.py`, `test_unit_consistency.py` | token-layout vs unit-layout drafter evaluation | — |

## Distillation and engines

| script | what it does | paper |
|---|---|---|
| `train_dflash.py` | fine-tune a released DFlash drafter on target responses (`torchrun`) | §5 data dose, App. |
| `train_ar_drafter.py` | distil the AR drafter (Qwen3-0.6B / Llama-3.2-1B) | §3 |
| `train_unit.py` | unit-aligned block layout pilot (route M1) | — |
| `engine_bench.py` *(in `worlddraft/`)* | stock vLLM throughput per drafter / concurrency | Table 1 |
| `engine_profile.py` *(in `worlddraft/`)* | per-step cost decomposition of speculative decoding | §5 serving cost |
| `prof_summary.py`, `prof_exact.py` | engine cost decomposition, untimed wall clock | App. cost |
| `round3_summary.py` | the round-3 engine / HF / per-language summaries | Table 1 |
| `tau_summary.py` | acceptance length per condition from real spec runs | App. |
| `round3_summary.py` | engine speedups (same-engine AR baseline), HF-level speedups, per-language H0 (Llama) | Table 1, §6 |
| `test_lex_triton.py` | validate the Triton walk against the Python reference, time it | App. runtime |
| `sched.py` | tiny dependency-aware GPU job scheduler used to run the grid | — |

Scripts marked "route …" are earlier design explorations that the paper does not
report as results; they are included so the negative direction is reproducible.
