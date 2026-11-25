"""Optimised-engine (vLLM, CUDA graphs) throughput per (task, lang) at several concurrency levels.

--method none | draft_model | eagle3 | dflash      (dflash needs a vLLM build that provides it)
For each concurrency c, prompts of one (task, lang) group are submitted in chunks of c (c=1: one request at a time).
Records output tokens / wall time and the engine's own acceptance length tau = 1 + accepted / drafts.
"""
import argparse
import json
import time
from collections import defaultdict

from common import MAX_NEW, build_messages, chat_ids, jload, pdump

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--method", default="none")
ap.add_argument("--spec-model", default=None)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conc", default="1,8,32")
ap.add_argument("--per-group", type=int, default=32)
ap.add_argument("--tasks", default="math,translate,chat")
ap.add_argument("--langs", default="")
ap.add_argument("--max-new", type=int, default=0)
args = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

tok = AutoTokenizer.from_pretrained(args.model)
conc = [int(c) for c in args.conc.split(",")]
spec = None
if args.method != "none":
    spec = {"method": args.method, "model": args.spec_model, "num_speculative_tokens": args.k}
llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, max_num_seqs=max(conc),
          enable_prefix_caching=False, seed=0, speculative_config=spec, disable_log_stats=False)


def counters():
    v = {}
    for m in llm.get_metrics():
        if "spec_decode" in m.name and not hasattr(m, "values"):
            v[m.name] = m.value
    return v


groups = defaultdict(list)
tasks = set(args.tasks.split(","))
langs = set(args.langs.split(",")) if args.langs else None
for r in jload(args.prompts):
    if r["task"] in tasks and (langs is None or r["lang"] in langs):
        groups[(r["task"], r["lang"])].append(r)
# warm-up (CUDA graph capture etc.)
llm.generate([TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages("chat", "en", "Hello!")))],
             SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
res = []
for key in sorted(groups):
    rows = groups[key][: args.per_group]
    ps = [TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))) for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new or MAX_NEW.get(key[0], 2048))
    for c in conc:
        before = counters()
        t, ntok = time.time(), 0
        for i in range(0, len(ps), c):
            outs = llm.generate(ps[i:i + c], sp, use_tqdm=False)
            ntok += sum(len(o.outputs[0].token_ids) for o in outs)
        dt = time.time() - t
        after = counters()
        d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        drafts = d.get("vllm:spec_decode_num_drafts", 0)
        tau = 1 + d.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts if drafts else float("nan")
        res.append({"task": key[0], "lang": key[1], "conc": c, "n": len(rows), "tokens": ntok, "time": dt,
                    "tps": ntok / dt, "tau": tau})
        print(json.dumps(res[-1]), flush=True)
pdump({"args": vars(args), "res": res}, args.out)

import multiprocessing as _mp, os as _os, sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
for _c in _mp.active_children():
    _c.kill()
try:
    import psutil as _ps
    for _c in _ps.Process().children(recursive=True):
        _c.kill()
except Exception:
    pass
_os._exit(0)
