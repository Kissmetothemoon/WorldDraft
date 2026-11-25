"""Where does the time go in the optimised engine? Per-step decomposition of speculative decoding cost (vLLM nightly, V2 runner).

Same protocol as engine_bench.py (greedy, per (task, lang) group, chunks of `conc` requests), on top of it:
  pass "off": plain timing (tokens/s, engine tau, per-position acceptance)
  pass "on" : the first --prof-n prompts again with cuda-synchronised timers around
              GPUModelRunner.execute_model (target forward = verification),
              GPUModelRunner.sample (rejection sampling) and the rest of sample_tokens (drafting),
              one record per engine step: (n_reqs, n_tokens, n_spec, t_exec, t_sample, t_draft, t_wall)
--verify-sched a-b:k,...  uses vLLM dynamic SD (num_speculative_tokens_per_batch_size): the drafter still drafts --k tokens,
                          only the first k(batch size) are verified.
"""
import argparse
import json
import os
import time
from collections import defaultdict

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # engine in-process so the timers below see the model runner

import numpy as np
import torch

from common import MAX_NEW, build_messages, chat_ids, jload, pdump

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--method", default="none")
ap.add_argument("--spec-model", default=None)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--verify-sched", default="")
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conc", default="1,8,32")
ap.add_argument("--per-group", type=int, default=32)
ap.add_argument("--prof-n", type=int, default=32)
ap.add_argument("--tasks", default="math,translate,chat")
ap.add_argument("--langs", default="")
args = ap.parse_args()

from vllm.v1.worker.gpu import model_runner as MR

PROF = {"on": False, "steps": [], "cur": None}


def _sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


_exec, _sample, _st = MR.GPUModelRunner.execute_model, MR.GPUModelRunner.sample, MR.GPUModelRunner.sample_tokens


def execute_model(self, scheduler_output, *a, **kw):
    if not PROF["on"] or kw.get("dummy_run"):
        return _exec(self, scheduler_output, *a, **kw)
    t0 = _sync_time()
    out = _exec(self, scheduler_output, *a, **kw)
    t1 = _sync_time()
    n_spec = sum(len(v) for v in (scheduler_output.scheduled_spec_decode_tokens or {}).values())
    PROF["cur"] = {"t0": t0, "n_reqs": len(scheduler_output.num_scheduled_tokens),
                   "n_tokens": int(scheduler_output.total_num_scheduled_tokens), "n_spec": n_spec, "t_exec": t1 - t0,
                   "t_sample": 0.0}
    return out


def sample(self, *a, **kw):
    if not PROF["on"] or PROF["cur"] is None:
        return _sample(self, *a, **kw)
    t0 = _sync_time()
    out = _sample(self, *a, **kw)
    PROF["cur"]["t_sample"] += _sync_time() - t0
    return out


def sample_tokens(self, *a, **kw):
    if not PROF["on"] or PROF["cur"] is None:
        return _st(self, *a, **kw)
    t0 = _sync_time()
    out = _st(self, *a, **kw)
    t1 = _sync_time()
    c = PROF["cur"]
    c["t_draft"] = (t1 - t0) - c["t_sample"]  # propose() + bookkeeping
    c["t_end"] = t1
    PROF["steps"].append(c)
    PROF["cur"] = None
    return out


MR.GPUModelRunner.execute_model, MR.GPUModelRunner.sample, MR.GPUModelRunner.sample_tokens = execute_model, sample, sample_tokens

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

tok = AutoTokenizer.from_pretrained(args.model)
conc = [int(c) for c in args.conc.split(",")]
spec = None
if args.method != "none":
    spec = {"method": args.method, "model": args.spec_model, "num_speculative_tokens": args.k}
    if args.verify_sched:
        sched = []
        for part in args.verify_sched.split(","):
            rng, k = part.split(":")
            a, b = rng.split("-")
            sched.append((int(a), int(b), int(k)))
        spec["num_speculative_tokens_per_batch_size"] = sched
llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, max_num_seqs=max(conc),
          enable_prefix_caching=False, seed=0, speculative_config=spec, disable_log_stats=False)


def counters():
    v = {}
    for m in llm.get_metrics():
        if "spec_decode" in m.name:
            v[m.name] = list(m.values) if hasattr(m, "values") else m.value
    return v


def diff(a, b):
    d = {}
    for k, v in b.items():
        if isinstance(v, list):
            u = a.get(k, [0] * len(v))
            d[k] = [x - y for x, y in zip(v, u)]
        else:
            d[k] = v - a.get(k, 0)
    return d


groups = defaultdict(list)
tasks = set(args.tasks.split(","))
langs = set(args.langs.split(",")) if args.langs else None
for r in jload(args.prompts):
    if r["task"] in tasks and (langs is None or r["lang"] in langs):
        groups[(r["task"], r["lang"])].append(r)
llm.generate([TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages("chat", "en", "Hello!")))],
             SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
res = []
for key in sorted(groups):
    rows = groups[key][: args.per_group]
    ps = [TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))) for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW.get(key[0], 2048))
    for c in conc:
        before = counters()
        t, ntok, outs_all = time.time(), 0, []
        for i in range(0, len(ps), c):
            outs = llm.generate(ps[i:i + c], sp, use_tqdm=False)
            ntok += sum(len(o.outputs[0].token_ids) for o in outs)
            outs_all += [list(o.outputs[0].token_ids) for o in outs]
        dt = time.time() - t
        d = diff(before, counters())
        drafts = d.get("vllm:spec_decode_num_drafts", 0)
        tau = 1 + d.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts if drafts else float("nan")
        rec = {"task": key[0], "lang": key[1], "conc": c, "n": len(rows), "ids": [r["id"] for r in rows], "tokens": ntok,
               "time": dt, "tps": ntok / dt, "tau": tau, "counters": d, "out_ids": outs_all}
        # profiled pass
        PROF["on"], PROF["steps"] = True, []
        pp = ps[: args.prof_n]
        t = time.time()
        np_tok = 0
        for i in range(0, len(pp), c):
            outs = llm.generate(pp[i:i + c], sp, use_tqdm=False)
            np_tok += sum(len(o.outputs[0].token_ids) for o in outs)
        PROF["on"] = False
        st = PROF["steps"]
        rec["prof"] = {"tokens": np_tok, "time": time.time() - t,
                       "steps": np.array([[s["n_reqs"], s["n_tokens"], s["n_spec"], s["t_exec"], s["t_sample"], s.get("t_draft", 0.0),
                                           (st[i + 1]["t0"] - s["t0"]) if i + 1 < len(st) else np.nan] for i, s in enumerate(st)],
                                         dtype=np.float64)}
        res.append(rec)
        S = rec["prof"]["steps"]
        dec = S[S[:, 1] <= S[:, 0] * (args.k + 1)] if len(S) else S
        print(json.dumps({k: rec[k] for k in ("task", "lang", "conc", "tokens", "tps", "tau")}),
              f"steps {len(S)} exec {dec[:, 3].mean()*1e3 if len(dec) else 0:.2f}ms draft {dec[:, 5].mean()*1e3 if len(dec) else 0:.2f}ms "
              f"sample {dec[:, 4].mean()*1e3 if len(dec) else 0:.2f}ms", flush=True)
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
