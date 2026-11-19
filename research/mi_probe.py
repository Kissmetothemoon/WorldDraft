"""Within-block mutual information probe (target-side, drafter-independent).

For anchors s on a trajectory, the target samples M continuations of length KMAX at temperature 1 with top-L logprobs.
For offset k, with p_m = p(. | x_<=s, sampled x_{s+1..s+k-1}^(m)):
    Hc_k  = mean_m H(p_m)              (conditional entropy: what an AR drafter faces)
    Hmix_k = H(mean_m p_m)             (marginal entropy over the undrafted block: what a block-parallel drafter faces)
    I_k   = Hmix_k - Hc_k  >= 0        = I(x_{s+k}; x_{s+1..s+k-1} | x_<=s)
Top-L probabilities are kept per sample; the remaining mass is one shared 'tail' bucket (a lower-bound approximation,
applied identically to both entropies). k=1 gives I=0 up to sampling noise (sanity check).
"""
import argparse
import math
import random
import time

import numpy as np

from common import jload, lang_ok, pdump, shard

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gen", required=True, help="trajectory jsonl (prompt_ids, output_ids, text_out)")
ap.add_argument("--out", required=True)
ap.add_argument("--anchors", type=int, default=12)
ap.add_argument("--M", type=int, default=32)
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--L", type=int, default=20)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
args = ap.parse_args()

rows = shard(jload(args.gen), args.shard, args.nshard) if args.nshard > 1 else jload(args.gen)
rng = random.Random(0)
reqs, meta = [], []
for si, r in enumerate(rows):
    out = r["output_ids"]
    T = len(out)
    g4 = [tuple(out[i:i + 4]) for i in range(max(0, T - 3))]
    if T < args.kmax + 4 or not lang_ok(r["text_out"], r["lang"]) or 1 - len(set(g4)) / max(1, len(g4)) > 0.5:
        continue
    P = len(r["prompt_ids"])
    cand = list(range(0, T - args.kmax - 1))  # response index of the anchor token
    for a in sorted(rng.sample(cand, min(args.anchors, len(cand)))):
        reqs.append(r["prompt_ids"] + out[: a + 1])
        meta.append({"id": r["id"], "task": r["task"], "lang": r["lang"], "a": a})
print(f"{len(meta)} anchors from {len(rows)} sequences", flush=True)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=4096,
          enable_prefix_caching=True, seed=0, max_logprobs=args.L)
sp = SamplingParams(n=args.M, temperature=1.0, max_tokens=args.kmax, logprobs=args.L, seed=0)
t0 = time.time()
outs = llm.generate([TokensPrompt(prompt_token_ids=q) for q in reqs], sp, use_tqdm=False)
print(f"sampling {time.time()-t0:.0f}s", flush=True)


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


for m, o in zip(meta, outs):
    Hc = np.full(args.kmax, np.nan)
    Hm = np.full(args.kmax, np.nan)
    tail = np.full(args.kmax, np.nan)
    for k in range(args.kmax):
        dists = []
        for c in o.outputs:
            if c.logprobs is None or len(c.logprobs) <= k:
                continue
            lp = c.logprobs[k]
            d = {tid: math.exp(v.logprob) for tid, v in lp.items()}
            s = sum(d.values())
            d[-1] = max(0.0, 1.0 - s)  # tail bucket
            dists.append(d)
        if len(dists) < 2:
            continue
        hc = [ent(np.array(list(d.values()))) for d in dists]
        mix = {}
        for d in dists:
            for t, p in d.items():
                mix[t] = mix.get(t, 0.0) + p / len(dists)
        Hc[k] = float(np.mean(hc))
        Hm[k] = ent(np.array(list(mix.values())))
        tail[k] = float(np.mean([d[-1] for d in dists]))
    m["Hc"], m["Hmix"], m["I"], m["tail"] = Hc, Hm, Hm - Hc, tail
pdump(meta, args.out)
print("done", flush=True)

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
