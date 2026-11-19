"""Candidate-word completions for the suffix-conflict (D, C) and length-divergence (L) measures.

For sampled word starts w of each trajectory, take the target's top-4 first-token candidates at w,
greedily extend each alternative by 12 tokens with the target (vLLM, prefix caching), and record
continuations and local word lengths. Analysis turns these into per-token D_j / C_j and per-word L.
"""
import argparse
import random
import time

from common import classify_tokens, pdump, pload, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tf", required=True, help="tf_dflash output shard (pickle)")
ap.add_argument("--gen", required=True, help="matching gen jsonl shard (for prompt ids)")
ap.add_argument("--out", required=True)
ap.add_argument("--n-multi", type=int, default=40)
ap.add_argument("--n-single", type=int, default=12)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--ext", type=int, default=12)
args = ap.parse_args()

from transformers import AutoTokenizer
from common import jload

tok = AutoTokenizer.from_pretrained(args.model)
bt = token_byte_table(tok)
tf = pload(args.tf)
gen = {r["id"]: r for r in jload(args.gen)}
rng = random.Random(0)


def local_wlen(prev, toks, lang):
    """Token length of the word started by toks[0] given preceding tokens prev (0 = continues prev word)."""
    c = classify_tokens(prev + toks, bt, lang)
    i = len(prev)
    if c["cls"][i] == 1:
        return c["wlen"][i]
    if c["cls"][i] == 2:
        return 0
    return 1


reqs, meta = [], []
for si, r in enumerate(tf):
    P, t0 = r["P"], r["t0"]
    out = r["output_ids"][: t0 - P]
    if len(out) < 4:
        continue
    c = classify_tokens(out, bt, r["lang"])
    multi = [k for k in range(len(out) - 1) if c["cls"][k] == 1 and c["wlen"][k] >= 2]
    single = [k for k in range(len(out) - 1) if c["cls"][k] == 1 and c["wlen"][k] == 1]
    rng.shuffle(multi)
    rng.shuffle(single)
    pick = sorted(multi[: args.n_multi] + single[: args.n_single])
    pref = gen[r["id"]]["prompt_ids"]
    for k in pick:
        cands = [int(x) for x in r["top5_id"][k][: args.k]]
        if cands[0] != out[k]:
            continue
        m = {"seq": si, "k": k, "cands": cands, "pi": [float(x) for x in r["top5_p"][k][: args.k]],
             "conts": [out[k + 1: k + 1 + args.ext]] + [None] * (len(cands) - 1)}
        meta.append(m)
        for ci, a in enumerate(cands[1:], 1):
            reqs.append((len(meta) - 1, ci, pref + out[:k] + [a]))

print(f"{len(meta)} word starts, {len(reqs)} requests", flush=True)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=4096,
          enable_prefix_caching=True, seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=args.ext)
t = time.time()
outs = llm.generate([TokensPrompt(prompt_token_ids=q) for _, _, q in reqs], sp, use_tqdm=False)
print(f"generation {time.time()-t:.0f}s", flush=True)
for (mi, ci, _), o in zip(reqs, outs):
    meta[mi]["conts"][ci] = list(o.outputs[0].token_ids)

for m in meta:
    r = tf[m["seq"]]
    out = r["output_ids"][: r["t0"] - r["P"]]
    prev = out[max(0, m["k"] - 8): m["k"]]
    m["wlens"] = [local_wlen(prev, [a] + (cont or []), r["lang"]) for a, cont in zip(m["cands"], m["conts"])]
    m["id"] = r["id"]
pdump(meta, args.out)
print("done", flush=True)

# vLLM's engine shutdown can hang after the output is written: kill our own child processes
# (VLLM::EngineCore etc.) first so nothing is left holding GPU memory, then exit immediately.
import multiprocessing as _mp, os as _os, signal as _sig, sys as _sys
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
