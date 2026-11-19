"""Batch-1 wall-clock: target AR baseline vs DFlash drafters vs AR-draft-model speculative decoding, per (task, lang).

--dflash name=path   (repeatable)      --ardraft name=path:K   (repeatable)
Every prompt runs the AR baseline (HF generate) first, then each method; outputs are compared to the baseline.
"""
import argparse
import time
from collections import defaultdict

import torch

from common import MAX_NEW, build_messages, chat_ids, jload, pdump, shard
from sd_lib import ar_sd_generate, dflash_generate

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dflash", action="append", default=[])
ap.add_argument("--ardraft", action="append", default=[])
ap.add_argument("--per-group", type=int, default=12)
ap.add_argument("--tasks", default="math,translate,chat")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
args = ap.parse_args()

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev,
                                              attn_implementation="sdpa").eval()
methods = []
for spec in args.dflash:
    name, path = spec.split("=", 1)
    m = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
    methods.append((name, "dflash", m, None))
for spec in args.ardraft:
    name, rest = spec.split("=", 1)
    path, K = rest.rsplit(":", 1)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=dev,
                                             attn_implementation="sdpa").eval()
    methods.append((name, "ar", m, int(K)))
stop = sorted({tok.eos_token_id} | {t for t in [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")]
                                    if isinstance(t, int) and 0 <= t < len(tok) and t != tok.unk_token_id})


def run(meth, pid, mx):
    name, kind, m, K = meth
    if kind == "dflash":
        return dflash_generate(target, m, pid, mx, stop)
    return ar_sd_generate(target, m, pid, mx, stop, K)


tasks = set(args.tasks.split(","))
groups = defaultdict(list)
for r in jload(args.prompts):
    if r["task"] in tasks:
        groups[(r["task"], r["lang"])].append(r)
sel = []
for key in sorted(groups):
    sel += groups[key][: args.per_group]
sel = shard(sel, args.shard, args.nshard)

warm = torch.tensor([chat_ids(tok, build_messages("chat", "en", "Hello!"))], device=dev)
target.generate(warm, max_new_tokens=16, do_sample=False, pad_token_id=stop[0])
for meth in methods:
    run(meth, warm, 32)

rep = []
for i, r in enumerate(sel):
    pid = torch.tensor([chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))], device=dev)
    mx = MAX_NEW[r["task"]]
    torch.cuda.synchronize(); t = time.time()
    ar = target.generate(pid, max_new_tokens=mx, do_sample=False, eos_token_id=stop, pad_token_id=stop[0])[0, pid.shape[1]:].tolist()
    torch.cuda.synchronize()
    rec = {"id": r["id"], "task": r["task"], "lang": r["lang"], "n_ar": len(ar), "t_ar": time.time() - t, "m": {}}
    for meth in methods:
        torch.cuda.synchronize(); t = time.time()
        gen, accs = run(meth, pid, mx)
        torch.cuda.synchronize()
        k = min(len(gen), len(ar))
        rec["m"][meth[0]] = {"n": len(gen), "t": time.time() - t, "accs": accs, "identical": gen == ar,
                             "same_prefix": next((j for j in range(k) if gen[j] != ar[j]), k)}
    rep.append(rec)
    if i % 10 == 0:
        msg = " ".join(f"{n}:{v['n']/v['t']:.0f}tok/s,tau{sum(v['accs'])/max(1,len(v['accs'])):.2f}" for n, v in rec["m"].items())
        print(f"[{i}/{len(sel)}] {r['id']} AR {rec['n_ar']/rec['t_ar']:.0f}tok/s {msg}", flush=True)
pdump(rep, args.out)
print("done", len(rep), flush=True)
