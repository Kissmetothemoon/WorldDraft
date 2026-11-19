"""Offline greedy acceptance for an independent autoregressive draft model (vanilla speculative decoding).

Under greedy decoding, while every drafted token so far matched the target, the draft model's context equals the
true prefix. So the accepted length from anchor s (chain of K drafted tokens) is the number of consecutive
positions t > s whose draft prediction given the true prefix x[:t] equals the target's token. One teacher-forced
forward of the draft model over the trajectory gives all anchors exactly.

Gold tokens / deviation flags / entropy are reused from a DFlash tf record of the SAME trajectories, and the output
keeps that record format (corr matrix (R, K)), so analyze.py works unchanged.
"""
import argparse
import time

import numpy as np
import torch

from common import jload, pdump, pload

ap = argparse.ArgumentParser()
ap.add_argument("--draft", required=True)
ap.add_argument("--tf", required=True, help="DFlash tf_*.pkl of the same trajectories (gold/dev/entropy)")
ap.add_argument("--gen", required=True, help="matching gen jsonl (prompt ids)")
ap.add_argument("--out", required=True)
ap.add_argument("--K", type=int, default=15)
args = ap.parse_args()

from transformers import AutoModelForCausalLM

dev = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.bfloat16, device_map=dev,
                                             attn_implementation="sdpa").eval()
gen = {r["id"]: r for r in jload(args.gen)}
recs = pload(args.tf)
out, t0 = [], time.time()
K = args.K
for i, r in enumerate(recs):
    pid = gen[r["id"]]["prompt_ids"]
    ids = pid + r["output_ids"]
    P, N = len(pid), len(ids)
    assert P == r["P"] and N == r["N"], (r["id"], P, r["P"], N, r["N"])
    with torch.inference_mode():
        lg = model(torch.tensor([ids], device=dev)).logits[0, P - 1:N - 1]
        pr = torch.softmax(lg.float(), -1)
        pm, pred = pr.max(-1)
    pred, pm = pred.cpu().numpy(), pm.cpu().numpy()
    gold = np.asarray(r["top5_id"])[:, 0]
    devm = np.asarray(r["dev"], dtype=bool)
    T = N - P
    match = (pred == gold)
    R = N - 1 - P
    a = np.arange(R)[:, None]
    tidx = a + np.arange(1, K + 1)[None, :]          # response index of the k-th drafted token from anchor a
    inb = tidx <= T - 1
    tc = np.clip(tidx, 0, T - 1)
    d = devm[tc] & inb
    prior = np.cumsum(d, 1) - d
    valid = inb & (prior == 0)
    corr = np.where(valid, match[tc].astype(np.uint8), np.uint8(2))
    rr = dict(r)
    rr["corr"] = corr
    rr["dprob"] = pm[tc].astype(np.float16)
    out.append(rr)
    if i % 200 == 0:
        print(f"[{i}/{len(recs)}] {time.time()-t0:.0f}s", flush=True)
pdump(out, args.out)
print("done", len(out), f"{time.time()-t0:.0f}s", flush=True)
