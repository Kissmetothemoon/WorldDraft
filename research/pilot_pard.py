"""Pilot for method route M3: a parallel (block) drafter built from a small multilingually PRE-TRAINED LM (PARD-style).

One forward predicts K future tokens: the regular causal output at anchor s gives x_{s+1}; K-1 mask tokens appended for
anchor s (positions s+1..s+K-1, attending to context <= s and to earlier masks of the same group) give x_{s+2..s+K}.
Many anchors share one forward through a 4D attention mask. No target features (target-independent drafter).
Training: target-generated responses (same files as the DFlash / AR distillation). Evaluation: offline greedy acceptance
on the evaluation trajectories, written in the tf-record format (gold / deviation from the DFlash base3 records), so
analyze.py gives tau per (task, lang) directly comparable with DFlash (ftml) and the AR drafter (ar06_ml).
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import jload, pdump, pload

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--max-train", type=int, default=50000)
ap.add_argument("--anchors", type=int, default=32)
ap.add_argument("--max-len", type=int, default=1536)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--gold-tf", required=True, help="glob of DFlash tf records of the eval trajectories (gold/dev)")
ap.add_argument("--gen", required=True, help="glob of the matching gen jsonl (prompt ids)")
ap.add_argument("--clean-ids", required=True)
ap.add_argument("--eval-per-group", type=int, default=100)
ap.add_argument("--out", required=True, help="output dir (tf_0.pkl + model)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.init)
model = AutoModelForCausalLM.from_pretrained(args.init, torch_dtype=torch.float32, device_map=dev, attn_implementation="sdpa")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
MASK = tok.convert_tokens_to_ids("<|fim_pad|>")
assert isinstance(MASK, int) and MASK >= 0
K = args.K


def build(ids, anchors):
    """ids list (len N), anchors (absolute positions). Returns input ids, position ids, bool 4D mask, and for every
    (anchor, offset k=1..K) the row index in the output whose logits predict x_{s+k}."""
    N, A = len(ids), len(anchors)
    G = K - 1
    L = N + A * G
    inp = torch.empty(L, dtype=torch.long)
    inp[:N] = torch.tensor(ids)
    inp[N:] = MASK
    pos = torch.empty(L, dtype=torch.long)
    pos[:N] = torch.arange(N)
    mask = torch.zeros(L, L, dtype=torch.bool)
    mask[:N, :N] = torch.tril(torch.ones(N, N, dtype=torch.bool))
    rows = torch.empty(A, K, dtype=torch.long)
    for i, s in enumerate(anchors):
        b = N + i * G
        pos[b:b + G] = torch.arange(s + 1, s + 1 + G)
        mask[b:b + G, :s + 1] = True
        mask[b:b + G, b:b + G] = torch.tril(torch.ones(G, G, dtype=torch.bool))
        rows[i, 0] = s
        rows[i, 1:] = torch.arange(b, b + G)
    return inp, pos, mask, rows


def forward(ids, anchors):
    inp, pos, mask, rows = build(ids, anchors)
    out = model(input_ids=inp[None].to(dev), position_ids=pos[None].to(dev), attention_mask=mask[None, None].to(dev))
    return out.logits[0], rows.to(dev)


# ---------------- training
data = []
for f in sorted(glob.glob(args.train)):
    data += jload(f)
random.shuffle(data)
data = data[: args.max_train]
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
nstep = max(1, len(data) // args.accum)
model.train()
t0, log, step = time.time(), [], 0
for i, r in enumerate(data):
    ids = (r["prompt_ids"] + r["output_ids"])[: args.max_len]
    P, N = len(r["prompt_ids"]), len(ids)
    if N - P < K + 2:
        continue
    anc = sorted(random.sample(range(P, N - 1), min(args.anchors, N - 1 - P)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, rows = forward(ids, anc)
    tgt = torch.tensor(anc, device=dev).unsqueeze(1) + torch.arange(1, K + 1, device=dev).unsqueeze(0)
    ok = tgt <= N - 1
    lab = torch.tensor(ids, device=dev)[tgt.clamp(max=N - 1)]
    loss_par = F.cross_entropy(lg[rows][ok].float(), lab[ok])
    # standard next-token loss on all response positions keeps offset-1 quality
    loss_ar = F.cross_entropy(lg[P - 1:N - 1].float(), torch.tensor(ids[P:N], device=dev))
    loss = (loss_par + loss_ar) / args.accum
    loss.backward()
    log.append(float(loss_par))
    if (i + 1) % args.accum == 0:
        lr = args.lr * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / nstep))
        for gp in opt.param_groups:
            gp["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
    if i % 1000 == 0:
        print(f"[train {i}/{len(data)}] par-loss {np.mean(log[-1000:]):.3f} {time.time()-t0:.0f}s", flush=True)
model.save_pretrained(f"{args.out}/model")
tok.save_pretrained(f"{args.out}/model")

# ---------------- evaluation in tf-record format
model.eval()
keep = set(json.load(open(args.clean_ids)))
gen = {}
for f in sorted(glob.glob(args.gen)):
    for r in jload(f):
        gen[r["id"]] = r
recs = []
for f in sorted(glob.glob(args.gold_tf)):
    recs += [r for r in pload(f) if r["id"] in keep]
cnt, sel = {}, []
for r in recs:
    k = (r["task"], r["lang"])
    if cnt.get(k, 0) < args.eval_per_group:
        cnt[k] = cnt.get(k, 0) + 1
        sel.append(r)
out, t0 = [], time.time()
for i, r in enumerate(sel):
    pid = gen[r["id"]]["prompt_ids"]
    ids = pid + r["output_ids"]
    P, N = len(pid), len(ids)
    T, R = N - P, N - 1 - P
    preds = np.zeros((R, K), dtype=np.int64)
    probs = np.zeros((R, K), dtype=np.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for c0 in range(0, R, 48):
            anc = list(range(P + c0, P + min(R, c0 + 48)))
            lg, rows = forward(ids, anc)
            pr = torch.softmax(lg[rows].float(), -1)
            pm, pa = pr.max(-1)
            preds[c0:c0 + len(anc)] = pa.cpu().numpy()
            probs[c0:c0 + len(anc)] = pm.cpu().numpy()
    gold = np.asarray(r["top5_id"])[:, 0]
    devm = np.asarray(r["dev"], dtype=bool)
    a = np.arange(R)[:, None]
    tidx = a + np.arange(1, K + 1)[None, :]
    inb = tidx <= T - 1
    tc = np.clip(tidx, 0, T - 1)
    d = devm[tc] & inb
    prior = np.cumsum(d, 1) - d
    valid = inb & (prior == 0)
    rr = dict(r)
    rr["corr"] = np.where(valid, (preds == gold[tc]).astype(np.uint8), np.uint8(2))
    rr["dprob"] = probs
    out.append(rr)
    if i % 200 == 0:
        print(f"[eval {i}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)
pdump(out, f"{args.out}/tf_0.pkl")
print("done", len(out), flush=True)
