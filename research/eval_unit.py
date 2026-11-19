"""Offline greedy evaluation of token-layout vs unit-layout DFlash drafters on the evaluation trajectories (same gold /
deviation rule as tf_dflash.analyse). Writes <out>.md / <out>.pkl.

Per (task, lang): tau (accepted length incl. bonus, greedy chain), mean drafted length, and the share of rejections whose
rejected token is an invalid UTF-8 continuation of the accepted prefix.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import jload, lang_ok, pdump, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--clean-ids", required=True)
ap.add_argument("--per-group", type=int, default=80)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=int, default=0, help="truncate every draft to this many tokens")
args = ap.parse_args()

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
bt = token_byte_table(tok)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
LIDS, BS, MASK = draft.target_layer_ids, draft.block_size, draft.mask_token_id
cfg = draft.config
HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
NH, NKV, HID = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
VOC = target.lm_head.out_features
EOU_ID = VOC
ex_path = f"{args.draft}/unit_extras.pt"
UNIT = os.path.exists(ex_path)
if UNIT:
    ck = torch.load(ex_path, map_location=dev)
    LAYOUT = ck["layout"]
    LAYOUT_ID = {tuple(int(x) for x in k.split("x")): v for k, v in ck["layout_id"].items()}
    pos_emb = nn.Embedding(len(LAYOUT_ID) * BS, HID).to(dev)
    eou = nn.Linear(HID, 1).to(dev)
    pos_emb.weight.data.copy_(ck["state"]["pos.weight"])
    eou.weight.data.copy_(ck["state"]["eou.weight"])
    eou.bias.data.copy_(ck["state"]["eou.bias"])


def _step(bs, r):
    for x in bs:
        if r > 0:
            if x & 0xC0 != 0x80:
                return -1
            r -= 1
        elif x < 0x80:
            pass
        elif x & 0xE0 == 0xC0:
            r = 1
        elif x & 0xF0 == 0xE0:
            r = 2
        elif x & 0xF8 == 0xF0:
            r = 3
        else:
            return -1
    return r


NEWR = [[-1] * (VOC + 1) for _ in range(4)]
for t, bs in enumerate(bt):
    for r in range(4):
        NEWR[r][t] = _step(bs, r) if bs else (0 if r == 0 else -1)


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


@torch.inference_mode()
def run(ids, P, lang, bsz=64):
    N = ids.shape[1]
    out = target(ids, output_hidden_states=True)
    feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1)
    am = out.logits[0, P - 1:N - 1].argmax(-1)
    th = draft.hidden_norm(draft.fc(feats))
    cos_c, sin_c = draft.rotary_emb(th, torch.arange(N, device=dev).unsqueeze(0))
    ctx = []
    for layer in draft.layers:
        at = layer.self_attn
        k = at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2)
        v = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
        ctx.append((rope(k, cos_c, sin_c), v))
    anchors = list(range(P, N - 1))
    preds = []
    for b0 in range(0, len(anchors), bsz):
        A = torch.tensor(anchors[b0:b0 + bsz], device=dev)
        B = A.numel()
        Lc = int(A.max().item())
        blk = torch.full((B, BS), MASK, dtype=torch.long, device=dev)
        blk[:, 0] = ids[0, A]
        h = target.model.embed_tokens(blk)
        if UNIT:
            lid = LAYOUT_ID[tuple(LAYOUT.get(lang, (5, 3)))]
            h = h + pos_emb(lid * BS + torch.arange(BS, device=dev)).to(h.dtype).unsqueeze(0)
        pos_n = A.unsqueeze(1) + torch.arange(BS, device=dev).unsqueeze(0)
        cos_n, sin_n = draft.rotary_emb(h, pos_n)
        allow = torch.arange(Lc, device=dev).unsqueeze(0) < A.unsqueeze(1)
        mask = torch.cat([allow, torch.ones(B, BS, dtype=torch.bool, device=dev)], 1)[:, None, None, :]
        for li, layer in enumerate(draft.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(B, BS, NH, HD)).transpose(1, 2)
            kn = at.k_norm(at.k_proj(x).view(B, BS, NKV, HD)).transpose(1, 2)
            vn = at.v_proj(x).view(B, BS, NKV, HD).transpose(1, 2)
            q, kn = rope(q, cos_n, sin_n), rope(kn, cos_n, sin_n)
            kc, vc = ctx[li]
            K_ = torch.cat([kc[:, :, :Lc].expand(B, -1, -1, -1), kn], 2)
            V_ = torch.cat([vc[:, :, :Lc].expand(B, -1, -1, -1), vn], 2)
            o = F.scaled_dot_product_attention(q, K_, V_, attn_mask=mask, scale=at.scaling, enable_gqa=True)
            h = h + at.o_proj(o.transpose(1, 2).reshape(B, BS, NH * HD))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hn = draft.norm(h)[:, 1:, :]
        lg = target.lm_head(hn).float()
        if UNIT:
            lg = torch.cat([lg, eou(hn.float())], -1)
        preds.append(lg.argmax(-1))
    return torch.cat(preds).cpu().numpy(), am


def flatten(row, lang):
    out = _flatten(row, lang)
    return out[: args.cap] if args.cap else out


def _flatten(row, lang):
    if not UNIT:
        return list(row)
    K, M = LAYOUT.get(lang, (5, 3))
    out = []
    for j in range(K):
        filled = True
        for m in range(M):
            t = int(row[j * M + m])
            if t == EOU_ID:
                filled = False
                break
            out.append(t)
        if not filled and m == 0 and j > 0:  # an empty later slot ends the draft; a full slot is assumed complete
            break
    return out[: BS - 1]


keep = set(json.load(open(args.clean_ids)))
ev = defaultdict(list)
for f in sorted(glob.glob(args.eval)):
    for r in jload(f):
        if r["id"] in keep:
            ev[(r["task"], r["lang"])].append(r)
sel = [r for k in sorted(ev) for r in ev[k][: args.per_group]]
sel = sel[args.shard::args.nshard]
S = defaultdict(lambda: defaultdict(float))
for i, r in enumerate(sel):
    pidl, oidl = r["prompt_ids"], r["output_ids"]
    ids_l = pidl + oidl
    P, N = len(pidl), len(ids_l)
    if N - P < 8:
        continue
    ids = torch.tensor([ids_l], device=dev)
    pred, am = run(ids, P, r["lang"])
    am = am.cpu().numpy()
    gold = np.concatenate([np.array(ids_l[:P]), am])
    devm = np.concatenate([np.zeros(P, bool), am != np.array(ids_l[P:])])
    rs, st = [0] * N, 0  # UTF-8 state after each position of the trajectory
    for p_ in range(P, N):
        st = max(NEWR[st][ids_l[p_]], 0)
        rs[p_] = st
    leads, flen, bad = [], [], 0
    R = N - 1 - P
    for a in range(R):
        s = P + a
        fl = flatten(pred[a], r["lang"])
        n = 0
        for k, t in enumerate(fl, 1):
            g = s + k
            if g >= N or devm[s + 1:g].any() or t != gold[g]:
                if g < N and not devm[s + 1:g].any() and NEWR[rs[g - 1]][t] < 0:
                    bad += 1
                break
            n += 1
        leads.append(n)
        flen.append(len(fl))
    a, accs = 0, []
    while a < R:
        acc = leads[a] + 1
        accs.append(acc)
        a += acc
    key = (r["task"], r["lang"])
    S[key]["acc"] += sum(accs)
    S[key]["rounds"] += len(accs)
    S[key]["flen"] += float(np.mean(flen))
    S[key]["n"] += 1
    S[key]["rej"] += sum(1 for x, f in zip(leads, flen) if x < f)
    S[key]["bad"] += bad
    if i % 100 == 0:
        print(f"[{i}/{len(sel)}]", flush=True)
pdump({k: dict(v) for k, v in S.items()}, args.out + ".pkl")
print("done", len(sel), flush=True)
