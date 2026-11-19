"""Pilot (route C'): joint in-word chunk prediction on top of a FROZEN DFlash drafter, keeping the token-aligned layout.

Chunk vocabulary: the most frequent in-word token n-grams (length 2-4, never crossing a word boundary), per-language quota,
counted from target-language text. A low-rank head on the drafter's hidden state at block position t predicts which chunk
starts at t (or NONE). Decoding walks the block: if the chunk head is confident (p >= theta) the whole chunk is emitted
(jointly predicted tokens t..t+L-1) and the walk jumps L positions; otherwise the drafter's own top-1 token is used.
Positions stay token-aligned, so the draft is directly comparable to plain DFlash. Teacher-forced exact evaluation.
"""
import argparse
import glob
import json
import math
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import jload, pdump, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--clean-ids", required=True)
ap.add_argument("--quota", type=int, default=2000, help="chunks per language")
ap.add_argument("--max-train", type=int, default=30000)
ap.add_argument("--anchors", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--per-group", type=int, default=80)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
bt = token_byte_table(tok)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
for p in list(target.parameters()) + list(draft.parameters()):
    p.requires_grad_(False)
LIDS, BS, MASK = draft.target_layer_ids, draft.block_size, draft.mask_token_id
cfg = draft.config
HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
NH, NKV, HID = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
VOC = target.lm_head.out_features
CONT = np.zeros(VOC, dtype=bool)
for t, bs in enumerate(bt):
    if bs:
        c0 = bs[0]
        CONT[t] = (c0 >= 0x80) or chr(c0).isalnum()


def words(out):
    """split a token list into word spans [(start, end)) (a word = start token + following continuation tokens)"""
    spans, s = [], None
    for i, t in enumerate(out):
        if CONT[t] and s is not None:
            continue
        if s is not None:
            spans.append((s, i))
        s = i if (CONT[t] or (bt[t][:1] == b" " and len(bt[t]) > 1)) else None
    if s is not None:
        spans.append((s, len(out)))
    return spans


rows = []
for f in sorted(sum((glob.glob(g) for g in args.train.split(",")), [])):
    rows += jload(f)
random.shuffle(rows)
rows = rows[: args.max_train]
# ---------------- chunk vocabulary (per-language quota)
cnt = defaultdict(Counter)
for r in rows:
    out = r["output_ids"]
    lang = r.get("lang") or "en"
    for a, b in words(out):
        for i in range(a, b):
            for L in (2, 3, 4):
                if i + L <= b:
                    cnt[lang][tuple(out[i:i + L])] += 1
vocab = {}
for lang, c in cnt.items():
    for ch, n in c.most_common(args.quota):
        if n >= 3 and ch not in vocab:
            vocab[ch] = len(vocab)
CH = list(vocab)
NONE = len(CH)
print(f"chunk vocab {len(CH)} from {len(rows)} sequences", flush=True)
CH_LEN = torch.tensor([len(c) for c in CH] + [1], device=dev)


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


@torch.inference_mode()
def blocks(ids, feats, anchors, bsz=96):
    N = ids.shape[1]
    th = draft.hidden_norm(draft.fc(feats))
    cos_c, sin_c = draft.rotary_emb(th, torch.arange(N, device=dev).unsqueeze(0))
    ctx = []
    for layer in draft.layers:
        at = layer.self_attn
        k = at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2)
        v = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
        ctx.append((rope(k, cos_c, sin_c), v))
    Hs, T1 = [], []
    for b0 in range(0, len(anchors), bsz):
        A = torch.tensor(anchors[b0:b0 + bsz], device=dev)
        B = A.numel()
        Lc = int(A.max().item())
        blk = torch.full((B, BS), MASK, dtype=torch.long, device=dev)
        blk[:, 0] = ids[0, A]
        h = target.model.embed_tokens(blk)
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
        Hs.append(hn)
        T1.append(target.lm_head(hn).argmax(-1))
    return torch.cat(Hs), torch.cat(T1)


class ChunkHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(HID, 1024), nn.SiLU(), nn.Linear(1024, 1024))
        self.out = nn.Linear(1024, len(CH) + 1)

    def forward(self, h):
        return self.out(self.f(h.float()))


head = ChunkHead().to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=args.lr)


def labels_for(ids_l, P, anchors):
    """chunk label at block position t of anchor s: the longest vocab chunk starting at absolute g = s + t"""
    out = ids_l[P:]
    wend = np.full(len(ids_l), -1)
    for a, b in words(out):
        wend[P + a:P + b] = P + b
    N = len(ids_l)
    lab = np.full((len(anchors), BS - 1), NONE)
    for i, s in enumerate(anchors):
        for t in range(1, BS):
            g = s + t
            if g >= N or wend[g] < 0:
                continue
            for L in (4, 3, 2):
                if g + L <= wend[g] and t + L - 1 <= BS - 1:
                    c = vocab.get(tuple(ids_l[g:g + L]))
                    if c is not None:
                        lab[i, t - 1] = c
                        break
    return torch.tensor(lab, device=dev)


t0, log = time.time(), []
for i, r in enumerate(rows):
    ids_l = (r["prompt_ids"] + r["output_ids"])[:2048]
    P, N = len(r["prompt_ids"]), len(ids_l)
    if N - P < 8:
        continue
    anc = sorted(random.sample(range(P, N - 1), min(args.anchors, N - 1 - P)))
    ids = torch.tensor([ids_l], device=dev)
    with torch.inference_mode():
        out = target(ids, output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1)
    hn, _ = blocks(ids, feats, anc)
    lab = labels_for(ids_l, P, anc)
    lg = head(hn.clone())
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for gp in opt.param_groups:
        gp["lr"] = args.lr * min(1.0, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * i / len(rows)))
    opt.step()
    log.append(float(loss))
    if i % 1000 == 0:
        print(f"[train {i}/{len(rows)}] loss {np.mean(log[-1000:]):.3f} {time.time()-t0:.0f}s", flush=True)

# ---------------- evaluation
head.eval()
THETAS = [0.3, 0.5, 0.7, 0.9]
keep = set(json.load(open(args.clean_ids)))
ev = defaultdict(list)
for f in sorted(glob.glob(args.eval)):
    for r in jload(f):
        if r["id"] in keep:
            ev[(r["task"], r["lang"])].append(r)
sel = [r for k in sorted(ev) for r in ev[k][: args.per_group]]
S = defaultdict(lambda: defaultdict(lambda: [0, 0]))
USE = defaultdict(lambda: [0, 0])  # lang -> [chunks emitted, rounds] at theta 0.5
for i, r in enumerate(sel):
    ids_l = r["prompt_ids"] + r["output_ids"]
    P, N = len(r["prompt_ids"]), len(ids_l)
    if N - P < 8:
        continue
    ids = torch.tensor([ids_l], device=dev)
    with torch.inference_mode():
        out = target(ids, output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1)
        am = out.logits[0, P - 1:N - 1].argmax(-1).cpu().numpy()
        anchors = list(range(P, N - 1))
        hn, t1 = blocks(ids, feats, anchors)
        pc = torch.softmax(torch.cat([head(hn[c0:c0 + 128]) for c0 in range(0, hn.shape[0], 128)]), -1)
        cp, ci = pc.max(-1)
    cp, ci, t1 = cp.cpu().numpy(), ci.cpu().numpy(), t1.cpu().numpy()
    gold = np.concatenate([np.array(ids_l[:P]), am])
    devm = np.concatenate([np.zeros(P, bool), am != np.array(ids_l[P:])])
    for th in [None] + THETAS:
        leads = []
        nch = 0
        for a, s in enumerate(anchors):
            draft_toks, t = [], 1
            while t <= BS - 1:
                c = ci[a, t - 1]
                if th is not None and c != NONE and cp[a, t - 1] >= th and t + len(CH[c]) - 1 <= BS - 1:
                    draft_toks += list(CH[c])
                    t += len(CH[c])
                    nch += 1
                else:
                    draft_toks.append(int(t1[a, t - 1]))
                    t += 1
            n = 0
            for k, tk in enumerate(draft_toks, 1):
                g = s + k
                if g >= N or devm[s + 1:g].any() or tk != gold[g]:
                    break
                n += 1
            leads.append(n)
        a, accs = 0, []
        while a < len(leads):
            acc = leads[a] + 1
            accs.append(acc)
            a += acc
        name = "top1" if th is None else f"chunk>={th}"
        S[(r["task"], r["lang"])][name][0] += sum(accs)
        S[(r["task"], r["lang"])][name][1] += len(accs)
    if i % 100 == 0:
        print(f"[eval {i}/{len(sel)}]", flush=True)
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
names = ["top1"] + [f"chunk>={t}" for t in THETAS]
L = [f"# chunk head on {args.draft.rstrip('/').split('/')[-1]} (vocab {len(CH)}, quota {args.quota})\n",
     "| lang | " + " | ".join(names) + " | best gain |", "|---|" + "---|" * (len(names) + 1)]
for l in LANGS:
    ks = [k for k in S if k[1] == l]
    if not ks:
        continue
    t = {n: np.mean([S[k][n][0] / S[k][n][1] for k in ks]) for n in names}
    best = max(names[1:], key=t.get)
    L.append(f"| {l} | " + " | ".join(f"{t[n]:.3f}" for n in names) + f" | {t[best]/t['top1']-1:+.1%} ({best}) |")
txt = "\n".join(L) + "\n"
open(args.out + ".md", "w", encoding="utf-8").write(txt)
pdump({f"{k[0]}|{k[1]}": dict(v) for k, v in S.items()}, args.out + ".pkl")
print(txt)
