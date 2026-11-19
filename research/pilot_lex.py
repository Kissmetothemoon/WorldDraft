"""Pilot: training-free lexical prior for block drafting (route B).

An in-word token n-gram model per language is counted from target-language text (here: target-generated responses that
the drafter never saw; any monolingual text would do). The context is the current word's tokens so far (reset at every
word start, max 3 tokens, backoff to shorter suffixes). For each block position the drafter's candidates are re-scored:
    score(b) = log p_draft(b) + lam * log p_lex(b | in-word context)      (b continues the word)
    score(b) = log p_draft(b) + lam * log p_lex(<word end> | context)     (b starts a new word / is non-word)
Optionally the lexicon's top-E continuations are added as extra candidates (exact drafter log-probs are gathered).
Teacher-forced evaluation (exact for greedy acceptance), tau@15 per (task, lang), several lam / expansion variants at once.
"""
import argparse
import glob
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common import jload, pdump, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--lex", required=True, help="comma list of globs of jsonl with output_ids + lang (lexicon text)")
ap.add_argument("--eval", required=True)
ap.add_argument("--clean-ids", required=True)
ap.add_argument("--per-group", type=int, default=80)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--lex-max-tokens", type=float, default=0, help="per-language cap on lexicon tokens")
ap.add_argument("--lex-langs", default="", help="only build the lexicon for these languages")
ap.add_argument("--sel-head", default="", help="state dict of a trained selector (pilot_heads.py) to combine with")
ap.add_argument("--sel-type", default="pair", choices=["pair", "pathrnn", "wordrnn"])
ap.add_argument("--lams", default="0.25,0.5,0.75,1.0,1.5")
ap.add_argument("--xword", action="store_true", help="also score word-start candidates with a cross-word bigram")
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
NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
VOC = target.lm_head.out_features
TOPK, LX, E = 16, 32, 4
CONT = np.zeros(VOC, dtype=bool)  # token continues a word (no leading whitespace / punctuation)
for t, bs in enumerate(bt):
    if bs:
        c0 = bs[0]
        CONT[t] = (c0 >= 0x80) or chr(c0).isalnum()
CONT_t = torch.tensor(CONT, device=dev)
END = -1

# ---------------- lexicon counts: (lang, ctx) -> Counter(next or END)
t0 = time.time()
C = defaultdict(Counter)
XW = defaultdict(Counter)
LL = set(args.lex_langs.split(",")) if args.lex_langs else None
used = Counter()
for f in sorted(sum((glob.glob(g) for g in args.lex.split(",")), [])):
    for r in jload(f):
        lang = r.get("lang") or "en"
        if (LL is not None and lang not in LL) or (args.lex_max_tokens and used[lang] >= args.lex_max_tokens):
            continue
        out = r["output_ids"]
        used[lang] += len(out)
        word, prev_t = [], None
        for t in out:
            if CONT[t] and word:
                for n in range(1, 4):
                    if len(word) >= n:
                        C[(lang, tuple(word[-n:]))][t] += 1
                word.append(t)
            else:
                if word:
                    for n in range(1, 4):
                        if len(word) >= n:
                            C[(lang, tuple(word[-n:]))][END] += 1
                if args.xword and prev_t is not None:
                    XW[(lang, prev_t)][t] += 1
                word = [t] if CONT[t] or (bt[t][:1] == b" " and len(bt[t]) > 1) else []
            prev_t = t
print(f"lexicon contexts {len(C)} in {time.time()-t0:.0f}s; tokens per language {dict(used)}", flush=True)


def lex_dist(lang, word):
    """backoff: longest in-word suffix (<=3) with >= 2 observations -> (ids, logprobs, log p_end, log floor)"""
    for n in (3, 2, 1):
        if len(word) >= n:
            c = C.get((lang, tuple(word[-n:])))
            if c is not None:
                tot = sum(c.values())
                if tot >= 2:
                    pe = c.get(END, 0)
                    items = [(k, v) for k, v in c.most_common(LX + 1) if k != END][:LX]
                    floor = math.log(0.5 / (tot + len(c)))
                    return ([k for k, _ in items], [math.log(v / tot) for _, v in items],
                            math.log((pe + 0.5) / (tot + 1)), floor)
    return None


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


@torch.inference_mode()
def blocks(ids, feats, anchors, extra, bsz=48):
    """returns top-16 ids / logprobs and logprobs of the extra ids (A,T,E)"""
    N = ids.shape[1]
    th = draft.hidden_norm(draft.fc(feats))
    cos_c, sin_c = draft.rotary_emb(th, torch.arange(N, device=dev).unsqueeze(0))
    ctx = []
    for layer in draft.layers:
        at = layer.self_attn
        k = at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2)
        v = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
        ctx.append((rope(k, cos_c, sin_c), v))
    Ti, Tl, El, Hs = [], [], [], []
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
        lp = torch.log_softmax(target.lm_head(hn).float(), -1)
        tl, ti = lp.topk(TOPK, -1)
        Ti.append(ti)
        Tl.append(tl)
        El.append(lp.gather(-1, extra[b0:b0 + bsz]))
    return torch.cat(Ti), torch.cat(Tl), torch.cat(El), torch.cat(Hs)


LAMS = [float(x) for x in args.lams.split(",")]
VARIANTS = [("top1", 0.0, False, False), ("lam0.5", 0.5, False, False)] + [(f"lam{l}+exp", l, True, False) for l in LAMS]
if args.xword:
    VARIANTS += [(f"xw lam{l}+exp", l, True, False) for l in LAMS]
SEL = None
if args.sel_head:
    import torch.nn as nn
    HID = cfg.hidden_size
    EMB = target.model.embed_tokens

    class PairSel(nn.Module):
        def __init__(self, r=256):
            super().__init__()
            self.A, self.B = nn.Linear(HID, r, bias=False), nn.Linear(HID, r, bias=False)
            self.H = nn.Sequential(nn.Linear(HID, r), nn.Sigmoid())
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, h, prev, cand, ulp):
            s_ = (self.A(EMB(prev).float()) * self.H(h.float())).unsqueeze(-2) * self.B(EMB(cand).float())
            return self.scale * ulp + s_.sum(-1)

    class PathSel(nn.Module):
        def __init__(self, reset, r=256):
            super().__init__()
            self.reset = reset
            self.inp = nn.Linear(HID, r, bias=False)
            self.gru = nn.GRUCell(r, r)
            self.A, self.B = nn.Linear(r, r, bias=False), nn.Linear(HID, r, bias=False)
            self.H = nn.Sequential(nn.Linear(HID, r), nn.Sigmoid())
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, h, prev, cand, ulp):
            Aa, T = prev.shape
            x = self.inp(EMB(prev).float())
            g = torch.zeros(Aa, x.shape[-1], device=x.device)
            gs = []
            for t in range(T):
                if self.reset:
                    g = g * (1 - (~CONT_t[prev[:, t]]).unsqueeze(1).float())
                g = self.gru(x[:, t], g)
                gs.append(g)
            G = torch.stack(gs, 1)
            s_ = (self.A(G) * self.H(h.float())).unsqueeze(-2) * self.B(EMB(cand).float())
            return self.scale * ulp + s_.sum(-1)

    SEL = (PairSel() if args.sel_type == "pair" else PathSel(args.sel_type == "wordrnn")).to(dev)
    SEL.load_state_dict({k: v for k, v in torch.load(args.sel_head, map_location=dev).items() if not k.startswith(("bias", "wvalid"))})
    SEL.eval()
    VARIANTS += [("sel", 0.0, False, True)] + [(f"sel+lam{l}+exp", l, True, True) for l in LAMS]
keep = set(json.load(open(args.clean_ids)))
ev = defaultdict(list)
for f in sorted(glob.glob(args.eval)):
    for r in jload(f):
        if r["id"] in keep:
            ev[(r["task"], r["lang"])].append(r)
sel = [(r, j % 2) for k in sorted(ev) for j, r in enumerate(ev[k][: args.per_group])][args.shard::args.nshard]
S = defaultdict(lambda: defaultdict(lambda: [0, 0]))


def chain(corr):
    lead = np.cumprod((corr == 1).astype(np.int32), axis=1).sum(1)
    a, accs = 0, []
    while a < corr.shape[0]:
        acc = int(lead[a]) + 1
        accs.append(acc)
        a += acc
    return accs


t0 = time.time()
for i, (r, split) in enumerate(sel):
    lang = r["lang"]
    pidl, oidl = r["prompt_ids"], r["output_ids"]
    ids_l = pidl + oidl
    P, N = len(pidl), len(ids_l)
    if N - P < 8:
        continue
    ids = torch.tensor([ids_l], device=dev)
    with torch.inference_mode():
        out = target(ids, output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1)
        am = out.logits[0, P - 1:N - 1].argmax(-1)
    gold_all = torch.cat([ids[0, :P], am])
    dev_all = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), am != ids[0, P:N]])
    # lexical distribution for predicting absolute position g, from the gold in-word context ending at g-1
    LXI = torch.full((N, LX), -1, dtype=torch.long)
    LXL = torch.full((N, LX), -30.0)
    LEND = torch.zeros(N)
    LFL = torch.full((N,), -30.0)
    HAS = torch.zeros(N, dtype=torch.bool)
    word = []
    for g in range(P, N):
        prev = ids_l[g - 1]
        if CONT[prev] and word:
            word.append(prev)
        else:
            word = [prev] if (CONT[prev] or (bt[prev][:1] == b" " and len(bt[prev]) > 1)) else []
        d = lex_dist(lang, word) if word else None
        if d:
            k = len(d[0])
            LXI[g, :k] = torch.tensor(d[0])
            LXL[g, :k] = torch.tensor(d[1])
            LEND[g], LFL[g], HAS[g] = d[2], d[3], True
    XWI = torch.full((N, LX), -1, dtype=torch.long)
    XWL = torch.full((N, LX), -30.0)
    XWF = torch.full((N,), -30.0)
    if args.xword:
        for g in range(P, N):
            c = XW.get((lang, ids_l[g - 1]))
            if c:
                tot = sum(c.values())
                items = c.most_common(LX)
                XWI[g, :len(items)] = torch.tensor([k for k, _ in items])
                XWL[g, :len(items)] = torch.tensor([math.log(v / tot) for _, v in items])
                XWF[g] = math.log(0.5 / (tot + len(c)))
    LXI, LXL, LEND, LFL, HAS, XWI, XWL, XWF = (x.to(dev) for x in (LXI, LXL, LEND, LFL, HAS, XWI, XWL, XWF))
    anchors = list(range(P, N - 1))
    A = torch.tensor(anchors, device=dev)
    gidx = A.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0)
    inb = gidx <= N - 1
    g = gidx.clamp(max=N - 1)
    prior = torch.cumsum(dev_all[g].int(), 1) - dev_all[g].int()
    valid = inb & (prior == 0)
    gold = gold_all[g]
    extra = LXI[g][..., :E].clamp(min=0)  # (A,T,E)
    ti, tl, el, hn = blocks(ids, feats, anchors, extra)
    cand_all = torch.cat([ti, extra], -1)
    lp_all = torch.cat([tl, torch.where(LXI[g][..., :E] >= 0, el, torch.full_like(el, -1e9))], -1)
    # lexical score of each candidate
    match = cand_all.unsqueeze(-1) == LXI[g].unsqueeze(-2)  # (A,T,K,LX)
    lx = torch.where(match.any(-1), (LXL[g].unsqueeze(-2) * match).sum(-1), LFL[g].unsqueeze(-1).expand_as(cand_all))
    lx = torch.where(CONT_t[cand_all], lx, LEND[g].unsqueeze(-1).expand_as(lx))
    lx = torch.where(HAS[g].unsqueeze(-1), lx, torch.zeros_like(lx))
    if args.xword:
        mx = cand_all.unsqueeze(-1) == XWI[g].unsqueeze(-2)
        xw = torch.where(mx.any(-1), (XWL[g].unsqueeze(-2) * mx).sum(-1), XWF[g].unsqueeze(-1).expand_as(cand_all))
        ws = ~CONT_t[cand_all]
        lx_xw = torch.where(ws, torch.where(HAS[g].unsqueeze(-1), LEND[g].unsqueeze(-1), torch.zeros_like(lx)) + xw, lx)
    if SEL is not None:
        prev = torch.cat([ids[0, A].unsqueeze(1), gold[:, :-1]], 1)
        with torch.inference_mode():
            ssel = torch.cat([SEL(hn[c0:c0 + 64], prev[c0:c0 + 64], cand_all[c0:c0 + 64], lp_all[c0:c0 + 64].clamp(min=-60))
                              for c0 in range(0, cand_all.shape[0], 64)])
    for name, lam, exp, use_sel in VARIANTS:
        sc = (ssel if use_sel else lp_all) + lam * (lx_xw if name.startswith("xw") else lx)
        if use_sel:
            sc = torch.where(lp_all < -1e8, torch.full_like(sc, -1e9), sc)
        if not exp:
            sc = sc.clone()
            sc[..., TOPK:] = -1e9
        pick = cand_all.gather(-1, sc.argmax(-1, keepdim=True)).squeeze(-1)
        corr = torch.where(valid, (pick == gold).to(torch.uint8), torch.full_like(pick, 2, dtype=torch.uint8)).cpu().numpy()
        a = chain(corr)
        S[(r["task"], lang, split)][name][0] += sum(a)
        S[(r["task"], lang, split)][name][1] += len(a)
    if i % 100 == 0:
        print(f"[{i}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)
pdump({f"{k[0]}|{k[1]}|{k[2]}": {n: v for n, v in d.items()} for k, d in S.items()}, args.out + ".pkl")
print("done", flush=True)
