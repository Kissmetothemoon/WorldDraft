"""Pilot for method routes M2 (coherence selector over block candidates) and M1's core mechanism (sequential in-block
conditioning on the previous token), trained on top of a FROZEN DFlash drafter; offline greedy evaluation per language.

--head pair     DFlash-2 style path selector: S_t(a,b) = U_t(b) + <A(a) * H(h_t), B(b)>, b in the drafter's top-16
--head pairbyte same + UTF-8 byte-state features: bias[state after a, leading continuation bytes of b] + w * valid(a,b)
--head seq      DSpark-style sequential head over the full vocab: logits = lm_head(norm(h_t + MLP([h_t; E(prev)])))
Training uses teacher forcing (prev = gold), which is exact for greedy acceptance: a position only matters if every
earlier drafted token was accepted, i.e. equal to gold.
Evaluation (same gold / validity as tf_dflash.analyse) reports tau@15 per (task, lang) for:
  top1 | utf8 (first valid UTF-8 candidate) | head | head+utf8 | (seq head) head only where the drafter's top-1 continues a word
"""
import argparse
import glob
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import jload, lang_ok, pdump, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--head", choices=["pair", "pairbyte", "seq", "wordrnn", "pathrnn"], required=True)
ap.add_argument("--train", required=True, help="glob of target-generated jsonl (prompt_ids, output_ids)")
ap.add_argument("--eval", required=True, help="glob of evaluation trajectories (gen_*.jsonl)")
ap.add_argument("--clean-ids", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-train", type=int, default=30000)
ap.add_argument("--anchors", type=int, default=64)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval-per-group", type=int, default=40)
ap.add_argument("--rank", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--load-head", default="")
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
for p in list(target.parameters()) + list(draft.parameters()):
    p.requires_grad_(False)
LIDS, BS, MASK = draft.target_layer_ids, draft.block_size, draft.mask_token_id
cfg = draft.config
HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
HID = cfg.hidden_size
VOC = target.lm_head.out_features
TOPK = 16
EMB = target.model.embed_tokens

# ---------------- UTF-8 tables: NEWR[r, t] = pending continuation bytes after token t given r pending before (-1 invalid)
bt = token_byte_table(tok)


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


NEWR_l = [[-1] * VOC for _ in range(4)]
LEAD_l = [0] * VOC
for t, bs in enumerate(bt):
    for r in range(4):
        NEWR_l[r][t] = _step(bs, r) if bs else (0 if r == 0 else -1)
    n = 0
    for x in bs:
        if x & 0xC0 == 0x80:
            n += 1
        else:
            break
    LEAD_l[t] = min(n, 3)
NEWR_c = torch.tensor(NEWR_l, dtype=torch.long)
NEWR = NEWR_c.to(dev)
LEAD = torch.tensor(LEAD_l, dtype=torch.long, device=dev)
# "continues a word": token bytes do not start with whitespace / punctuation (used by the in-word-only seq variant)
CONT = torch.zeros(VOC, dtype=torch.bool)
for t, bs in enumerate(bt):
    if bs:
        c0 = bs[0]
        CONT[t] = (c0 >= 0x80) or (chr(c0).isalnum())
CONT = CONT.to(dev)


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


@torch.inference_mode()
def target_pass(ids):
    out = target(ids, output_hidden_states=True)
    return out.logits[0], torch.cat([out.hidden_states[l + 1] for l in LIDS], dim=-1)


@torch.inference_mode()
def blocks(ids, feats, anchors, bsz=96):
    """frozen DFlash block pass: returns normed hidden (A,BS-1,H), top-16 ids, top-16 logprobs, top-1 prob."""
    N = ids.shape[1]
    th = draft.hidden_norm(draft.fc(feats))
    pos_ctx = torch.arange(N, device=dev).unsqueeze(0)
    cos_c, sin_c = draft.rotary_emb(th, pos_ctx)
    ctx = []
    for layer in draft.layers:
        at = layer.self_attn
        k = at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2)
        v = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
        ctx.append((rope(k, cos_c, sin_c), v))
    Hs, Ti, Tl = [], [], []
    for b0 in range(0, len(anchors), bsz):
        A = torch.tensor(anchors[b0:b0 + bsz], device=dev)
        B = A.numel()
        Lc = int(A.max().item())
        blk = torch.full((B, BS), MASK, dtype=torch.long, device=dev)
        blk[:, 0] = ids[0, A]
        h = EMB(blk)
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
        lp = torch.log_softmax(target.lm_head(hn).float(), -1)
        tl, ti = lp.topk(TOPK, -1)
        Hs.append(hn)
        Ti.append(ti)
        Tl.append(tl)
    return torch.cat(Hs), torch.cat(Ti), torch.cat(Tl)


class PairSel(nn.Module):
    def __init__(self, byte):
        super().__init__()
        r = args.rank
        self.A = nn.Linear(HID, r, bias=False)
        self.B = nn.Linear(HID, r, bias=False)
        self.H = nn.Sequential(nn.Linear(HID, r), nn.Sigmoid())
        self.byte = byte
        if byte:
            self.bias = nn.Parameter(torch.zeros(5, 4))  # state after a in {0,1,2,3,invalid} x leading cont bytes of b
            self.wvalid = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, h, prev, cand, ulp, rprev):
        """h (A,T,H); prev (A,T) previous token; cand (A,T,16); ulp (A,T,16) drafter logprobs; rprev (A,T) UTF-8 state
        before prev. Returns scores (A,T,16)."""
        ea = EMB(prev).float()
        eb = EMB(cand).float()
        s = (self.A(ea) * self.H(h.float())).unsqueeze(-2) * self.B(eb)
        S = self.scale * ulp + s.sum(-1)
        if self.byte:
            sa = NEWR[rprev.clamp(min=0), prev]  # state after a (-1 invalid)
            sa = torch.where(sa < 0, torch.full_like(sa, 4), sa)
            nb = NEWR[sa.clamp(max=3).unsqueeze(-1).expand_as(cand), cand]
            valid = ((nb >= 0) & (sa.unsqueeze(-1) < 4)).float()
            S = S + self.bias[sa.unsqueeze(-1).expand_as(cand), LEAD[cand]] + self.wvalid * valid
        return S


class PathSel(nn.Module):
    """Higher-order selector: a GRU summarises the chosen path; with reset=True it restarts at every word start, so the
    score of candidate b conditions on the whole in-word prefix (not only the previous token)."""

    def __init__(self, reset):
        super().__init__()
        r = args.rank
        self.reset = reset
        self.inp = nn.Linear(HID, r, bias=False)
        self.gru = nn.GRUCell(r, r)
        self.A = nn.Linear(r, r, bias=False)
        self.B = nn.Linear(HID, r, bias=False)
        self.H = nn.Sequential(nn.Linear(HID, r), nn.Sigmoid())
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, h, prev, cand, ulp, rprev):
        Aa, T = prev.shape
        x = self.inp(EMB(prev).float())  # (A,T,r)
        g = torch.zeros(Aa, x.shape[-1], device=x.device)
        gs = []
        for t in range(T):
            if self.reset:
                ws = (~CONT[prev[:, t]]).unsqueeze(1).float()  # previous token starts a new word / is non-word
                g = g * (1 - ws)
            g = self.gru(x[:, t], g)
            gs.append(g)
        G = torch.stack(gs, 1)
        eb = EMB(cand).float()
        s = (self.A(G) * self.H(h.float())).unsqueeze(-2) * self.B(eb)
        return self.scale * ulp + s.sum(-1)


class SeqHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * HID, 1024), nn.SiLU(), nn.Linear(1024, HID))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, h, prev):
        x = torch.cat([h.float(), EMB(prev).float()], -1)
        return h.float() + self.mlp(x)


def logits_from_hidden(hh):
    return target.lm_head(draft.norm(hh.to(torch.bfloat16))).float()


head = (SeqHead() if args.head == "seq" else PathSel(args.head == "wordrnn") if args.head in ("wordrnn", "pathrnn")
        else PairSel(args.head == "pairbyte")).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)


def utf8_state(seq_ids):
    """UTF-8 pending-byte state after every position of the full sequence (prompt positions = 0)."""
    rs, r = [0] * len(seq_ids), 0
    for i, t in enumerate(seq_ids):
        r = max(int(NEWR_c[r, t]), 0)
        rs[i] = r
    return torch.tensor(rs, device=dev)


def prepare(r, anchors=None):
    pidl, oidl = r["prompt_ids"], r["output_ids"]
    ids_l = (pidl + oidl)[: args.max_len]
    P, N = len(pidl), len(ids_l)
    if N - P < 8:
        return None
    ids = torch.tensor([ids_l], device=dev)
    logits, feats = target_pass(ids)
    am = logits[P - 1:N - 1].argmax(-1)
    gold_all = torch.cat([ids[0, :P], am])
    dev_all = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), am != ids[0, P:N]])
    if anchors is None:
        anchors = list(range(P, N - 1))
    A = torch.tensor(anchors, device=dev)
    g = (A.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0))
    inb = g <= N - 1
    g = g.clamp(max=N - 1)
    prior = torch.cumsum(dev_all[g].int(), 1) - dev_all[g].int()
    valid = inb & (prior == 0)
    gold = gold_all[g]
    prev = torch.cat([ids[0, A].unsqueeze(1), gold[:, :-1]], 1)  # teacher forcing: previous = gold
    rs = utf8_state(ids_l)
    rprev = torch.cat([rs[(A - 1).clamp(min=0)].unsqueeze(1), rs[(g[:, :-1] - 1).clamp(min=0)]], 1)  # state before prev
    hn, ti, tl = blocks(ids, feats, anchors)
    return dict(gold=gold, valid=valid, prev=prev, rprev=rprev, h=hn, ti=ti, tl=tl, P=P, task=r.get("task"), lang=r.get("lang"))


def head_scores(d):
    if args.head == "seq":
        return None
    return head(d["h"], d["prev"], d["ti"], d["tl"], d["rprev"])


# ---------------- training
rows = []
for f in sorted(sum((glob.glob(g) for g in args.train.split(",")), [])):
    rows += jload(f)
random.shuffle(rows)
rows = rows[: args.max_train]
if args.load_head:
    sd = torch.load(args.load_head, map_location=dev)
    missing = head.load_state_dict(sd, strict=False)
    print("loaded head", args.load_head, missing, flush=True)
    rows = []
log = []
t0 = time.time()
head.train()
for i, r in enumerate(rows):
    N = min(len(r["prompt_ids"]) + len(r["output_ids"]), args.max_len)
    P = len(r["prompt_ids"])
    if N - P < 8:
        continue
    anc = sorted(random.sample(range(P, N - 1), min(args.anchors, N - 1 - P)))
    d = prepare(r, anc)
    if d is None:
        continue
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        if args.head == "seq":
            hh = head(d["h"].clone(), d["prev"])
            lg = logits_from_hidden(hh)
            m = d["valid"]
            loss = F.cross_entropy(lg[m], d["gold"][m])
        else:
            S = head(d["h"].clone(), d["prev"], d["ti"], d["tl"].clone(), d["rprev"])
            hit = d["ti"] == d["gold"].unsqueeze(-1)
            m = d["valid"] & hit.any(-1)
            loss = F.cross_entropy(S[m], hit[m].float().argmax(-1)) if m.any() else S.sum() * 0
    opt.zero_grad(set_to_none=True)
    loss.backward()
    lr = args.lr * min(1.0, (i + 1) / 500) * 0.5 * (1 + math.cos(math.pi * i / len(rows)))
    for gp in opt.param_groups:
        gp["lr"] = lr
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    log.append(float(loss))
    if i % 500 == 0:
        print(f"[train {i}/{len(rows)}] loss {np.mean(log[-500:]):.4f} {time.time()-t0:.0f}s", flush=True)
torch.save(head.state_dict(), args.out + ".pt")

# ---------------- evaluation
head.eval()
keep = set(json.load(open(args.clean_ids)))
ev = defaultdict(list)
for f in sorted(glob.glob(args.eval)):
    for r in jload(f):
        if r["id"] in keep:
            ev[(r["task"], r["lang"])].append(r)
sel = [r for k in sorted(ev) for r in ev[k][: args.eval_per_group]]


def chain(corr):
    lead = np.cumprod((corr == 1).astype(np.int32), axis=1).sum(1)
    a, accs = 0, []
    while a < corr.shape[0]:
        acc = int(lead[a]) + 1
        accs.append(acc)
        a += acc
    return accs


S = defaultdict(lambda: defaultdict(lambda: [0, 0]))
t0 = time.time()
for i, r in enumerate(sel):
    with torch.inference_mode():
        d = prepare(r)
        if d is None:
            continue
        gold, valid, ti = d["gold"], d["valid"], d["ti"]
        V = {}

        def cm(pred):
            return torch.where(valid, (pred == gold).to(torch.uint8), torch.full_like(pred, 2, dtype=torch.uint8)).cpu().numpy()

        top1 = ti[..., 0]
        V["top1"] = cm(top1)
        sa = NEWR[d["rprev"].clamp(min=0), d["prev"]]  # UTF-8 state after the (gold) previous token
        ok = NEWR[sa.clamp(min=0).unsqueeze(-1).expand_as(ti), ti] >= 0
        ok = ok & (sa.unsqueeze(-1) >= 0)
        first = torch.where(ok.any(-1), ok.float().argmax(-1), torch.zeros_like(top1))
        V["utf8"] = cm(ti.gather(-1, first.unsqueeze(-1)).squeeze(-1))
        if args.head == "seq":
            hps, htis = [], []
            for c0 in range(0, d["h"].shape[0], 64):
                lg = logits_from_hidden(head(d["h"][c0:c0 + 64], d["prev"][c0:c0 + 64]))
                hps.append(lg.argmax(-1))
                htis.append(lg.topk(TOPK, -1).indices)
                del lg
            hp, hti = torch.cat(hps), torch.cat(htis)
            V["head"] = cm(hp)
            # head+utf8: best valid token under the head's distribution among its top-16
            okh = (NEWR[sa.clamp(min=0).unsqueeze(-1).expand_as(hti), hti] >= 0) & (sa.unsqueeze(-1) >= 0)
            fh = torch.where(okh.any(-1), okh.float().argmax(-1), torch.zeros_like(hp))
            V["head+utf8"] = cm(hti.gather(-1, fh.unsqueeze(-1)).squeeze(-1))
            V["head in-word only"] = cm(torch.where(CONT[top1], hp, top1))
        else:
            Sc = torch.cat([head(d["h"][c0:c0 + 64], d["prev"][c0:c0 + 64], ti[c0:c0 + 64], d["tl"][c0:c0 + 64], d["rprev"][c0:c0 + 64])
                            for c0 in range(0, ti.shape[0], 64)])
            V["head"] = cm(ti.gather(-1, Sc.argmax(-1, keepdim=True)).squeeze(-1))
            Sm = Sc.masked_fill(~ok, -1e9)
            V["head+utf8"] = cm(ti.gather(-1, Sm.argmax(-1, keepdim=True)).squeeze(-1))
        key = (r["task"], r["lang"])
        for name, c in V.items():
            a = chain(c)
            S[key][name][0] += sum(a)
            S[key][name][1] += len(a)
    if i % 100 == 0:
        print(f"[eval {i}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
names = list(next(iter(S.values())).keys())
lines = [f"# pilot {args.head} on {args.draft.rstrip('/').split('/')[-1]} (train {args.train.split('/')[-1]}, {len(rows)} seqs)\n",
         "tau@15 per language (mean over tasks); gain = head+utf8 / top1 - 1\n",
         "| lang | " + " | ".join(names) + " | gain head | gain head+utf8 |", "|---|" + "---|" * (len(names) + 2)]
tab = {}
for l in LANGS:
    ks = [k for k in S if k[1] == l]
    if not ks:
        continue
    t = {n: np.mean([S[k][n][0] / S[k][n][1] for k in ks]) for n in names}
    tab[l] = t
    lines.append(f"| {l} | " + " | ".join(f"{t[n]:.3f}" for n in names) + f" | {t['head']/t['top1']-1:+.1%} | {t['head+utf8']/t['top1']-1:+.1%} |")
txt = "\n".join(lines) + "\n"
open(args.out + ".md", "w", encoding="utf-8").write(txt)
pdump({"tab": tab, "raw": {f"{k[0]}|{k[1]}": dict(v) for k, v in S.items()}, "loss": log}, args.out + ".pkl")
print(txt)
