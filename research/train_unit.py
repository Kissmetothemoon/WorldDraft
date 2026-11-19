"""Controlled pilot for method M1: unit-aligned block layout for a DFlash drafter (vs the standard token layout).

--layout token : standard DFlash block, position p (1..15) = p-th token after the anchor (control arm)
--layout unit  : the 15 block positions are K units x M sub-positions (K, M chosen per language from tokens/word);
                 slot 0 = remainder of the anchor's unit, slot j = j-th following unit; sub-position m = m-th token of the
                 unit, followed by an end-of-unit (EOU) label; learned (layout, position) embeddings on the block input and
                 an EOU logit next to the target lm_head. Units = words (common._segment; jieba/fugashi/pythainlp for
                 zh/ja/th) or a single whitespace/punctuation token.
Everything else (init, data, target features, attention, optimiser, steps) is identical between the two arms.
Inference (unit): expand slot by slot (tokens until EOU, stop after a slot that fills all M sub-positions without EOU),
cap 15 tokens, verify as usual (lossless).

torchrun --nproc_per_node 4 train_unit.py --layout unit --target T --init ftml --data 'glob' --out DIR
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from common import classify_tokens, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--layout", choices=["token", "unit"], required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--lr-new", type=float, default=1e-3, help="lr of the new layout embeddings / EOU head")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--anchors", type=int, default=32)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--decay", type=float, default=7.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-seqs", type=int, default=0)
ap.add_argument("--tau-beta", type=float, default=0.0, help="weight of the expected-accepted-length surrogate (token layout)")
ap.add_argument("--joint-sel", choices=["none", "pair", "wordrnn"], default="none", help="train a path selector jointly with the drafter")
ap.add_argument("--sel-weight", type=float, default=1.0)
ap.add_argument("--lang-weights", default="", help="lang:w,... resample the training set (same size) by language weight")
args = ap.parse_args()

# (K units, M sub-positions) per language, from the target's tokens/word (pilot_cpu P1a); K*M <= 15
LAYOUT = {**{l: (7, 2) for l in ("en", "fr", "es", "zh", "ja", "hi-Latn")},
          **{l: (5, 3) for l in ("de", "ru", "ko", "ar", "th", "sw", "sr-Cyrl", "sr-Latn")},
          **{l: (3, 5) for l in ("hi", "am")}, **{l: (2, 7) for l in ("bn", "te")}}
LAYOUT_ID = {km: i for i, km in enumerate(sorted(set(LAYOUT.values())))}

ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = torch.device("cuda", torch.cuda.current_device())
random.seed(args.seed + rank)
torch.manual_seed(args.seed)

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half

tok = AutoTokenizer.from_pretrained(args.target)
bt = token_byte_table(tok)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map={"": dev},
                                              attn_implementation="sdpa").eval()
for p in target.parameters():
    p.requires_grad_(False)
draft = AutoModel.from_pretrained(args.init, trust_remote_code=True, torch_dtype=torch.float32, device_map={"": dev})
draft.train()
LIDS, BS, MASK = draft.target_layer_ids, draft.block_size, draft.mask_token_id
cfg = draft.config
HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
HID = cfg.hidden_size


class UnitExtras(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Embedding(len(LAYOUT_ID) * BS, HID)
        nn.init.zeros_(self.pos.weight)
        self.eou = nn.Linear(HID, 1)
        nn.init.zeros_(self.eou.weight)
        nn.init.constant_(self.eou.bias, -4.0)


extras = UnitExtras().to(dev) if args.layout == "unit" else None
EMB = target.model.embed_tokens
CONT = torch.zeros(target.lm_head.out_features, dtype=torch.bool)
for _t, _bs in enumerate(bt):
    if _bs:
        CONT[_t] = (_bs[0] >= 0x80) or chr(_bs[0]).isalnum()
CONT = CONT.to(dev)


class PairSel(nn.Module):
    def __init__(self, r=256):
        super().__init__()
        self.A = nn.Linear(HID, r, bias=False)
        self.B = nn.Linear(HID, r, bias=False)
        self.H = nn.Sequential(nn.Linear(HID, r), nn.Sigmoid())
        self.byte = False
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, h, prev, cand, ulp):
        s = (self.A(EMB(prev).float()) * self.H(h.float())).unsqueeze(-2) * self.B(EMB(cand).float())
        return self.scale * ulp + s.sum(-1)


class PathSel(nn.Module):
    def __init__(self, reset=True, r=256):
        super().__init__()
        self.reset = reset
        self.inp = nn.Linear(HID, r, bias=False)
        self.gru = nn.GRUCell(r, r)
        self.A = nn.Linear(r, r, bias=False)
        self.B = nn.Linear(HID, r, bias=False)
        self.H = nn.Sequential(nn.Linear(HID, r), nn.Sigmoid())
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, h, prev, cand, ulp):
        Aa, T = prev.shape
        x = self.inp(EMB(prev).float())
        g = torch.zeros(Aa, x.shape[-1], device=x.device)
        gs = []
        for t in range(T):
            if self.reset:
                g = g * (1 - (~CONT[prev[:, t]]).unsqueeze(1).float())
            g = self.gru(x[:, t], g)
            gs.append(g)
        G = torch.stack(gs, 1)
        s = (self.A(G) * self.H(h.float())).unsqueeze(-2) * self.B(EMB(cand).float())
        return self.scale * ulp + s.sum(-1)


sel = {"none": None, "pair": PairSel, "wordrnn": PathSel}[args.joint_sel]
sel = sel().to(dev) if sel is not None else None


def units_of(out_ids, lang):
    """unit id per response token: word-start index for word tokens, own index for whitespace/punct tokens."""
    c = classify_tokens(out_ids, bt, lang)
    return [w if (cl != 0 and w >= 0) else k for k, (cl, w) in enumerate(zip(c["cls"], c["wstart"]))]


# ------------------------------------------------------------------ data
rows = []
for f in sorted(sum((glob.glob(g) for g in args.data.split(",")), [])):
    with open(f, encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            if len(r["output_ids"]) < 8:
                continue
            rows.append((r["prompt_ids"], r["output_ids"], r.get("lang") or "en"))
random.Random(args.seed).shuffle(rows)
if args.lang_weights:
    wl = {k: float(v) for k, v in (x.split(":") for x in args.lang_weights.split(","))}
    rw = [wl.get(r[2], 1.0) for r in rows]
    rows = random.Random(args.seed + 1).choices(rows, weights=rw, k=len(rows))
if args.max_seqs:
    rows = rows[: args.max_seqs]
mine = rows[rank::world]
n_steps = int(len(mine) * args.epochs) // args.accum
if ddp:
    t_steps = torch.tensor([n_steps], device=dev)
    dist.all_reduce(t_steps, op=dist.ReduceOp.MIN)
    n_steps = int(t_steps.item())
if rank == 0:
    print(f"{len(rows)} sequences, {world} GPUs, {n_steps} optimizer steps, layout={args.layout}", flush=True)

params = [p for p in draft.parameters() if p.requires_grad]
groups = [{"params": params, "lr": args.lr}]
if extras is not None:
    groups.append({"params": list(extras.parameters()), "lr": args.lr_new})
if sel is not None:
    groups.append({"params": list(sel.parameters()), "lr": args.lr_new})
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(1, min(100, n_steps // 20))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, n_steps))))))
posw = torch.exp(-torch.arange(BS - 1, device=dev, dtype=torch.float32) / args.decay)
EOU_ID = target.lm_head.out_features  # extra class index


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def unit_targets(ids, P, A, units, lang):
    """labels (nb, BS-1) with -100 = ignore; position p-1 for slot j sub m (p = 1 + j*M + m)."""
    K, M = LAYOUT.get(lang, (5, 3))
    N = len(ids)
    lab = torch.full((len(A), BS - 1), -100, dtype=torch.long)
    for b, s in enumerate(A):
        t = s + 1
        cur = units[s - P]
        for j in range(K):
            # tokens of this slot: slot 0 continues the anchor's unit, slot j>=1 is the next unit
            toks = []
            if j == 0:
                while t < N and units[t - P] == cur:
                    toks.append(t)
                    t += 1
            else:
                if t >= N:
                    break
                u = units[t - P]
                while t < N and units[t - P] == u:
                    toks.append(t)
                    t += 1
            for m in range(M):
                p = 1 + j * M + m
                if m < len(toks):
                    lab[b, p - 1] = ids[toks[m]]
                elif m == len(toks):
                    lab[b, p - 1] = EOU_ID
                    break
            if len(toks) > M:  # truncated slot: later slots are still trained (their alignment is well defined)
                pass
    return lab


def block_forward(ids_t, feats, At, lang):
    N = ids_t.shape[1]
    nb = At.numel()
    th = draft.hidden_norm(draft.fc(feats))
    cos_c, sin_c = draft.rotary_emb(th, torch.arange(N, device=dev).unsqueeze(0))
    blk = torch.full((nb, BS), MASK, dtype=torch.long, device=dev)
    blk[:, 0] = ids_t[0, At]
    with torch.no_grad():
        h = target.model.embed_tokens(blk).float()
    if extras is not None:
        lid = LAYOUT_ID[LAYOUT.get(lang, (5, 3))]
        h = h + extras.pos(lid * BS + torch.arange(BS, device=dev)).unsqueeze(0)
    h = h.reshape(1, nb * BS, -1)
    pos_n = (At.unsqueeze(1) + torch.arange(BS, device=dev).unsqueeze(0)).reshape(1, -1)
    cos_n, sin_n = draft.rotary_emb(h, pos_n)
    qb = torch.arange(nb, device=dev).repeat_interleave(BS)
    m_ctx = torch.arange(N, device=dev).unsqueeze(0) < At[qb].unsqueeze(1)
    m_blk = qb.unsqueeze(1) == qb.unsqueeze(0)
    mask = torch.cat([m_ctx, m_blk], 1)[None, None]
    for layer in draft.layers:
        at = layer.self_attn
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x).view(1, -1, NH, HD)).transpose(1, 2)
        kn = at.k_norm(at.k_proj(x).view(1, -1, NKV, HD)).transpose(1, 2)
        vn = at.v_proj(x).view(1, -1, NKV, HD).transpose(1, 2)
        q, kn = rope(q, cos_n, sin_n), rope(kn, cos_n, sin_n)
        kc = rope(at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2), cos_c, sin_c)
        vc = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
        K_, V_ = torch.cat([kc, kn], 2), torch.cat([vc, vn], 2)
        o = F.scaled_dot_product_attention(q, K_, V_, attn_mask=mask, scale=at.scaling, enable_gqa=True)
        h = h + at.o_proj(o.transpose(1, 2).reshape(1, -1, NH * HD))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    hb = draft.norm(h).view(nb, BS, -1)[:, 1:, :]
    logits = target.lm_head(hb.to(torch.bfloat16)).float()
    if extras is not None:
        logits = torch.cat([logits, extras.eou(hb.float())], -1)
    return logits, hb


def seq_loss(P, out_ids, lang, units):
    ids_list = (P + out_ids)[: args.max_len]
    Pn = len(P)
    ids = torch.tensor([ids_list], device=dev)
    N = ids.shape[1]
    with torch.no_grad():
        out = target.model(ids, output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1).float()
    A = sorted(random.sample(range(Pn, N - 1), min(args.anchors, N - 1 - Pn)))
    At = torch.tensor(A, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, hb = block_forward(ids, feats, At, lang)
    if args.layout == "token":
        gidx = At.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0)
        labels = torch.where(gidx < N, ids[0, gidx.clamp(max=N - 1)], torch.full_like(gidx, -100))
    else:
        labels = unit_targets(ids_list, Pn, A, units, lang).to(dev)
    valid = labels != -100
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.clamp(min=0).reshape(-1), reduction="none").view(labels.shape)
    w = posw.unsqueeze(0) * valid
    loss = (ce * w).sum() / w.sum().clamp(min=1)
    if sel is not None:
        # pseudo-likelihood of the gold path under the selector (teacher forcing: previous = gold), candidates = top-16
        lp = torch.log_softmax(logits, -1)
        ti = lp.detach().topk(16, -1).indices
        prev = torch.cat([ids[0, At].unsqueeze(1), labels[:, :-1].clamp(min=0)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            Ssc = sel(hb, prev, ti, lp.gather(-1, ti))
        hit = (ti == labels.unsqueeze(-1)) & valid.unsqueeze(-1)
        m = hit.any(-1)
        if m.any():
            loss = loss + args.sel_weight * F.cross_entropy(Ssc[m], hit[m].float().argmax(-1))
    if args.tau_beta > 0:
        # expected number of accepted draft tokens under the drafter's own probabilities of the gold tokens
        pg = torch.where(valid, torch.exp(-ce), torch.zeros_like(ce))
        e_acc = torch.cumprod(pg, 1).sum(1)
        loss = loss - args.tau_beta * e_acc.mean() / 4.0
    acc = ((logits.argmax(-1) == labels) & valid).float().sum(0) / valid.float().sum(0).clamp(min=1)
    return loss, acc


ucache = {}
t0 = time.time()
ema_loss, ema_acc = None, None
for step in range(n_steps):
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for k in range(args.accum):
        i = (step * args.accum + k) % len(mine)
        P, out_ids, lang = mine[i]
        if args.layout == "unit" and i not in ucache:
            ucache[i] = units_of(out_ids[: max(0, args.max_len - len(P))], lang)
        loss, acc = seq_loss(P, out_ids, lang, ucache.get(i))
        (loss / args.accum).backward()
        tot += loss.item() / args.accum
        ema_acc = acc if ema_acc is None else 0.98 * ema_acc + 0.02 * acc
    if ddp:
        for p in params + (list(extras.parameters()) if extras is not None else []) + (list(sel.parameters()) if sel is not None else []):
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(params + (list(extras.parameters()) if extras is not None else []) + (list(sel.parameters()) if sel is not None else []), 1.0)
    opt.step()
    sched.step()
    ema_loss = tot if ema_loss is None else 0.98 * ema_loss + 0.02 * tot
    if rank == 0 and (step % 25 == 0 or step == n_steps - 1):
        el = time.time() - t0
        print(f"step {step}/{n_steps} loss {ema_loss:.4f} acc@1..4 {[round(x, 3) for x in ema_acc[:4].tolist()]} "
              f"{el/60:.1f}min eta {el/(step+1)*(n_steps-step-1)/60:.1f}min", flush=True)

if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    draft.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    for f in glob.glob(f"{args.init}/*.py") + [f"{args.init}/config.json"]:
        shutil.copy(f, args.out)
    if extras is not None:
        torch.save({"state": extras.state_dict(), "layout": LAYOUT, "layout_id": {f"{k[0]}x{k[1]}": v for k, v in LAYOUT_ID.items()}},
                   f"{args.out}/unit_extras.pt")
    if sel is not None:
        torch.save(sel.state_dict(), f"{args.out}/sel.pt")
    open(f"{args.out}/TRAINED_OK", "w").write(json.dumps(vars(args)))
    print("saved", args.out, flush=True)
if ddp:
    dist.barrier()
    dist.destroy_process_group()
