"""Teacher-forced DFlash analysis.

For a greedy target trajectory, re-run the DFlash drafter from EVERY anchor position s in the response
(context = target features of positions < s, block = [token_s, MASK x 15]) and record whether each of the
15 drafted tokens equals the trajectory token. Under greedy decoding this reproduces exactly what DFlash
would propose and accept whenever its block starts at s.

Modes:
  --mode tf        : analyse a shard of trajectories (from gen_vllm.py)
  --mode validate  : run real spec_generate on some prompts and check the offline chain reproduces it
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

from common import build_messages, chat_ids, jload, pdump, shard

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="tf", choices=["tf", "validate", "speed", "reveal", "solve"])
ap.add_argument("--words-per-seq", type=int, default=40)
ap.add_argument("--temp", type=float, default=0.0, help="speed mode: target sampling temperature (DFlash sample-match)")
ap.add_argument("--no-ar", action="store_true", help="speed mode: skip the AR baseline (tau only)")
ap.add_argument("--speed-per-group", type=int, default=20)
ap.add_argument("--speed-max-new", type=int, default=0, help="override max new tokens in speed mode (official protocol: 2048)")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--batch", type=int, default=96)
ap.add_argument("--nval", type=int, default=24)
ap.add_argument("--twopass", default="", help="solve mode: comma list of m for two-pass self-reveal (reveal-trained drafters)")
ap.add_argument("--clean-ids", default="", help="solve mode: only trajectories in this json id list")
args = ap.parse_args()

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import rotate_half

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev,
                                              attn_implementation="sdpa").eval()
draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
LIDS = draft.target_layer_ids
BS = draft.block_size
MASK = draft.mask_token_id
cfg = draft.config
HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads


def rope(x, cos, sin):  # x (B,H,L,D); cos/sin (B,L,D)
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


@torch.inference_mode()
def target_pass(ids):
    out = target(ids, output_hidden_states=True)
    feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], dim=-1)
    return out.logits[0], feats


@torch.inference_mode()
def offline_blocks(ids, feats, anchors, blocks=None, topk=1):
    """ids (1,N); feats (1,N,5H); anchors list[int]; blocks optional (A,BS) block contents (default [x_s, MASK...]).
    Returns pred (A,BS-1) long, prob (A,BS-1) float; with topk > 1 also top-k ids (A,BS-1,topk)."""
    N = ids.shape[1]
    th = draft.hidden_norm(draft.fc(feats))  # (1,N,H)
    pos_ctx = torch.arange(N, device=dev).unsqueeze(0)
    cos_c, sin_c = draft.rotary_emb(th, pos_ctx)
    ctx = []
    for layer in draft.layers:
        at = layer.self_attn
        k = at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2)
        v = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
        ctx.append((rope(k, cos_c, sin_c), v))
    preds, probs, tops = [], [], []
    for b0 in range(0, len(anchors), args.batch):
        A = torch.tensor(anchors[b0:b0 + args.batch], device=dev)
        B = A.numel()
        Lc = int(A.max().item())  # context needed: positions < s
        if blocks is None:
            blk = torch.full((B, BS), MASK, dtype=torch.long, device=dev)
            blk[:, 0] = ids[0, A]
        else:
            blk = blocks[b0:b0 + args.batch]
        h = target.model.embed_tokens(blk)
        pos_n = A.unsqueeze(1) + torch.arange(BS, device=dev).unsqueeze(0)
        cos_n, sin_n = draft.rotary_emb(h, pos_n)
        allow_ctx = torch.arange(Lc, device=dev).unsqueeze(0) < A.unsqueeze(1)  # (B,Lc)
        mask = torch.cat([allow_ctx, torch.ones(B, BS, dtype=torch.bool, device=dev)], 1)[:, None, None, :]
        for li, layer in enumerate(draft.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(B, BS, NH, HD)).transpose(1, 2)
            kn = at.k_norm(at.k_proj(x).view(B, BS, NKV, HD)).transpose(1, 2)
            vn = at.v_proj(x).view(B, BS, NKV, HD).transpose(1, 2)
            q, kn = rope(q, cos_n, sin_n), rope(kn, cos_n, sin_n)
            kc, vc = ctx[li]
            K = torch.cat([kc[:, :, :Lc].expand(B, -1, -1, -1), kn], 2)
            V = torch.cat([vc[:, :, :Lc].expand(B, -1, -1, -1), vn], 2)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, scale=at.scaling, enable_gqa=True)
            h = h + at.o_proj(o.transpose(1, 2).reshape(B, BS, NH * HD))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = target.lm_head(draft.norm(h)[:, 1:, :]).float()
        p = torch.softmax(logits, -1)
        mp, am = p.max(-1)
        preds.append(am)
        probs.append(mp)
        if topk > 1:
            tops.append(p.topk(topk, -1).indices)
    if topk > 1:
        return torch.cat(preds), torch.cat(probs), torch.cat(tops)
    return torch.cat(preds), torch.cat(probs)


def analyse(prompt_ids, output_ids):
    ids = torch.tensor([prompt_ids + output_ids], device=dev)
    P, N = len(prompt_ids), ids.shape[1]
    logits, feats = target_pass(ids)
    lg = logits[P - 1:N - 1].float()  # predicts tokens P..N-1
    lp = torch.log_softmax(lg, -1)
    ent = -(lp.exp() * lp).sum(-1)
    tp, ti = lp.exp().topk(5, -1)
    am = ti[:, 0]
    tgt = ids[0, P:N]
    devm = am != tgt  # trajectory token differs from HF target argmax (bf16 near-ties between vLLM and HF)
    # gold for position g = HF target argmax given the trajectory prefix; the trajectory is only the context.
    gold_all = torch.cat([ids[0, :P], am])  # index by absolute position
    dev_all = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), devm])
    t0 = N  # kept for compatibility: no truncation
    anchors = list(range(P, N - 1))
    res = {"P": P, "N": N, "t0": t0, "ndev": int(devm.sum().item()), "dev": devm.cpu().numpy(),
           "ent": ent.half().cpu().numpy(), "top5_id": ti.int().cpu().numpy(), "top5_p": tp.half().cpu().numpy()}
    if not anchors:
        res.update(corr=None, dprob=None)
        return res
    pred, prob = offline_blocks(ids, feats, anchors)
    A = torch.tensor(anchors, device=dev)
    gidx = A.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0)  # (A,15)
    inb = gidx <= N - 1
    g = gidx.clamp(max=N - 1)
    # a block position is valid if every earlier in-block position matched the trajectory (no deviation before it)
    prior_dev = torch.cumsum(dev_all[g].int(), 1) - dev_all[g].int()
    valid = inb & (prior_dev == 0)
    corr = torch.where(valid, (pred == gold_all[g]).to(torch.uint8), torch.full_like(pred, 2, dtype=torch.uint8))
    res.update(corr=corr.cpu().numpy(), dprob=prob.half().cpu().numpy())
    return res


@torch.inference_mode()
def tsample(logits):
    if args.temp < 1e-5:
        return logits.argmax(-1)
    p = torch.softmax(logits.float() / args.temp, -1)
    return torch.multinomial(p.view(-1, p.shape[-1]), 1).view(p.shape[:-1])


@torch.inference_mode()
def spec_generate(input_ids, max_new_tokens, stop_ids):
    """Copy of DFlash spec_generate (greedy) that also returns acceptance lengths."""
    from transformers import DynamicCache
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    out_ids = torch.full((1, max_len + BS), MASK, dtype=torch.long, device=dev)
    pos = torch.arange(out_ids.shape[1], device=dev).unsqueeze(0)
    kv_t, kv_d = DynamicCache(), DynamicCache()
    o = target(input_ids, position_ids=pos[:, :n_in], past_key_values=kv_t, use_cache=True, logits_to_keep=1,
               output_hidden_states=True)
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = tsample(o.logits)[0, -1]
    th = torch.cat([o.hidden_states[l + 1] for l in LIDS], -1)
    accs, start = [], n_in
    while start < max_len:
        b = out_ids[:, start:start + BS].clone()
        emb = target.model.embed_tokens(b)
        dl = target.lm_head(draft(target_hidden=th, noise_embedding=emb,
                                  position_ids=pos[:, kv_d.get_seq_length():start + BS], past_key_values=kv_d,
                                  use_cache=True, is_causal=False)[:, -BS + 1:, :])
        kv_d.crop(start)
        b[:, 1:] = dl.argmax(-1)
        o = target(b, position_ids=pos[:, start:start + BS], past_key_values=kv_t, use_cache=True,
                   output_hidden_states=True)
        post = tsample(o.logits)
        a = int((b[:, 1:] == post[:, :-1]).cumprod(1).sum(1)[0].item())
        out_ids[:, start:start + a + 1] = b[:, :a + 1]
        out_ids[:, start + a + 1] = post[:, a]
        start += a + 1
        kv_t.crop(start)
        th = torch.cat([o.hidden_states[l + 1] for l in LIDS], -1)[:, :a + 1, :]
        accs.append(a + 1)
        if any(s in out_ids[0, n_in:start].tolist() for s in stop_ids):
            break
    gen = out_ids[0, n_in:start].tolist()
    for k, t in enumerate(gen):
        if t in stop_ids:
            gen = gen[:k + 1]
            break
    return gen, accs


def chain_from_corr(res, n_out):
    """Acceptance lengths of the real SD loop, reconstructed from the offline matrix."""
    P = res["P"]
    accs, s = [], P
    while s - P < n_out - 1 and res["corr"] is not None and s - P < len(res["corr"]):
        row = res["corr"][s - P]
        a = 0
        while a < BS - 1 and row[a] == 1:
            a += 1
        accs.append(a + 1)
        s += a + 1
    return accs


stop = sorted({tok.eos_token_id} | {t for t in [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")]
                                    if isinstance(t, int) and 0 <= t < len(tok) and t != tok.unk_token_id})

if args.mode == "speed":
    # Stage 0: wall-clock AR (HF generate) vs DFlash spec_generate, batch 1, greedy, same max tokens
    from collections import defaultdict
    from common import MAX_NEW
    groups = defaultdict(list)
    for r in jload(args.inp):
        groups[(r["task"], r["lang"])].append(r)
    sel = []
    for key in sorted(groups):
        sel += groups[key][: args.speed_per_group]
    sel = shard(sel, args.shard, args.nshard)
    rep = []
    warm = torch.tensor([chat_ids(tok, build_messages("chat", "en", "Hello!"))], device=dev)
    target.generate(warm, max_new_tokens=16, do_sample=False)
    spec_generate(warm, 32, stop)
    for i, r in enumerate(sel):
        pid = torch.tensor([chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))], device=dev)
        mx = args.speed_max_new or MAX_NEW[r["task"]]
        torch.cuda.synchronize(); t = time.time()
        if args.no_ar:
            ar = []
        else:
            ar = target.generate(pid, max_new_tokens=mx, do_sample=False, eos_token_id=stop, pad_token_id=stop[0])[0, pid.shape[1]:].tolist()
        torch.cuda.synchronize(); t_ar = max(time.time() - t, 1e-6)
        t = time.time()
        sp, accs = spec_generate(pid, mx, stop)
        torch.cuda.synchronize(); t_sp = time.time() - t
        k = min(len(ar), len(sp))
        rep.append({"id": r["id"], "task": r["task"], "lang": r["lang"], "n_ar": len(ar), "n_sp": len(sp),
                    "t_ar": t_ar, "t_sp": t_sp, "tau": sum(accs) / max(1, len(accs)), "accs": accs,
                    "same_prefix": next((j for j in range(k) if ar[j] != sp[j]), k), "identical": ar == sp})
        if i % 10 == 0:
            print(rep[-1]["id"], f"ar {len(ar)/t_ar:.1f} tok/s  dflash {len(sp)/t_sp:.1f} tok/s  tau {rep[-1]['tau']:.2f}", flush=True)
    pdump(rep, args.out)
elif args.mode == "reveal":
    # Same-anchor intervention: anchor s = w-2 for a multi-token word starting at w.
    #   V0: nothing revealed | Vstart: true word start revealed at offset 2 | Vprev: true token w-1 revealed (control)
    #   Vself: drafter's own V0 prediction for the word start revealed (I2)
    # Outcome: correctness of in-word tokens t = w+1.. (teacher-forced; positions after a trajectory deviation dropped).
    import random
    from common import classify_tokens, token_byte_table
    bt = token_byte_table(tok)
    rows = shard(jload(args.inp), args.shard, args.nshard)
    rng = random.Random(args.shard)
    out, t = [], time.time()
    for i, r in enumerate(rows):
        pidl, oidl = r["prompt_ids"], r["output_ids"]
        ids = torch.tensor([pidl + oidl], device=dev)
        P, N = len(pidl), ids.shape[1]
        with torch.inference_mode():
            logits, feats = target_pass(ids)
            am = logits[P - 1:N - 1].argmax(-1)
        gold = torch.cat([ids[0, :P], am])
        devm = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), am != ids[0, P:N]])
        cl = classify_tokens(oidl, bt, r["lang"])
        words = [k for k in range(2, len(oidl) - 1) if cl["cls"][k] == 1 and cl["wlen"][k] >= 2]
        rng.shuffle(words)
        words = sorted(words[: args.words_per_seq])
        rec = []
        if words:
            W_ = torch.tensor([P + k for k in words], device=dev)
            anc = (W_ - 2).tolist()
            base = torch.full((len(words), BS), MASK, dtype=torch.long, device=dev)
            base[:, 0] = ids[0, W_ - 2]
            v_start, v_prev = base.clone(), base.clone()
            v_start[:, 2] = ids[0, W_]
            v_prev[:, 1] = ids[0, W_ - 1]
            p0, q0 = offline_blocks(ids, feats, anc, base)
            v_self = base.clone()
            v_self[:, 2] = p0[:, 1]
            p1, _ = offline_blocks(ids, feats, anc, v_start)
            p2, _ = offline_blocks(ids, feats, anc, v_prev)
            p3, _ = offline_blocks(ids, feats, anc, v_self)
            for wi, k in enumerate(words):
                w = P + k
                m = cl["wlen"][k]
                for j in range(1, m):
                    off = j + 2  # block offset of token t = w + j
                    tpos = w + j
                    if off > BS - 1 or tpos >= N or bool(devm[w - 1:tpos].any()):
                        break
                    g = int(gold[tpos])
                    rec.append((k, j, m, int(p0[wi, off - 1] == g), int(p1[wi, off - 1] == g), int(p2[wi, off - 1] == g),
                                int(p3[wi, off - 1] == g), int(p0[wi, 1] == gold[w]), int(p0[wi, 0] == gold[w - 1]),
                                float(q0[wi, 1]), int(cl["bpiece"][k + j])))
        out.append({"id": r["id"], "task": r["task"], "lang": r["lang"], "text_out": r["text_out"],
                    "output_ids": oidl, "rec": rec})
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] {time.time()-t:.0f}s", flush=True)
    pdump(out, args.out)
    print("done", len(out), flush=True)
elif args.mode == "solve":
    # Ceilings of block-level fixes, per position of every offline block (same anchors / gold / validity as analyse()):
    #   rank      : rank of the gold token among the drafter's top-16 (255 = absent) -> oracle path selector (DFlash-2 style)
    #   corr_utf8 : first candidate (in drafter order) that is a valid UTF-8 continuation of the accepted prefix
    #   corr2[m]  : two-pass self-reveal: positions 1..m from pass 1 are fed back as block inputs, the rest re-drafted
    import json as _json
    from common import token_byte_table
    bt = token_byte_table(tok)
    V = len(bt)

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

    NEWR = torch.full((4, max(V, target.lm_head.out_features)), -1, dtype=torch.long)
    for t, bs in enumerate(bt):
        for r in range(4):
            NEWR[r, t] = _step(bs, r) if bs else (0 if r == 0 else -1)
    NEWR_c = NEWR.clone()
    NEWR = NEWR.to(dev)
    ms = [int(x) for x in args.twopass.split(",") if x]
    keep = set(_json.load(open(args.clean_ids))) if args.clean_ids else None
    rows = [r for r in shard(jload(args.inp), args.shard, args.nshard) if keep is None or r["id"] in keep]
    out, t = [], time.time()
    for i, r in enumerate(rows):
        pidl, oidl = r["prompt_ids"], r["output_ids"]
        ids = torch.tensor([pidl + oidl], device=dev)
        P, N = len(pidl), ids.shape[1]
        if N - P < 8:
            continue
        logits, feats = target_pass(ids)
        am = logits[P - 1:N - 1].argmax(-1)
        gold_all = torch.cat([ids[0, :P], am])
        dev_all = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), am != ids[0, P:N]])
        anchors = list(range(P, N - 1))
        A = torch.tensor(anchors, device=dev)
        gidx = A.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0)
        inb = gidx <= N - 1
        g = gidx.clamp(max=N - 1)
        prior_dev = torch.cumsum(dev_all[g].int(), 1) - dev_all[g].int()
        valid = inb & (prior_dev == 0)
        gold = gold_all[g]
        pred, prob, top = offline_blocks(ids, feats, anchors, topk=16)

        def cm(x):
            return torch.where(valid, (x == gold).to(torch.uint8), torch.full_like(x, 2, dtype=torch.uint8)).cpu().numpy()

        match = top == gold.unsqueeze(-1)
        rank = torch.where(match.any(-1), match.float().argmax(-1), torch.full_like(pred, 255))
        # UTF-8 state of the trajectory prefix before every position (prompt ends on special tokens -> state 0)
        rs, r_ = [0] * N, 0
        for p_, tid in enumerate(oidl, P):
            r_ = max(int(NEWR_c[r_, tid]), 0)
            rs[p_] = r_
        rs = torch.tensor(rs, device=dev)
        rprev = rs[(g - 1).clamp(min=0)]
        ok = NEWR[rprev.unsqueeze(-1), top] >= 0  # (A,15,16)
        first = torch.where(ok.any(-1), ok.float().argmax(-1), torch.zeros_like(pred))
        chosen = top.gather(-1, first.unsqueeze(-1)).squeeze(-1)
        res = {"id": r["id"], "task": r["task"], "lang": r["lang"], "P": P, "output_ids": oidl, "text_out": r["text_out"],
               "corr": cm(pred), "rank": rank.clamp(max=255).to(torch.uint8).cpu().numpy(), "corr_utf8": cm(chosen),
               "top1_bad": (~ok[..., 0]).to(torch.uint8).cpu().numpy(), "dprob": prob.half().cpu().numpy(), "corr2": {}}
        for m in ms:
            blk = torch.full((len(anchors), BS), MASK, dtype=torch.long, device=dev)
            blk[:, 0] = ids[0, A]
            blk[:, 1:m + 1] = pred[:, :m]
            p2, _ = offline_blocks(ids, feats, anchors, blk)
            res["corr2"][m] = cm(torch.cat([pred[:, :m], p2[:, m:]], 1))
        out.append(res)
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] {time.time()-t:.0f}s", flush=True)
    pdump(out, args.out)
    print("done", len(out), f"{time.time()-t:.0f}s", flush=True)
elif args.mode == "validate":
    rows = jload(args.inp)[: args.nval]
    rep = []
    for r in rows:
        pid = chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))
        gen, accs = spec_generate(torch.tensor([pid], device=dev), 256, stop)
        res = analyse(pid, gen)
        off = chain_from_corr(res, len(gen))
        # anchor-matched: offline acceptance at exactly the anchors the real loop used
        P, same_anchor, n_anchor, s = res["P"], 0, 0, res["P"]
        for a in accs:
            if res["corr"] is None or s - P >= len(res["corr"]) or s + a >= res["t0"]:
                break
            row = res["corr"][s - P]
            ao = 0
            while ao < BS - 1 and row[ao] == 1:
                ao += 1
            same_anchor += int(ao + 1 == a)
            n_anchor += 1
            s += a
        # implementation check: official draft forward (no cache, full context) vs batched offline_blocks
        ids = torch.tensor([pid + gen], device=dev)
        _, feats = target_pass(ids)
        anc = list(range(P, min(res["t0"] - 1, P + 200), 7))[:24]
        mine, _ = offline_blocks(ids, feats, anc)
        agree = tot_tok = 0
        with torch.inference_mode():
            for i_a, s0 in enumerate(anc):
                blk = torch.full((1, BS), MASK, dtype=torch.long, device=dev)
                blk[0, 0] = ids[0, s0]
                hid = draft(target_hidden=feats[:, :s0], noise_embedding=target.model.embed_tokens(blk),
                            position_ids=torch.arange(s0 + BS, device=dev).unsqueeze(0), is_causal=False)
                ref = target.lm_head(hid[:, 1:, :]).argmax(-1)[0]
                agree += int((ref == mine[i_a]).sum().item())
                tot_tok += BS - 1
        rep.append({"id": r["id"], "n_rounds": len(accs), "anchor_match": same_anchor / max(1, n_anchor),
                    "impl_token_agree": agree / max(1, tot_tok), "tau_real": sum(accs) / len(accs),
                    "tau_off": sum(off) / max(1, len(off)), "t0_minus_P": res["t0"] - res["P"], "n_gen": len(gen)})
        print(rep[-1], flush=True)
    print("VALIDATION anchor-matched acc agreement:", sum(x["anchor_match"] for x in rep) / len(rep),
          " impl token agreement:", sum(x["impl_token_agree"] for x in rep) / len(rep), flush=True)
    pdump(rep, args.out)
else:
    rows = shard(jload(args.inp), args.shard, args.nshard)
    out, t = [], time.time()
    for i, r in enumerate(rows):
        res = analyse(r["prompt_ids"], r["output_ids"])
        res.update(id=r["id"], task=r["task"], lang=r["lang"], output_ids=r["output_ids"], finish=r["finish"],
                   text_out=r["text_out"])
        out.append(res)
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] {time.time()-t:.0f}s", flush=True)
    pdump(out, args.out)
    print("done", len(out), f"{time.time()-t:.0f}s", flush=True)
