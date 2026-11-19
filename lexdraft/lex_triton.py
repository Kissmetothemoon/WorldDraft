"""LexDraft walk on GPU: one Triton program per request walks the block positions left to right.

At block position s the current word (accepted history + tokens drafted so far in this block) gives an in-word context
(longest of n = 3, 2, 1 found in the table). Candidates = drafter top-16 + the context's top-4 continuations;
score = log q_drafter + lam * log p_lex (continuation tokens: pair table / floor; word-start or non-word tokens:
log p(word ends)); greedy pick; update the word state. Identical rule to sd_lex.py (validated by lex_walk_ref).
"""
import numpy as np
import torch
import triton
import triton.language as tl

NB = 32  # candidate lanes (16 top-k + 4 lexicon + padding)


@triton.jit
def _bsearch(keys_ptr, n, q, ITERS: tl.constexpr):
    # q: tensor of int64 queries; returns index of q in keys or -1
    lo = tl.zeros_like(q)
    hi = tl.zeros_like(q) + n
    for _ in range(ITERS):
        mid = (lo + hi) // 2
        active = lo < hi
        midc = tl.minimum(mid, n - 1)
        km = tl.load(keys_ptr + midc, mask=active, other=0)
        go_right = active & (km < q)
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)
    loc = tl.minimum(lo, n - 1)
    hit = (lo < n) & (tl.load(keys_ptr + loc, mask=lo < n, other=-1) == q)
    return tl.where(hit, lo, -1)


@triton.jit
def _lex_walk_kernel(topi_ptr, topl_ptr, logit_ptr, lse_ptr, hist_ptr, out_ptr,
                     ckeys_ptr, cend_ptr, cfloor_ptr, ctop_ptr, pkeys_ptr, plp_ptr, cont_ptr, sw_ptr,
                     meta_ptr, lam_ptr, V,
                     S: tl.constexpr, H: tl.constexpr, K: tl.constexpr, E: tl.constexpr, NBL: tl.constexpr,
                     ITERS: tl.constexpr):
    b = tl.program_id(0)
    nctx = tl.load(meta_ptr)
    npair = tl.load(meta_ptr + 1)
    lam = tl.load(lam_ptr)
    # ---- word state from history (right-aligned, -1 = pad): walk back until a non-continuation token
    wl = 0
    a1 = 0
    a2 = 0
    a3 = 0
    stop = 0
    for j in range(H):
        t = tl.load(hist_ptr + b * H + (H - 1 - j))
        valid = (stop == 0) & (t >= 0)
        c = tl.load(cont_ptr + tl.maximum(t, 0))
        a3 = tl.where(valid & (wl == 2), t, a3)
        a2 = tl.where(valid & (wl == 1), t, a2)
        a1 = tl.where(valid & (wl == 0), t, a1)
        wl = tl.where(valid, wl + 1, wl)
        # the first non-continuation token met is the word start: it must be a valid word start
        sw = tl.load(sw_ptr + tl.maximum(t, 0))
        ends = valid & (c == 0)
        wl = tl.where(ends & (sw == 0), 0, wl)
        stop = tl.where(ends | (t < 0), 1, stop)
    lanes = tl.arange(0, NBL)
    for s in range(S):
        row = b * S + s
        # ---- context lookup with backoff 3 -> 2 -> 1
        k3 = (3 | (a1.to(tl.int64) << 2) | (a2.to(tl.int64) << 20) | (a3.to(tl.int64) << 38))
        k2 = (2 | (a1.to(tl.int64) << 2) | (a2.to(tl.int64) << 20))
        k1 = (1 | (a1.to(tl.int64) << 2))
        q = tl.where(lanes == 0, k3, tl.where(lanes == 1, k2, k1)).to(tl.int64)
        need = ((lanes == 0) & (wl >= 3)) | ((lanes == 1) & (wl >= 2)) | ((lanes == 2) & (wl >= 1))
        ci_all = _bsearch(ckeys_ptr, nctx, q, ITERS)
        ci_all = tl.where(need, ci_all, -1)
        c3 = tl.max(tl.where(lanes == 0, ci_all, -1), 0)
        c2 = tl.max(tl.where(lanes == 1, ci_all, -1), 0)
        c1 = tl.max(tl.where(lanes == 2, ci_all, -1), 0)
        ci = tl.where(c3 >= 0, c3, tl.where(c2 >= 0, c2, c1))
        has = ci >= 0
        cis = tl.maximum(ci, 0)
        # ---- candidates
        is_top = lanes < K
        is_ext = (lanes >= K) & (lanes < K + E)
        cand_top = tl.load(topi_ptr + row * K + lanes, mask=is_top, other=-1)
        cand_ext = tl.load(ctop_ptr + cis * E + (lanes - K), mask=is_ext & has, other=-1)
        cand = tl.where(is_top, cand_top, cand_ext)
        live = cand >= 0
        lp_top = tl.load(topl_ptr + row * K + lanes, mask=is_top, other=-1e30)
        lse = tl.load(lse_ptr + row)
        lg_ext = tl.load(logit_ptr + row.to(tl.int64) * V + tl.maximum(cand, 0), mask=is_ext & live, other=-1e30).to(tl.float32)
        lp = tl.where(is_top, lp_top, lg_ext - lse)
        # ---- lexical score
        contc = tl.load(cont_ptr + tl.maximum(cand, 0), mask=live, other=0)
        pq = (cis.to(tl.int64) << 18) | tl.maximum(cand, 0).to(tl.int64)
        pi = _bsearch(pkeys_ptr, npair, pq, ITERS)
        plp = tl.load(plp_ptr + tl.maximum(pi, 0), mask=pi >= 0, other=0.0)
        fl = tl.load(cfloor_ptr + cis)
        en = tl.load(cend_ptr + cis)
        lex = tl.where(contc != 0, tl.where(pi >= 0, plp, fl), en)
        lex = tl.where(has, lex, 0.0)
        score = tl.where(live, lp + lam * lex, -1e30)
        best = tl.argmax(score, 0)
        x = tl.max(tl.where(lanes == best, cand, -1), 0)
        tl.store(out_ptr + row, x)
        # ---- word state update
        cx = tl.load(cont_ptr + x)
        sx = tl.load(sw_ptr + x)
        cont_word = (cx != 0) & (wl > 0)
        a3 = tl.where(cont_word, a2, tl.where(sx != 0, 0, a3))
        a2 = tl.where(cont_word, a1, tl.where(sx != 0, 0, a2))
        a1 = tl.where(cont_word | (sx != 0), x, a1)
        wl = tl.where(cont_word, wl + 1, tl.where(sx != 0, 1, 0))


class LexTables:
    """Fixed-capacity GPU buffers (pointer-stable, so CUDA-graph safe); load(lang) copies a language in place."""

    def __init__(self, table_dir, cont, sw, device, cap_ctx=4_000_000, cap_pair=24_000_000):
        self.dir, self.dev = table_dir, device
        big = torch.iinfo(torch.int64).max
        self.ckeys = torch.full((cap_ctx,), big, dtype=torch.int64, device=device)
        self.cend = torch.zeros(cap_ctx, device=device)
        self.cfloor = torch.zeros(cap_ctx, device=device)
        self.ctop = torch.full((cap_ctx, 4), -1, dtype=torch.int32, device=device)
        self.pkeys = torch.full((cap_pair,), big, dtype=torch.int64, device=device)
        self.plp = torch.zeros(cap_pair, device=device)
        self.meta = torch.zeros(2, dtype=torch.int64, device=device)
        self.lam = torch.zeros(1, device=device)
        self.cont = torch.tensor(cont.astype(np.int8), device=device)
        self.sw = torch.tensor(sw.astype(np.int8), device=device)
        self.iters = int(np.ceil(np.log2(cap_pair))) + 1
        self.lang, self.np = None, None

    def load(self, lang, lam):
        import os
        path = f"{self.dir}/{lang}.npz"
        self.lam.fill_(float(lam) if os.path.exists(path) else 0.0)
        if not os.path.exists(path):
            self.meta.zero_()
            self.lang, self.np = lang, None
            return
        if lang != self.lang:
            z = np.load(path)
            nc, npair = len(z["ctx_keys"]), len(z["pair_keys"])
            assert nc <= self.ckeys.numel() and npair <= self.pkeys.numel(), (lang, nc, npair)
            self.ckeys[:nc].copy_(torch.from_numpy(z["ctx_keys"]))
            self.ckeys[nc:].fill_(torch.iinfo(torch.int64).max)
            self.cend[:nc].copy_(torch.from_numpy(z["ctx_end"]))
            self.cfloor[:nc].copy_(torch.from_numpy(z["ctx_floor"]))
            self.ctop[:nc].copy_(torch.from_numpy(z["ctx_top"]))
            self.pkeys[:npair].copy_(torch.from_numpy(z["pair_keys"]))
            self.pkeys[npair:].fill_(torch.iinfo(torch.int64).max)
            self.plp[:npair].copy_(torch.from_numpy(z["pair_lp"]))
            self.meta.copy_(torch.tensor([nc, npair]))
            self.lang, self.np = lang, z
        torch.cuda.synchronize()


def lex_walk(logits, hist, tabs, K=16, E=4):
    """logits (B,S,V) any float dtype; hist (B,H) int32 (right-aligned, -1 pad) -> tokens (B,S) int32"""
    B, S, V = logits.shape
    # greedy: log-softmax normaliser is a per-row constant and does not change the argmax -> score raw logits
    tl_, ti = logits.topk(K, -1)
    tl_ = tl_.float().contiguous()
    lse = torch.zeros(B, S, device=logits.device)
    out = torch.empty(B, S, dtype=torch.int32, device=logits.device)
    _lex_walk_kernel[(B,)](ti.to(torch.int32).contiguous(), tl_, logits.contiguous(), lse, hist.contiguous(), out,
                           tabs.ckeys, tabs.cend, tabs.cfloor, tabs.ctop, tabs.pkeys, tabs.plp, tabs.cont, tabs.sw,
                           tabs.meta, tabs.lam, V,
                           S=S, H=hist.shape[1], K=K, E=E, NBL=NB, ITERS=tabs.iters)
    return out


def lex_walk_ref(logits, hist, tabs, K=16, E=4):
    lam = float(tabs.lam.item())
    """pure-python reference of the same rule (for validation)"""
    z = tabs.np
    ck = {int(k): i for i, k in enumerate(z["ctx_keys"])}
    pk = {int(k): float(v) for k, v in zip(z["pair_keys"], z["pair_lp"])}
    cont, sw = tabs.cont.cpu().numpy(), tabs.sw.cpu().numpy()
    lpall = torch.log_softmax(logits.float(), -1).cpu()
    outs = []
    for b in range(logits.shape[0]):
        h = [int(t) for t in hist[b].tolist() if t >= 0]
        w = []
        for t in reversed(h):
            w.append(t)
            if not cont[t]:
                break
        w.reverse()
        if w and not sw[w[0]]:
            w = []
        row = []
        for s in range(logits.shape[1]):
            lp = lpall[b, s]
            top = logits[b, s].topk(K).indices.tolist()  # same candidate set (and tie order) as the kernel
            ci = None
            for n in (3, 2, 1):
                if len(w) >= n:
                    key = n | (w[-1] << 2) | ((w[-2] if n >= 2 else 0) << 20) | ((w[-3] if n >= 3 else 0) << 38)
                    if key in ck:
                        ci = ck[key]
                        break
            cand = list(top)
            if ci is not None:
                cand += [int(x) for x in z["ctx_top"][ci] if x >= 0]
            best, bx = -1e30, None
            for c in cand:
                lx = 0.0
                if ci is not None:
                    lx = pk.get((ci << 18) | c, float(z["ctx_floor"][ci])) if cont[c] else float(z["ctx_end"][ci])
                sc = float(lp[c]) + lam * lx
                if sc > best:
                    best, bx = sc, c
            row.append(bx)
            if cont[bx] and w:
                w.append(bx)
            elif sw[bx]:
                w = [bx]
            else:
                w = []
        outs.append(row)
    return torch.tensor(outs)
