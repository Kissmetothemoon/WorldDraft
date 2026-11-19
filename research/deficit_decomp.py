"""Where is the block drafter's per-token deficit? Decompose the excess error (block - AR) at positions WITHOUT within-block
dependency (I_k < 0.01) by token type, language, offset and target entropy, for several block / AR drafter pairs (dose-response).

usage: deficit_decomp.py <mi_tag> <tokenizer> <block_tag>:<ar_tag> [...]  -> work/mi_<mi_tag>/deficit_<mi_tag>.md
token types (target token at output index a+k): ws = word start, iw = in-word whole-character token,
bp = byte piece (starts or ends inside a multi-byte character), o = whitespace / punctuation only.
"""
import glob
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, classify_tokens, pload, token_byte_table

mi_tag, tokp = sys.argv[1], sys.argv[2]
pairs = [p.split(":") for p in sys.argv[3:]]
from transformers import AutoTokenizer

bt = token_byte_table(AutoTokenizer.from_pretrained(tokp))
MI = defaultdict(dict)
for f in sorted(glob.glob(f"{ROOT}/work/mi_{mi_tag}/mi_*.pkl")):
    for m in pload(f):
        MI[m["id"]][m["a"]] = m
LANG_ORDER = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
TYPES = ["ws", "iw", "bp", "o"]
CLS = {}


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/work/{tag}/tf_*.pkl")):
        for r in pload(f):
            if r["id"] in MI and r.get("corr") is not None:
                out[r["id"]] = r
    return out


def ttype(sid, rec, t):
    if sid not in CLS:
        c = classify_tokens(rec["output_ids"], bt, rec["lang"])
        CLS[sid] = (np.array(c["cls"]), np.array(c["bpiece"]))
    cls, bp = CLS[sid]
    if t >= len(cls):
        return None
    if bp[t]:
        return "bp"
    return {0: "o", 1: "ws", 2: "iw"}[int(cls[t])]


def rows_for(btag, atag):
    B, A = load(btag), load(atag)
    rows = []
    for sid, anc in MI.items():
        if sid not in B or sid not in A:
            continue
        rb, ra = B[sid], A[sid]
        cb, ca = rb["corr"], ra["corr"]
        for a, m in anc.items():
            if a >= cb.shape[0] or a >= ca.shape[0]:
                continue
            for k in range(2, 9):
                vb, va = cb[a, k - 1], ca[a, k - 1]
                if vb == 2 or va == 2 or not np.isfinite(m["I"][k - 1]):
                    continue
                ty = ttype(sid, rb, a + k)
                if ty is None:
                    continue
                rows.append((rb["lang"], ty, k, float(max(m["I"][k - 1], 0.0)), float(m["Hc"][k - 1]), int(vb == 0), int(va == 0)))
    return rows


L = [f"# Per-token deficit of block drafting at positions without within-block dependency (MI={mi_tag})\n",
     "Rows: anchor x offset k=2..8 with I_k < 0.01 (target-sampled, T=1, M=32); excess = block error - AR error (teacher-forced).",
     "Types: ws word start, iw in-word whole-character token, bp byte piece, o whitespace/punctuation.",
     "'share of excess' = n_type * excess_type / sum over types (how much of the language's total deficit each type carries).\n"]
dose = []
for bi, (btag, atag) in enumerate(pairs):
    R = rows_for(btag, atag)
    lang = np.array([r[0] for r in R])
    ty = np.array([r[1] for r in R])
    X = np.array([r[2:] for r in R], dtype=np.float64)  # k, I, Hc, bw, aw
    z = X[:, 1] < 0.01
    L.append(f"## {btag} vs {atag} (records {len(R)}, with I<0.01: {int(z.sum())})\n")
    L.append("| lang | n | type mix ws/iw/bp/o (%) | excess ws | excess iw | excess bp | excess o | share of excess ws/iw/bp/o (%) | excess, Hc<0.05 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    pooled = {}
    for l in LANG_ORDER + ["ALL"]:
        s = z & ((lang == l) if l != "ALL" else True)
        n = int(s.sum())
        if n < 300:
            continue
        mix, exc, contrib = [], {}, {}
        for t in TYPES:
            u = s & (ty == t)
            mix.append(100 * u.sum() / n)
            if u.sum() >= 30:
                exc[t] = X[u, 3].mean() - X[u, 4].mean()
                contrib[t] = u.sum() * exc[t]
            else:
                exc[t], contrib[t] = np.nan, 0.0
        tot = sum(contrib.values())
        lowH = s & (X[:, 2] < 0.05)
        eH = X[lowH, 3].mean() - X[lowH, 4].mean() if lowH.sum() >= 30 else np.nan
        pooled[l] = exc
        L.append(f"| {l} | {n} | {'/'.join(f'{x:.0f}' for x in mix)} | " + " | ".join(f"{exc[t]:+.3f}" for t in TYPES)
                 + f" | {'/'.join(f'{100*contrib[t]/tot:.0f}' if tot > 0 else '-' for t in TYPES)} | {eH:+.3f} |")
    L.append("")
    # by offset, pooled over languages
    L.append("By offset k (I<0.01, all languages): excess per type\n")
    L.append("| k | " + " | ".join(TYPES) + " |")
    L.append("|---|" + "---|" * len(TYPES))
    for k in range(2, 9):
        cells = []
        for t in TYPES:
            u = z & (ty == t) & (X[:, 0] == k)
            cells.append(f"{X[u, 3].mean() - X[u, 4].mean():+.3f} (n={int(u.sum())})" if u.sum() >= 30 else "-")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    dose.append((btag, atag, pooled))
# dose-response summary
L.append("## Dose-response summary: excess at I<0.01 per type (rows = drafter pairs)\n")
for l in ["ALL", "en", "zh", "th", "hi", "bn", "te", "am", "sr-Cyrl"]:
    L.append(f"### {l}\n")
    L.append("| block vs AR | " + " | ".join(TYPES) + " |")
    L.append("|---|" + "---|" * len(TYPES))
    for btag, atag, pooled in dose:
        if l in pooled:
            L.append(f"| {btag} vs {atag} | " + " | ".join(f"{pooled[l][t]:+.3f}" for t in TYPES) + " |")
    L.append("")
txt = "\n".join(L) + "\n"
open(f"{ROOT}/work/mi_{mi_tag}/deficit_{mi_tag}.md", "w", encoding="utf-8").write(txt)
print(txt)
