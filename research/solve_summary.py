"""Summarise tf_dflash.py --mode solve outputs: per-language ceilings of block-level fixes.

usage: solve_summary.py <tag>   reads work/<tag>/solve_*.pkl, writes work/<tag>/solve_report.md and solve_summary.csv
tau@15 = mean accepted length (incl. bonus) of the greedy chain over the trajectory, same as analyze.py.
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, lang_ok, pload

tag = sys.argv[1]
W = f"{ROOT}/work/{tag}"
LANG_ORDER = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]


def chain(corr):
    lead = np.cumprod((corr == 1).astype(np.int32), axis=1).sum(1)
    a, accs = 0, []
    while a < corr.shape[0]:
        acc = int(lead[a]) + 1
        accs.append(acc)
        a += acc
    return accs


S = defaultdict(lambda: defaultdict(float))
REC = defaultdict(lambda: np.zeros((2, 15)))  # lang -> [hits@1, hits@16] per offset (valid positions)
NV = defaultdict(lambda: np.zeros(15))
CAL = defaultdict(lambda: np.zeros((10, 2)))  # lang -> per confidence decile: [n, correct]
variants = None
for f in sorted(glob.glob(f"{W}/solve_*.pkl")):
    for r in pload(f):
        out = r["output_ids"]
        g4 = [tuple(out[i:i + 4]) for i in range(max(0, len(out) - 3))]
        if not lang_ok(r["text_out"], r["lang"]) or (1 - len(set(g4)) / max(1, len(g4))) > 0.5:
            continue
        c = r["corr"]
        valid = c != 2
        rank = r["rank"].astype(np.int32)
        V = {"top1": c, "utf8": r["corr_utf8"]}
        for K in (2, 4, 16):
            V[f"oracle@{K}"] = np.where(valid, (rank < K).astype(np.uint8), 2)
        for m, cc in sorted(r["corr2"].items()):
            V[f"twopass m={m}"] = cc
        variants = list(V)
        key = (r["task"], r["lang"])
        Sk = S[key]
        Sk["n"] += 1
        for name, cc in V.items():
            accs = chain(cc)
            Sk[name + "_acc"] += sum(accs)
            Sk[name + "_rounds"] += len(accs)
        Sk["pos_valid"] += valid.sum()
        Sk["top1_bad"] += (r["top1_bad"].astype(bool) & valid).sum()
        Sk["err"] += (c == 0).sum()
        Sk["err_bad"] += ((c == 0) & r["top1_bad"].astype(bool)).sum()
        # predicted vs realised acceptance from the drafter's own confidence (survival = running product)
        p = r["dprob"].astype(np.float64)
        lead = np.cumprod((c == 1).astype(np.int32), axis=1).sum(1)
        surv = np.cumprod(p, axis=1).sum(1)
        Sk["pred_acc"] += surv.sum()
        Sk["real_acc"] += lead.sum()
        Sk["anchors"] += len(lead)
        l = r["lang"]
        K = c.shape[1]
        REC[l][0][:K] += ((c == 1) & valid).sum(0)
        REC[l][1][:K] += ((rank < 16) & valid).sum(0)
        NV[l][:K] += valid.sum(0)
        d = np.clip((p * 10).astype(int), 0, 9)
        for b in range(10):
            u = (d == b) & valid
            CAL[l][b] += [u.sum(), (u & (c == 1)).sum()]

L = [f"# Ceilings of block-level fixes per language ({tag})\n",
     "tau@15 of the greedy chain; oracle@K = perfect selector among the drafter's top-K per position (DFlash-2 path-selector ceiling);",
     "utf8 = take the first top-16 candidate that is a valid UTF-8 continuation; twopass = re-draft with own first m tokens revealed.\n",
     "| lang | n | " + " | ".join(variants) + " | top-1 invalid UTF-8 (% of positions / % of errors) | predicted / realised accepted drafts |",
     "|---|---|" + "---|" * len(variants) + "---|---|"]
rows = []
for l in LANG_ORDER:
    ks = [k for k in S if k[1] == l]
    if not ks:
        continue
    agg = defaultdict(float)
    for k in ks:
        for a, v in S[k].items():
            agg[a] += v
    taus = {v: agg[v + "_acc"] / agg[v + "_rounds"] for v in variants}
    rows.append({"lang": l, "n": int(agg["n"]), **taus})
    L.append(f"| {l} | {int(agg['n'])} | " + " | ".join(f"{taus[v]:.2f}" for v in variants)
             + f" | {100*agg['top1_bad']/agg['pos_valid']:.2f} / {100*agg['err_bad']/max(1,agg['err']):.2f}"
             + f" | {agg['pred_acc']/agg['anchors']:.2f} / {agg['real_acc']/agg['anchors']:.2f} |")
L += ["", "## Recall@1 / recall@16 per offset (valid positions)\n", "| lang | " + " | ".join(f"k={k}" for k in (1, 2, 4, 8, 15)) + " |",
      "|---|" + "---|" * 5]
for l in LANG_ORDER:
    if l not in NV:
        continue
    cells = [f"{REC[l][0][k-1]/NV[l][k-1]:.2f} / {REC[l][1][k-1]/NV[l][k-1]:.2f}" if NV[l][k-1] else "-" for k in (1, 2, 4, 8, 15)]
    L.append(f"| {l} | " + " | ".join(cells) + " |")
L += ["", "## Calibration of the drafter's top-1 probability (P(correct) in confidence deciles 0.5-0.6 / 0.7-0.8 / 0.9-1.0)\n",
      "| lang | 0.5-0.6 | 0.7-0.8 | 0.9-1.0 |", "|---|---|---|---|"]
for l in LANG_ORDER:
    if l not in CAL:
        continue
    C = CAL[l]
    L.append(f"| {l} | " + " | ".join(f"{C[b,1]/max(1,C[b,0]):.2f} (n={int(C[b,0])})" for b in (5, 7, 9)) + " |")
txt = "\n".join(L) + "\n"
open(f"{W}/solve_report.md", "w", encoding="utf-8").write(txt)
with open(f"{W}/solve_summary.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(txt)
