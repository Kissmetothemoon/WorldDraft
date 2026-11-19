"""Does within-block mutual information explain where block-parallel drafting loses to AR drafting?

Joins mi_probe outputs (per anchor: I_k, Hc_k for offsets k=1..8) with the offline records of a block drafter and an
AR drafter on the SAME trajectories (same anchor s = P + a, same offset k).
usage: mi_analysis.py <mi_tag> <block_tag> <ar_tag> [<block_tag2> <ar_tag2> ...]; writes work/<mi_tag>/mi_report_<block>_<ar>.md
"""
import glob
import math
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, pload

mi_tag = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
MI = defaultdict(dict)  # id -> a -> record
for f in sorted(glob.glob(f"{ROOT}/work/mi_{mi_tag}/mi_*.pkl")):
    for m in pload(f):
        MI[m["id"]][m["a"]] = m
print("anchors", sum(len(v) for v in MI.values()), "sequences", len(MI))


def corr_map(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/work/{tag}/tf_*.pkl")):
        for r in pload(f):
            if r["id"] in MI:
                out[r["id"]] = (r["corr"], r["task"], r["lang"], np.asarray(r["ent"], dtype=np.float32))
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


for btag, atag in pairs:
    B, A = corr_map(btag), corr_map(atag)
    rows = []  # (lang, task, k, I, Hc, b_wrong, a_wrong, both prefixes accepted?)
    for sid, anc in MI.items():
        if sid not in B or sid not in A:
            continue
        cb, task, lang, _ = B[sid]
        ca = A[sid][0]
        for a, m in anc.items():
            if a >= cb.shape[0]:
                continue
            for k in range(2, 9):
                vb, va = cb[a, k - 1], ca[a, k - 1]
                if vb == 2 or va == 2 or not np.isfinite(m["I"][k - 1]):
                    continue
                rows.append((lang, task, k, float(max(m["I"][k - 1], 0.0)), float(m["Hc"][k - 1]), int(vb == 0), int(va == 0)))
    R = np.array([r[2:] for r in rows], dtype=np.float64)  # k, I, Hc, bw, aw
    langs = np.array([r[0] for r in rows])
    tasks = np.array([r[1] for r in rows])
    L = [f"# Within-block mutual information vs drafter errors: block={btag}, AR={atag}, MI={mi_tag}\n",
         f"Token-level records (anchor x offset k=2..8, teacher-forced correctness): {len(R)}.",
         "I_k = I(x_{s+k}; x_{s+1..s+k-1} | x_<=s) estimated by target sampling (T=1, M=32); Hc_k = mean conditional entropy.\n",
         "## 1. Token level: excess block error vs I_k, stratified by conditional entropy Hc and offset k\n",
         "| I_k bin | n | block error | AR error | block - AR | P(block wrong, AR right) |", "|---|---|---|---|---|---|"]
    Ib = np.digitize(R[:, 1], [0.01, 0.05, 0.2, 0.5, 1.0])
    names = ["<0.01", "0.01-0.05", "0.05-0.2", "0.2-0.5", "0.5-1.0", ">=1.0"]
    for b in range(6):
        s = Ib == b
        if s.sum() < 200:
            continue
        L.append(f"| {names[b]} | {int(s.sum())} | {R[s,3].mean():.3f} | {R[s,4].mean():.3f} | {R[s,3].mean()-R[s,4].mean():+.3f} | "
                 f"{((R[s,3]==1)&(R[s,4]==0)).mean():.3f} |")
    # stratified (Hc bin x k) comparison of high-I vs low-I: Mantel-Haenszel style risk difference
    Hb = np.digitize(R[:, 2], [0.05, 0.2, 0.5, 1.0, 2.0])
    hi, lo = R[:, 1] >= 0.2, R[:, 1] < 0.02
    num = den = 0.0
    for key in set(zip(Hb, R[:, 0].astype(int))):
        s = (Hb == key[0]) & (R[:, 0] == key[1])
        x1, x0 = s & hi, s & lo
        if x1.sum() >= 20 and x0.sum() >= 20:
            w = x1.sum() * x0.sum() / (x1.sum() + x0.sum())
            d1 = R[x1, 3].mean() - R[x1, 4].mean()
            d0 = R[x0, 3].mean() - R[x0, 4].mean()
            num += w * (d1 - d0)
            den += w
    L.append(f"\nStratified by Hc bin x offset: excess block error (block - AR) is **{num/den:+.3f}** higher for I>=0.2 than for I<0.02.\n" if den else "")
    # per-language
    L += ["## 2. Per language (all tasks pooled)\n",
          "| lang | n | mean I_k | mean Hc | block err | AR err | excess (block-AR) | excess at I<0.02 | excess at I>=0.2 |",
          "|---|---|---|---|---|---|---|---|---|"]
    lang_stats = {}
    for l in sorted(set(langs)):
        s = langs == l
        x = R[s]
        e_lo = x[x[:, 1] < 0.02]
        e_hi = x[x[:, 1] >= 0.2]
        lang_stats[l] = (x[:, 1].mean(), x[:, 2].mean(), x[:, 3].mean() - x[:, 4].mean())
        f = lambda z: (z[:, 3].mean() - z[:, 4].mean()) if len(z) >= 50 else float("nan")
        L.append(f"| {l} | {len(x)} | {x[:,1].mean():.3f} | {x[:,2].mean():.3f} | {x[:,3].mean():.3f} | {x[:,4].mean():.3f} | "
                 f"{x[:,3].mean()-x[:,4].mean():+.3f} | {f(e_lo):+.3f} | {f(e_hi):+.3f} |")
    # condition level: does mean I predict the excess? compare with fertility from summary.csv
    import csv
    summ = {}
    for t in (btag, atag):
        p = f"{ROOT}/work/{t}/report/summary.csv"
        try:
            summ[t] = {(r["task"], r["lang"]): r for r in csv.DictReader(open(p, encoding="utf-8"))}
        except FileNotFoundError:
            summ[t] = {}
    cond = []
    for (task, lang) in sorted(set(zip(tasks, langs))):
        s = (tasks == task) & (langs == lang)
        if s.sum() < 300 or (task, lang) not in summ[btag] or (task, lang) not in summ[atag]:
            continue
        rb, ra = summ[btag][(task, lang)], summ[atag][(task, lang)]
        if int(rb["n_used"]) < 30:
            continue
        cond.append((task, lang, R[s, 1].mean(), R[s, 2].mean(), float(rb["fert"]),
                     math.log(float(ra["tau15"]) / float(rb["tau15"])), R[s, 3].mean() - R[s, 4].mean()))
    if len(cond) >= 8:
        C = np.array([c[2:] for c in cond])
        L += ["\n## 3. Condition level (task x language): what predicts the AR/block gap?\n",
              f"n = {len(C)} conditions. Outcome A = log(tau_AR / tau_block); outcome B = token-level excess block error.\n",
              "| predictor | Spearman with log(tau_AR/tau_block) | Spearman with excess error |", "|---|---|---|"]
        for j, nm in ((0, "mean I_k (k=2..8)"), (1, "mean conditional entropy Hc"), (2, "tokens per word")):
            L.append(f"| {nm} | {spearman(C[:, j], C[:, 3]):+.2f} | {spearman(C[:, j], C[:, 4]):+.2f} |")
        X = np.column_stack([np.ones(len(C)), C[:, 0], C[:, 1], np.log(C[:, 2])])
        for cols, nm in (([0, 1], "I"), ([0, 2], "Hc"), ([0, 3], "log fert"), ([0, 1, 2], "I + Hc"), ([0, 1, 2, 3], "I + Hc + log fert")):
            co, *_ = np.linalg.lstsq(X[:, cols], C[:, 3], rcond=None)
            pred = X[:, cols] @ co
            r2 = 1 - ((C[:, 3] - pred) ** 2).sum() / ((C[:, 3] - C[:, 3].mean()) ** 2).sum()
            L.append(f"\nOLS log(tau_AR/tau_block) ~ {nm}: R^2 = {r2:.2f}, coefficients {np.round(co, 3).tolist()}")
    out = "\n".join(L) + "\n"
    open(f"{ROOT}/work/mi_{mi_tag}/mi_report_{btag}_{atag}.md", "w").write(out)
    print(out)
