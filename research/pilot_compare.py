"""Compare drafters on EXACTLY the same evaluation trajectories (ids of the first PARD record set).
usage: pilot_compare.py <out.md> tag1 tag2 ...   (tags = work/<tag>/tf_*.pkl in tf-record format)
tau@7 and tau@15 (accepted length incl. bonus, greedy chain), mean over tasks, per language.
"""
import glob
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, lang_ok, pload

outp, tags = sys.argv[1], sys.argv[2:]
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/work/{tag}/tf_*.pkl")):
        for r in pload(f):
            out[r["id"]] = r
    return out


R = {t: load(t) for t in tags}
ids = set(R[tags[0]])
for t in tags[1:]:
    ids &= set(R[t])


def chain(corr, Kmax):
    lead = np.minimum(np.cumprod((corr == 1).astype(np.int32), 1).sum(1), Kmax)
    a, accs = 0, []
    while a < corr.shape[0]:
        acc = int(lead[a]) + 1
        accs.append(acc)
        a += acc
    return accs


S = defaultdict(lambda: [0, 0])
for sid in ids:
    r0 = R[tags[0]][sid]
    out = r0["output_ids"]
    g4 = [tuple(out[i:i + 4]) for i in range(max(0, len(out) - 3))]
    if not lang_ok(r0["text_out"], r0["lang"]) or (1 - len(set(g4)) / max(1, len(g4))) > 0.5:
        continue
    for t in tags:
        c = R[t][sid]["corr"]
        for Km in (7, 15):
            if c.shape[1] < Km and Km == 15 and c.shape[1] < 15:
                pass
            a = chain(c, Km)
            k = (t, Km, r0["task"], r0["lang"])
            S[k][0] += sum(a)
            S[k][1] += len(a)
L = [f"# Drafters on identical trajectories ({len(ids)} ids before filtering)\n", "tau@7 / tau@15 (tau@15 is capped by the block length: K=8 drafters give at most 9)\n",
     "| lang | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
for l in LANGS:
    cells = []
    for t in tags:
        v = []
        for Km in (7, 15):
            ks = [k for k in S if k[0] == t and k[1] == Km and k[3] == l]
            v.append(np.mean([S[k][0] / S[k][1] for k in ks]) if ks else np.nan)
        cells.append(f"{v[0]:.2f} / {v[1]:.2f}")
    if any("nan" not in c for c in cells):
        L.append(f"| {l} | " + " | ".join(cells) + " |")
txt = "\n".join(L) + "\n"
open(outp, "w", encoding="utf-8").write(txt)
print(txt)
