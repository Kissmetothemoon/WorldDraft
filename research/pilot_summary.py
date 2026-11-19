"""Merge variant tables from pilot_lex / pilot_chunk style pkls ({'task|lang': {variant: [acc, rounds]}}).
usage: pilot_summary.py <out.md> <glob of pkls> [baseline variant name]"""
import glob, sys
from collections import defaultdict
import numpy as np
from common import pload
outp, pat = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "top1"
S = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for f in glob.glob(pat):
    for k, d in pload(f).items():
        for n, v in d.items():
            S[k][n][0] += v[0]; S[k][n][1] += v[1]
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
names = list(next(iter(S.values())).keys())
L = ["| lang | " + " | ".join(names) + " | best vs " + base + " |", "|---|" + "---|" * (len(names) + 1)]
for l in LANGS:
    ks = [k for k in S if k.split("|")[1] == l]
    if not ks:
        continue
    t = {n: np.mean([S[k][n][0] / S[k][n][1] for k in ks]) for n in names}
    best = max((n for n in names if n != base), key=t.get)
    L.append(f"| {l} | " + " | ".join(f"{t[n]:.3f}" for n in names) + f" | {t[best]/t[base]-1:+.1%} ({best}) |")
txt = "\n".join(L) + "\n"; open(outp, "w", encoding="utf-8").write(txt); print(txt)
