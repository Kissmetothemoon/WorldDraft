"""Merge eval_unit.py shards: usage eval_unit_summary.py <out.md> name=<prefix> ... (prefix = path without _<shard>.pkl)"""
import glob, sys
from collections import defaultdict
import numpy as np
from common import pload
outp, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
T = {}
for name, pre in specs:
    S = defaultdict(lambda: defaultdict(float))
    for f in glob.glob(pre + "_*.pkl"):
        for k, v in pload(f).items():
            for a, x in v.items():
                S[k][a] += x
    T[name] = S
names = [n for n, _ in specs]
L = ["| lang | " + " | ".join(f"{n} tau" for n in names) + " | " + " | ".join(f"{n} vs {names[1]}" for n in names[2:])
     + " | " + " | ".join(f"{n}: draft len / invalid-UTF-8 rejections" for n in names) + " |",
     "|---|" + "---|" * (len(names) + len(names[2:]) + len(names))]
for l in LANGS:
    row, taus = [], {}
    for n in names:
        ks = [k for k in T[n] if k[1] == l]
        if not ks:
            break
        taus[n] = np.mean([T[n][k]["acc"] / T[n][k]["rounds"] for k in ks])
        fl = sum(T[n][k]["flen"] for k in ks) / sum(T[n][k]["n"] for k in ks)
        bad = sum(T[n][k]["bad"] for k in ks) / max(1, sum(T[n][k]["rej"] for k in ks))
        row.append(f"{fl:.1f} / {100*bad:.1f}%")
    if len(taus) < len(names):
        continue
    L.append(f"| {l} | " + " | ".join(f"{taus[n]:.2f}" for n in names) + " | "
             + " | ".join(f"{taus[n]/taus[names[1]]-1:+.1%}" for n in names[2:]) + " | " + " | ".join(row) + " |")
txt = "\n".join(L) + "\n"; open(outp, "w", encoding="utf-8").write(txt); print(txt)
