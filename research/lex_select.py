"""Dev-selected lexical-prior settings, reported on the held-out half (pilot_lex.py keys 'task|lang|split').

usage: lex_select.py <out.md> <glob of pkls> [family prefix ('' = 'lam', 'xw ', 'sel+')]
per language: choose among {top1 (= prior off), <family>lam*+exp} on split 0; report split 1 for top1, the fixed lam0.5+exp,
the dev-selected per-language setting, and the dev-selected global lam (same for all languages).
"""
import glob
import sys
from collections import defaultdict

import numpy as np

from common import pload

outp, pat = sys.argv[1], sys.argv[2]
fam = sys.argv[3] if len(sys.argv) > 3 else ""
S = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for f in glob.glob(pat):
    for k, d in pload(f).items():
        for n, v in d.items():
            S[k][n][0] += v[0]
            S[k][n][1] += v[1]
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
names = list(next(iter(S.values())).keys())
cands = [n for n in names if n.startswith(fam + "lam") and n.endswith("+exp")] if fam != "sel+" else [n for n in names if n.startswith("sel+")]
base = "sel" if fam == "sel+" else "top1"
cands = [base] + cands
fixed = (fam + "lam0.5+exp") if (fam + "lam0.5+exp") in names else cands[-1]


def tau(lang, split, n):
    ks = [k for k in S if k.split("|")[1] == lang and k.split("|")[2] == str(split)]
    return np.mean([S[k][n][0] / S[k][n][1] for k in ks]) if ks else np.nan


# global lam: maximise the mean relative dev gain over languages
glob_best = max(cands[1:], key=lambda n: np.nanmean([tau(l, 0, n) / tau(l, 0, base) for l in LANGS]))
L = [f"# dev-selected lexical prior (family '{fam or 'lam'}'), test = held-out half\n",
     f"global setting chosen on dev: {glob_best}\n",
     f"| lang | test {base} | test fixed {fixed} | test global {glob_best} | test per-language (dev choice) | dev choice |",
     "|---|---|---|---|---|---|"]
for l in LANGS:
    if np.isnan(tau(l, 1, base)):
        continue
    b = max(cands, key=lambda n: tau(l, 0, n))
    t0 = tau(l, 1, base)
    L.append(f"| {l} | {t0:.3f} | {tau(l,1,fixed):.3f} ({tau(l,1,fixed)/t0-1:+.1%}) | {tau(l,1,glob_best):.3f} ({tau(l,1,glob_best)/t0-1:+.1%}) | "
             f"{tau(l,1,b):.3f} ({tau(l,1,b)/t0-1:+.1%}) | {b} |")
txt = "\n".join(L) + "\n"
open(outp, "w", encoding="utf-8").write(txt)
print(txt)
