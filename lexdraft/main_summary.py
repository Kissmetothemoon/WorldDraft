"""Paper main table from engine_lex.py outputs: speedup vs AR (same engine, pooled over tasks, token-weighted), engine tau,
and greedy losslessness (identical outputs, LexDraft vs the same drafter without it).
usage: main_summary.py <dir> <out.md>"""
import glob, pickle, sys
from collections import defaultdict
import numpy as np
D, outp = sys.argv[1], sys.argv[2]
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
NAMES = ["ar", "draft", "eagle3", "dpub", "dpub_lex", "dml", "dml_lex"]
LAB = {"draft": "AR draft 0.6B-ml", "eagle3": "EAGLE-3", "dpub": "DFlash public", "dpub_lex": "DFlash public + LexDraft",
       "dml": "DFlash ml", "dml_lex": "DFlash ml + LexDraft"}
R = defaultdict(dict)
for n in NAMES:
    for f in glob.glob(f"{D}/{n}_[0-9].pkl"):
        for r in pickle.load(open(f, "rb"))["res"]:
            R[n][(r["task"], r["lang"], r["conc"])] = r
def agg(n, l, c):
    ks = [k for k in R[n] if k[1] == l and k[2] == c]
    if not ks: return None
    tok = sum(R[n][k]["tokens"] for k in ks); tm = sum(R[n][k]["time"] for k in ks)
    taus = [R[n][k]["tau"] for k in ks if np.isfinite(R[n][k]["tau"])]
    return tok / tm, (np.mean(taus) if taus else np.nan)
L = ["# Main table: vLLM nightly, Qwen3-8B, one NVIDIA L20, clean prompts, greedy; speedup vs AR in the same engine (tau)\n"]
for c in (1, 8, 32):
    L += [f"## concurrency {c}\n", "| lang | AR tok/s | " + " | ".join(LAB[n] for n in NAMES[1:]) + " | LexDraft gain (pub / ml) |",
          "|---|---|" + "---|" * (len(NAMES) - 1) + "---|"]
    for l in LANGS:
        a = agg("ar", l, c)
        if a is None: continue
        cells, sp = [], {}
        for n in NAMES[1:]:
            m = agg(n, l, c)
            sp[n] = m[0] / a[0] if m else np.nan
            cells.append(f"{m[0]/a[0]:.2f}x ({m[1]:.2f})" if m else "-")
        L.append(f"| {l} | {a[0]:.0f} | " + " | ".join(cells) + f" | {sp['dpub_lex']/sp['dpub']-1:+.1%} / {sp['dml_lex']/sp['dml']-1:+.1%} |")
    L.append("")
L += ["## Greedy losslessness: share of requests whose output is identical with and without LexDraft (same drafter)\n",
      "| lang | public c=1 | public c=8 | public c=32 | ml c=1 | ml c=8 | ml c=32 | AR vs DFlash-ml+Lex c=1 |", "|---|---|---|---|---|---|---|---|"]
def ident(a, b, l, c):
    ks = [k for k in R[a] if k[1] == l and k[2] == c and k in R[b]]
    x = [p == q for k in ks for p, q in zip(R[a][k]["out_ids"], R[b][k]["out_ids"])]
    return f"{100*np.mean(x):.0f}%" if x else "-"
for l in LANGS:
    L.append(f"| {l} | " + " | ".join(ident(a, b, l, c) for a, b in (("dpub", "dpub_lex"), ("dml", "dml_lex")) for c in (1, 8, 32))
             + f" | {ident('ar', 'dml_lex', l, 1)} |")
txt = "\n".join(L) + "\n"; open(outp, "w", encoding="utf-8").write(txt); print(txt)
