"""Raw (not entropy-standardised) per-token-type statistics for one tag's offline records.

Per language: share of in-word tokens, mean target entropy of word-start / in-word tokens,
reject@1 = P(drafter's one-step prediction of token t is wrong | anchor t-1)  [same quantity for AR and block drafters],
in-word reject when its word start was drafted in the same block = first in-word token at offset 2 from anchor t-2,
given the start (offset 1) was accepted.  For an AR drafter this equals reject@1 by construction; for a block drafter the
difference is the H1 cost at offset 2 (it also includes a one-step offset effect).
usage: token_type_stats.py <tag> [<tag> ...]
"""
import glob
import sys
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from common import ROOT, classify_tokens, lang_ok, pload, token_byte_table

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-8B")
bt = token_byte_table(tok)
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "hi", "bn", "te", "sr-Cyrl", "sr-Latn", "hi-Latn"]


def stats(tag):
    S = defaultdict(lambda: defaultdict(float))
    for f in sorted(glob.glob(f"{ROOT}/work/{tag}/tf_*.pkl")):
        for r in pload(f):
            out = r["output_ids"]
            T = len(out)
            if r["lang"] not in LANGS or T < 8 or not lang_ok(r["text_out"], r["lang"]):
                continue
            g4 = [tuple(out[i:i + 4]) for i in range(T - 3)]
            if 1 - len(set(g4)) / max(1, len(g4)) > 0.5:
                continue
            c = classify_tokens(out, bt, r["lang"])
            corr, ent = r["corr"], np.asarray(r["ent"], dtype=np.float32)
            s = S[r["lang"]]
            s["words"] += sum(1 for x in c["cls"] if x == 1)
            s["tokens"] += T
            for t in range(2, T - 1):
                cl = c["cls"][t]
                if cl == 0:
                    continue
                ty = "ws" if cl == 1 else "in"
                s["n_" + ty] += 1
                s["H_" + ty] += float(ent[t])
                v = corr[t - 1, 0]
                if v != 2:
                    s["e1n_" + ty] += 1
                    s["e1_" + ty] += int(v == 0)
                if ty == "in" and c["j"][t] == 1:
                    a = t - 2
                    if corr[a, 0] == 1 and corr[a, 1] != 2:
                        s["e2n"] += 1
                        s["e2"] += int(corr[a, 1] == 0)
    return S


for tag in sys.argv[1:]:
    S = stats(tag)
    print(f"== {tag}")
    print("lang     tok/word in-word%  H(ws)  H(in)  rej@1(ws) rej@1(in) rej(in|start drafted in block)")
    for l in LANGS:
        s = S[l]
        n = s["n_ws"] + s["n_in"]
        if not n:
            continue
        print(f"{l:8s} {s['tokens']/max(1,s['words']):7.2f} {100*s['n_in']/n:7.1f} {s['H_ws']/max(1,s['n_ws']):6.2f} "
              f"{s['H_in']/max(1,s['n_in']):6.2f}   {s['e1_ws']/max(1,s['e1n_ws']):.3f}     {s['e1_in']/max(1,s['e1n_in']):.3f}     "
              f"{s['e2']/max(1,s['e2n']):.3f} (n={int(s['e2n'])})")
