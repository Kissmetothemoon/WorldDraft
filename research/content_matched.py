"""Content-matched view (translate task: every language renders the SAME 250 English source questions).

Per language: tokens per response, mean target entropy per token, total target entropy per response (sum over tokens),
and for each drafter tag: tau@15 from the offline chain, rounds per response (= tokens / tau), words per round.
usage: content_matched.py <tag> [<tag> ...]   (first tag also supplies the target-side quantities)
"""
import glob
import sys
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from common import ROOT, classify_tokens, lang_ok, pload, token_byte_table

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-8B")
bt = token_byte_table(tok)


def chain_tau(corr, K=15):
    lead = np.cumprod(corr == 1, 1).sum(1)
    a, accs = 0, []
    while a < len(lead):
        x = min(int(lead[a]), K) + 1
        accs.append(x)
        a += x
    return sum(accs), len(accs)


keep = None
res = {}
for tag in sys.argv[1:]:
    S = defaultdict(lambda: defaultdict(float))
    ok_ids = set()
    for f in sorted(glob.glob(f"{ROOT}/work/{tag}/tf_*.pkl")):
        for r in pload(f):
            if r["task"] != "translate":
                continue
            out = r["output_ids"]
            T = len(out)
            g4 = [tuple(out[i:i + 4]) for i in range(max(0, T - 3))]
            if T < 8 or not lang_ok(r["text_out"], r["lang"]) or 1 - len(set(g4)) / max(1, len(g4)) > 0.5:
                continue
            if keep is not None and r["id"] not in keep:
                continue
            ok_ids.add(r["id"])
            s = S[r["lang"]]
            ent = np.asarray(r["ent"], dtype=np.float64)[:T]
            c = classify_tokens(out, bt, r["lang"])
            s["n"] += 1
            s["tokens"] += T
            s["words"] += sum(1 for x in c["cls"] if x == 1)
            s["Hsum"] += float(ent.sum())
            acc, rounds = chain_tau(r["corr"])
            s["acc"] += acc
            s["rounds"] += rounds
    if keep is None:
        keep = ok_ids  # same responses for every drafter
    res[tag] = S

first = sys.argv[1]
print("translate task, same 250 English sources; responses kept identically for all drafters")
hdr = "lang     n  tokens/resp  words/resp  H/token  H/resp | " + " | ".join(f"{t}: tau  rounds/resp  words/round" for t in sys.argv[1:])
print(hdr)
for l in sorted(res[first], key=lambda l: res[first][l]["tokens"] / max(1, res[first][l]["n"])):
    s = res[first][l]
    n = s["n"]
    row = f"{l:8s} {int(n):3d} {s['tokens']/n:10.1f} {s['words']/n:10.1f} {s['Hsum']/s['tokens']:8.3f} {s['Hsum']/n:7.1f} | "
    parts = []
    for t in sys.argv[1:]:
        q = res[t][l]
        tau = q["acc"] / max(1, q["rounds"])
        parts.append(f"{tau:5.2f} {q['rounds']/max(1,q['n']):8.1f} {q['words']/max(1,q['rounds']):9.2f}")
    print(row + " | ".join(parts))
