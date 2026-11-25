"""Quality of the engine outputs per config (lossless check): MGSM accuracy and translation chrF, conc=1 outputs.
usage: main_quality.py <main dir> <tokenizer> <out.md>"""
import glob, pickle, sys
import numpy as np
import pandas as pd
from common import LANGS, ROOT
from acc_eval import chrf, last_number
from transformers import AutoTokenizer
D, tokp, outp = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(tokp)
mg = {l: pd.read_parquet(f"{ROOT}/data/mgsm/{c}/test-00000-of-00001.parquet") for l, (_, _, _, c) in LANGS.items() if c}
NAMES = ["ar", "draft", "eagle3", "dpub", "dpub_lex", "dml", "dml_lex"]
Q = {}
for n in NAMES:
    acc, cf = {}, {}
    for f in glob.glob(f"{D}/{n}_[0-9].pkl"):
        for r in pickle.load(open(f, "rb"))["res"]:
            if r["conc"] != 1 or r["lang"] not in mg:
                continue
            for rid, o in zip(r["ids"], r["out_ids"]):
                i = int(rid.rsplit("-", 1)[1]); txt = tok.decode(o, skip_special_tokens=True)
                if r["task"] == "math":
                    p = last_number(txt); acc.setdefault(r["lang"], []).append(float(p is not None and abs(p - float(mg[r["lang"]]["answer_number"][i])) < 1e-6))
                elif r["task"] == "translate":
                    cf.setdefault(r["lang"], []).append(chrf(txt.strip(), mg[r["lang"]]["question"][i]))
    Q[n] = (acc, cf)
langs = [l for l in LANGS if l in mg]
L = ["| lang | " + " | ".join(f"{n} MGSM / chrF" for n in NAMES) + " |", "|---|" + "---|" * len(NAMES)]
for l in langs:
    L.append(f"| {l} | " + " | ".join(f"{100*np.mean(Q[n][0].get(l,[np.nan])):.1f} / {np.mean(Q[n][1].get(l,[np.nan])):.1f}" for n in NAMES) + " |")
L.append(f"| mean | " + " | ".join(f"{100*np.nanmean([np.mean(Q[n][0][l]) for l in langs if l in Q[n][0]]):.1f} / {np.nanmean([np.mean(Q[n][1][l]) for l in langs if l in Q[n][1]]):.1f}" for n in NAMES) + " |")
txt = "\n".join(L) + "\n"; open(outp, "w", encoding="utf-8").write(txt); print(txt)
