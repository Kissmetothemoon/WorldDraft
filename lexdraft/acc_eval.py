"""Task quality of the target's greedy outputs per language (identical for lossless speculative decoding).

math      : MGSM exact match of the last number in the response (native digits normalised)
translate : chrF (n=6, beta=2) against the MGSM question in the target language (MGSM languages only)
usage: acc_eval.py <gen glob> <out.md>
"""
import glob
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

from common import LANGS, ROOT, jload, lang_ok

mg = {}
for lang, (_, _, _, code) in LANGS.items():
    if code:
        df = pd.read_parquet(f"{ROOT}/data/mgsm/{code}/test-00000-of-00001.parquet")
        mg[lang] = (df["question"].tolist(), df["answer_number"].tolist())


def norm_digits(s):
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() and unicodedata.digit(c, None) is not None else c for c in s)


def last_number(s):
    s = norm_digits(s).replace(",", "").replace("٬", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def chrf(hyp, ref, n=6, beta=2.0):
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    ps, rs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        ov = sum((h & r).values())
        ps.append(ov / max(1, sum(h.values())))
        rs.append(ov / max(1, sum(r.values())))
    p, r = sum(ps) / n, sum(rs) / n
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


if __name__ == "__main__":
    pat, outp = sys.argv[1], sys.argv[2]
    S = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(pat)):
        for r in jload(f):
            task, lang = r["task"], r["lang"]
            i = int(r["id"].rsplit("-", 1)[1])
            txt = r["text_out"]
            if task == "math" and lang in mg:
                ans = mg[lang][1][i]
                pred = last_number(txt)
                S[lang]["math"].append(float(pred is not None and abs(pred - float(ans)) < 1e-6))
            elif task == "translate" and lang in mg:
                S[lang]["chrf"].append(chrf(txt.strip(), mg[lang][0][i]))
            S[lang][f"lang_ok_{task}"].append(float(lang_ok(txt, lang)))
    L = ["| lang | MGSM accuracy (n) | translation chrF vs MGSM reference (n) | language-check pass rate math / translate / chat |",
         "|---|---|---|---|"]
    for lang in LANGS:
        d = S.get(lang)
        if not d:
            continue
        acc = f"{100*sum(d['math'])/len(d['math']):.1f} ({len(d['math'])})" if d["math"] else "n/a (no MGSM)"
        cf = f"{sum(d['chrf'])/len(d['chrf']):.1f} ({len(d['chrf'])})" if d["chrf"] else "n/a (no MGSM)"
        lo = " / ".join(f"{100*sum(d[f'lang_ok_{t}'])/len(d[f'lang_ok_{t}']):.0f}%" if d[f"lang_ok_{t}"] else "n/a" for t in ("math", "translate", "chat"))
        L.append(f"| {lang} | {acc} | {cf} | {lo} |")
    txt = "\n".join(L) + "\n"
    open(outp, "w", encoding="utf-8").write(txt)
    print(txt)
