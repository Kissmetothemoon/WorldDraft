"""CPU ceiling for route D: per-position fusion of DFlash (target features) and a multilingually pre-trained parallel
drafter (PARD), on identical trajectories. Rules: pick the more confident drafter per position; oracle (either correct).
usage: pilot_fusion.py <out.md> <dflash_tag> <pard_tag>  (tf-record format, K = min of the two)"""
import glob, sys
from collections import defaultdict
import numpy as np
from common import ROOT, lang_ok, pload
outp, ta, tb = sys.argv[1:4]
def load(t):
    o = {}
    for f in glob.glob(f"{ROOT}/work/{t}/tf_*.pkl"):
        for r in pload(f):
            o[r["id"]] = r
    return o
A, B = load(ta), load(tb)
ids = [i for i in B if i in A]
def chain(c):
    lead = np.cumprod((c == 1).astype(np.int32), 1).sum(1); a, accs = 0, []
    while a < c.shape[0]:
        accs.append(int(lead[a]) + 1); a += accs[-1]
    return accs
S = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for i in ids:
    ra, rb = A[i], B[i]
    out = ra["output_ids"]; g4 = [tuple(out[j:j + 4]) for j in range(max(0, len(out) - 3))]
    if not lang_ok(ra["text_out"], ra["lang"]) or (1 - len(set(g4)) / max(1, len(g4))) > 0.5:
        continue
    K = min(ra["corr"].shape[1], rb["corr"].shape[1]); R = min(ra["corr"].shape[0], rb["corr"].shape[0])
    ca, cb = ra["corr"][:R, :K], rb["corr"][:R, :K]
    pa, pb = ra["dprob"][:R, :K].astype(np.float32), rb["dprob"][:R, :K].astype(np.float32)
    V = {"dflash": ca, "pard": cb, "conf-fusion": np.where(pa >= pb, ca, cb),
         "oracle": np.where((ca == 1) | (cb == 1), 1, np.where((ca == 2) | (cb == 2), 2, 0)).astype(np.uint8)}
    for n, c in V.items():
        a = chain(c); S[(ra["task"], ra["lang"])][n][0] += sum(a); S[(ra["task"], ra["lang"])][n][1] += len(a)
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]
names = ["dflash", "pard", "conf-fusion", "oracle"]
L = [f"# fusion ceiling: {ta} x {tb} (tau@{K}, identical trajectories)\n", "| lang | " + " | ".join(names) + " | fusion vs best single |", "|---|" + "---|" * 5]
for l in LANGS:
    ks = [k for k in S if k[1] == l]
    if not ks: continue
    t = {n: np.mean([S[k][n][0] / S[k][n][1] for k in ks]) for n in names}
    L.append(f"| {l} | " + " | ".join(f"{t[n]:.2f}" for n in names) + f" | {t['conf-fusion']/max(t['dflash'], t['pard'])-1:+.1%} |")
txt = "\n".join(L) + "\n"; open(outp, "w", encoding="utf-8").write(txt); print(txt)
