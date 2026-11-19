"""Engine cost decomposition from engine_profile.py outputs (same clean prompts for every config).

usage: prof_summary.py <dir with *.pkl>  -> <dir>/prof_report.md
speedup(method) = tokens/s(method) / tokens/s(AR), pooled over the language's tasks (token-weighted, from the un-timed pass).
Identity used for the decomposition (decode steps of the timed pass):
    speedup = tau x c_AR / c_SD,  c = mean wall time of one decode step at that concurrency
    c_SD = verify (target forward over 1+k tokens/request) + draft + sample + other
"""
import glob
import pickle
import sys
from collections import defaultdict

import numpy as np

D = sys.argv[1]
LANGS = ["en", "de", "zh", "ja", "ko", "th", "hi", "bn", "te", "sr-Cyrl"]
ORDER = ["ar", "draft", "dpub", "dpub_v3", "dml", "dml_v7", "dml_v3", "dml_dyn"]
LABEL = {"ar": "AR", "draft": "AR draft 0.6B-ml (k=5)", "dpub": "DFlash public (verify 15)", "dpub_v3": "DFlash public (verify 3)",
         "dml": "DFlash ml (verify 15)", "dml_v7": "DFlash ml (verify 7)", "dml_v3": "DFlash ml (verify 3)",
         "dml_dyn": "DFlash ml (verify 15/7/3 by batch)"}
R = {}
for f in glob.glob(f"{D}/*.pkl"):
    name = f.replace("\\", "/").split("/")[-1][:-4]
    R[name] = pickle.load(open(f, "rb"))
ks = {n: (R[n]["args"]["k"] if R[n]["args"]["method"] != "none" else 0) for n in R}


def agg(name, lang, conc):
    rs = [r for r in R[name]["res"] if r["lang"] == lang and r["conc"] == conc]
    if not rs:
        return None
    tok = sum(r["tokens"] for r in rs)
    tm = sum(r["time"] for r in rs)
    dr = sum(r["counters"].get("vllm:spec_decode_num_drafts", 0) for r in rs)
    acc = sum(r["counters"].get("vllm:spec_decode_num_accepted_tokens", 0) for r in rs)
    S = np.concatenate([r["prof"]["steps"] for r in rs if len(r["prof"]["steps"])])
    k = ks[name]
    dec = S[(S[:, 1] <= S[:, 0] * (k + 1)) & np.isfinite(S[:, 6]) & (S[:, 6] < 1.0)]
    return {"tps": tok / tm, "tau": 1 + acc / dr if dr else 1.0, "exec": dec[:, 3].mean(), "draft": dec[:, 5].mean(),
            "sample": dec[:, 4].mean(), "wall": dec[:, 6].mean(), "nreq": dec[:, 0].mean()}


L = ["# Engine cost decomposition (vLLM nightly, Qwen3-8B, one NVIDIA L20, clean prompts, greedy)\n",
     "speedup vs AR in the same engine; per-step times in ms (decode steps of the timed pass, cuda-synchronised).\n"]
for conc in (1, 8, 32):
    L.append(f"## concurrency {conc}: speedup (engine tau)\n")
    names = [n for n in ORDER if n in R]
    L.append("| lang | AR tok/s | " + " | ".join(LABEL[n] for n in names if n != "ar") + " |")
    L.append("|---|---|" + "---|" * (len(names) - 1))
    for l in LANGS:
        a = agg("ar", l, conc)
        if a is None:
            continue
        cells = []
        for n in names:
            if n == "ar":
                continue
            m = agg(n, l, conc)
            cells.append(f"{m['tps']/a['tps']:.2f}x ({m['tau']:.2f})" if m else "-")
        L.append(f"| {l} | {a['tps']:.0f} | " + " | ".join(cells) + " |")
    L.append("")
L.append("## Per-step cost (ms), averaged over languages: verify / draft / sample / other = wall; ratio to the AR step\n")
L.append("| method | conc | verify | draft | sample | other | step wall | c_SD / c_AR | mean tau | tau x c_AR/c_SD | measured speedup |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for n in [x for x in ORDER if x in R]:
    for conc in (1, 8, 32):
        rows = [(agg(n, l, conc), agg("ar", l, conc)) for l in LANGS]
        rows = [(m, a) for m, a in rows if m and a]
        if not rows:
            continue
        mm = {k: np.mean([m[k] for m, _ in rows]) for k in ("exec", "draft", "sample", "wall", "tau")}
        aw = np.mean([a["wall"] for _, a in rows])
        sp = np.mean([m["tps"] / a["tps"] for m, a in rows])
        other = mm["wall"] - mm["exec"] - mm["draft"] - mm["sample"]
        L.append(f"| {LABEL[n]} | {conc} | {1e3*mm['exec']:.1f} | {1e3*mm['draft']:.1f} | {1e3*mm['sample']:.1f} | {1e3*other:.1f} | "
                 f"{1e3*mm['wall']:.1f} | {mm['wall']/aw:.2f} | {mm['tau']:.2f} | {mm['tau']*aw/mm['wall']:.2f} | {sp:.2f} |")
L.append("")
# per-language break-even tau at conc 32 for each DFlash config: tau needed for speedup 1 = c_SD / c_AR
L.append("## Concurrency 32: per language, tau vs break-even tau (= c_SD / c_AR)\n")
L.append("| lang | " + " | ".join(f"{LABEL[n]}: tau / break-even" for n in ORDER if n in R and n != "ar") + " |")
L.append("|---|" + "---|" * (len([n for n in ORDER if n in R]) - 1))
for l in LANGS:
    a = agg("ar", l, 32)
    cells = []
    for n in ORDER:
        if n == "ar" or n not in R:
            continue
        m = agg(n, l, 32)
        cells.append(f"{m['tau']:.2f} / {m['wall']/a['wall']:.2f}" if m else "-")
    L.append(f"| {l} | " + " | ".join(cells) + " |")
txt = "\n".join(L) + "\n"
open(f"{D}/prof_report.md", "w", encoding="utf-8").write(txt)
print(txt)
