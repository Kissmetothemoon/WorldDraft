"""Summaries for round 3: engine speedups (same-engine AR baseline), HF-level speedups, H0 per language (Llama)."""
import csv
import glob
import pickle
import sys
from collections import defaultdict

import numpy as np

R = sys.argv[1]  # .../<results root>/round3
LANG_ORDER = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


out = []
# ---------------- engine (vLLM 0.19.1): tokens/s per (task, lang, conc), speedup vs AR in the same engine
for eng, sub in (("vLLM 0.19.1", "engine_q8"), ("vLLM nightly cu129", "engineN_q8")):
    files = glob.glob(f"{R}/pkl/{sub}/*.pkl")
    if not files:
        continue
    tps = defaultdict(dict)  # method -> (task, lang, conc) -> (tps, tau, tokens)
    for f in files:
        name = f.replace("\\", "/").split("/")[-1].rsplit("_", 1)[0]
        for r in load(f)["res"]:
            tps[name][(r["task"], r["lang"], r["conc"])] = (r["tps"], r["tau"], r["tokens"])
    if "ar" not in tps:
        continue
    methods = [m for m in tps if m != "ar"]
    out.append(f"## Engine speedups ({eng}), speedup = tokens/s(method) / tokens/s(AR, same engine), pooled over tasks\n")
    for conc in (1, 8, 32):
        out.append(f"### concurrency {conc}\n")
        out.append("| lang | AR tok/s | " + " | ".join(f"{m} speedup (tau)" for m in methods) + " |")
        out.append("|---|---|" + "---|" * len(methods))
        for l in LANG_ORDER:
            keys = [k for k in tps["ar"] if k[1] == l and k[2] == conc]
            if not keys:
                continue
            # token-weighted: total tokens / total time per method across the language's tasks
            def tot(m):
                ks = [k for k in keys if k in tps[m]]
                tk = sum(tps[m][k][2] for k in ks)
                tm = sum(tps[m][k][2] / tps[m][k][0] for k in ks)
                taus = [tps[m][k][1] for k in ks if np.isfinite(tps[m][k][1])]
                return (tk / tm if tm else np.nan), (np.mean(taus) if taus else np.nan)
            ar, _ = tot("ar")
            cells = []
            for m in methods:
                v, t = tot(m)
                cells.append(f"{v/ar:.2f}x ({t:.2f})")
            out.append(f"| {l} | {ar:.0f} | " + " | ".join(cells) + " |")
        out.append("")

# ---------------- HF-level wall clock (batch 1): speed_bench outputs
for sub, fam in (("speedhf_q8", "Qwen3-8B"), ("speedhf_ll8", "Llama-3.1-8B")):
    files = glob.glob(f"{R}/pkl/{sub}/*.pkl")
    if not files:
        continue
    recs = sum((load(f) for f in files), [])
    meths = list(recs[0]["m"].keys())
    out.append(f"## HF-level batch-1 wall clock ({fam}); speedup vs HF generate on the same prompts\n")
    out.append("| lang | n | AR tok/s | " + " | ".join(f"{m} speedup (tau)" for m in meths) + " |")
    out.append("|---|---|---|" + "---|" * len(meths))
    by = defaultdict(list)
    for r in recs:
        by[r["lang"]].append(r)
    for l in LANG_ORDER:
        xs = by.get(l)
        if not xs:
            continue
        ar = sum(x["n_ar"] for x in xs) / sum(x["t_ar"] for x in xs)
        cells = []
        for m in meths:
            v = sum(x["m"][m]["n"] for x in xs) / sum(x["m"][m]["t"] for x in xs)
            tau = sum(sum(x["m"][m]["accs"]) for x in xs) / sum(len(x["m"][m]["accs"]) for x in xs)
            cells.append(f"{v/ar:.2f}x ({tau:.2f})")
        out.append(f"| {l} | {len(xs)} | {ar:.1f} | " + " | ".join(cells) + " |")
    out.append("")

# ---------------- H0 on Llama: mean tau@15 over tasks per language
def summ(path):
    try:
        return {(r["task"], r["lang"]): r for r in csv.DictReader(open(path, encoding="utf-8")) if int(r["n_used"]) >= 30}
    except FileNotFoundError:
        return {}


cols = [("DFlash public", f"{R}/../ll8/summary.csv"), ("DFlash 50K en", f"{R}/fLen/summary.csv"), ("DFlash 50K ml", f"{R}/fLml/summary.csv"),
        ("AR 1B off-shelf", f"{R}/arL1/summary.csv"), ("AR 1B 50K en", f"{R}/arL1_en/summary.csv"), ("AR 1B 50K ml", f"{R}/arL1_ml/summary.csv")]
T = {n: summ(p) for n, p in cols}
out.append("## Llama-3.1-8B: mean tau@15 over tasks per language (H0 and AR vs block)\n")
out.append("| lang | " + " | ".join(n for n, _ in cols) + " |")
out.append("|---|" + "---|" * len(cols))
for l in LANG_ORDER:
    cells = []
    for n, _ in cols:
        xs = [float(r["tau15"]) for (t, la), r in T[n].items() if la == l]
        cells.append(f"{np.mean(xs):.2f}" if xs else "-")
    out.append(f"| {l} | " + " | ".join(cells) + " |")
txt = "\n".join(out) + "\n"
open(f"{R}/round3_tables.md", "w", encoding="utf-8").write(txt)
print(txt)
