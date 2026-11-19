"""Summarise the official-protocol reproduction (GSM8K / MATH-500, max 2048 new tokens, greedy, thinking off)."""
import glob
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, pload

tag = sys.argv[1] if len(sys.argv) > 1 else "official_q8"
sp = sum((pload(f) for f in sorted(glob.glob(f"{ROOT}/work/{tag}/speed_*.pkl"))), [])
g = defaultdict(list)
for x in sp:
    g[x["task"]].append(x)
paper = {"gsm8k": (5.15, 6.54), "math500": (6.08, 7.87)}  # DFlash Table 1, Qwen3-8B, T=0, H200, Transformers backend
lines = [f"# Official-protocol reproduction ({tag})\n",
         "| benchmark | n | AR tok/s | DFlash tok/s | speedup | tau | paper speedup / tau (H200) | identical outputs | median first divergence |",
         "|---|---|---|---|---|---|---|---|---|"]
for k in sorted(g):
    xs = g[k]
    ar = sum(x["n_ar"] for x in xs) / sum(x["t_ar"] for x in xs)
    dfl = sum(x["n_sp"] for x in xs) / sum(x["t_sp"] for x in xs)
    tau = sum(sum(x["accs"]) for x in xs) / sum(len(x["accs"]) for x in xs)
    div = [x["same_prefix"] for x in xs if not x["identical"]]
    ps, pt = paper.get(k, (float("nan"), float("nan")))
    lines.append(f"| {k} | {len(xs)} | {ar:.1f} | {dfl:.1f} | {dfl/ar:.2f}x | {tau:.2f} | {ps:.2f}x / {pt:.2f} | "
                 f"{sum(x['identical'] for x in xs)}/{len(xs)} | {int(np.median(div)) if div else '-'} |")
out = "\n".join(lines) + "\n"
open(f"{ROOT}/work/{tag}/official_report.md", "w").write(out)
print(out)
