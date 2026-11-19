"""Per-condition acceptance length from real spec_generate runs (e.g. temperature-1 sampling), filtered like analyze.py."""
import glob
import sys
from collections import defaultdict

from common import ROOT, pload

tag, prefix = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "speed")
g = defaultdict(list)
for f in sorted(glob.glob(f"{ROOT}/work/{tag}/{prefix}_*.pkl")):
    for x in pload(f):
        g[(x["task"], x["lang"])].append(x)
lines = [f"# Real spec_generate acceptance length ({tag}/{prefix})\n", "| task | lang | n | tau |", "|---|---|---|---|"]
for k in sorted(g):
    xs = g[k]
    tau = sum(sum(x["accs"]) for x in xs) / max(1, sum(len(x["accs"]) for x in xs))
    lines.append(f"| {k[0]} | {k[1]} | {len(xs)} | {tau:.2f} |")
out = "\n".join(lines) + "\n"
open(f"{ROOT}/work/{tag}/{prefix}_tau_report.md", "w").write(out)
print(out)
