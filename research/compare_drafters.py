"""AR drafters vs block-parallel (DFlash) on the same Qwen3-8B trajectories: language-structure sensitivity.

usage: compare_drafters.py <results_dir> ; reads <results_dir>/<tag>/summary.csv, writes <results_dir>/drafter_comparison.md
Each summary row = one task x language condition (same trajectories for every drafter, so comparisons are paired).
"""
import csv
import math
import os
import sys

import numpy as np

R = sys.argv[1]
FAMILY = sys.argv[2] if len(sys.argv) > 2 else "qwen"
GROUPS_LLAMA = [
    ("Llama, no fine-tuning (public DFlash / off-the-shelf Llama-3.2-1B)", "none", {"DFlash": "ll8", "AR Llama-3.2-1B": "round3/arL1"}),
    ("Llama, 50K English (matched)", "en", {"DFlash": "round3/fLen", "AR Llama-3.2-1B": "round3/arL1_en"}),
    ("Llama, 50K multilingual (matched)", "ml", {"DFlash": "round3/fLml", "AR Llama-3.2-1B": "round3/arL1_ml"}),
]
GROUPS = [  # (label, data, {family: tag})
    ("no fine-tuning (public / off-the-shelf)", "none", {"DFlash": "base3", "AR Qwen3-0.6B": "ar06", "AR Qwen3-1.7B": "ar17", "AR Qwen3-4B": "ar4"}),
    ("50K English (matched)", "en", {"DFlash": "ften", "AR Qwen3-0.6B": "ar06_en", "AR Qwen3-1.7B": "ar17_en"}),
    ("50K multilingual (matched)", "ml", {"DFlash": "ftml", "AR Qwen3-0.6B": "ar06_ml", "AR Qwen3-1.7B": "ar17_ml"}),
]
GROUPS = GROUPS_LLAMA if FAMILY == "llama" else GROUPS
MIN_N = 30
rng = np.random.default_rng(0)


def load(tag):
    p = f"{R}/{tag}/summary.csv"
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if int(r["n_used"]) >= MIN_N and r["tau15"] not in ("nan", ""):
            out[(r["task"], r["lang"])] = r
    return out


def fit(rows, keys, col):
    X = np.array([[1.0, math.log(float(rows[k]["fert"])), float(rows[k]["H"])] for k in keys])
    y = np.array([math.log(float(rows[k][col])) for k in keys])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


lines = ["# AR drafters vs block-parallel drafting: sensitivity to language structure\n",
         "Same target (Qwen3-8B), same evaluation trajectories, same filtering; conditions with >= 30 usable responses.",
         "Model per drafter: log(tau) = b0 + b_fert * log(tokens/word) + b_H * target entropy, fitted over task x language conditions.",
         "b_fert < 0 means acceptance drops in languages whose words are split into more tokens. Difference test: paired",
         "bootstrap over conditions (same conditions for both drafters), 2000 resamples.\n"]
for col, kname in (("tau15", "tau@15"), ("tau7", "tau@7")):
    lines.append(f"## {kname}\n")
    for label, data, fam in GROUPS:
        tabs = {f: load(t) for f, t in fam.items()}
        tabs = {f: v for f, v in tabs.items() if v}
        if "DFlash" not in tabs or len(tabs) < 2:
            continue
        keys = sorted(set.intersection(*(set(v) for v in tabs.values())))
        keys = [k for k in keys if all(float(tabs[f][k][col]) > 0 for f in tabs)]
        lines.append(f"### {label} (n={len(keys)} conditions)\n")
        lines.append("| drafter | mean tau | b_fert [95% CI] | b_H | Spearman(tau, tokens/word) | tau(l)/tau(en) median, per task | b_fert minus DFlash [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|")
        base = tabs["DFlash"]
        idx = np.arange(len(keys))
        boots = {f: [] for f in tabs}
        for _ in range(2000):
            s = rng.choice(idx, len(idx))
            ks = [keys[i] for i in s]
            for f in tabs:
                try:
                    boots[f].append(fit(tabs[f], ks, col)[1])
                except Exception:
                    boots[f].append(np.nan)
        for f, tb in tabs.items():
            c = fit(tb, keys, col)
            b = np.array(boots[f])
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            taus = np.array([float(tb[k][col]) for k in keys])
            fert = np.array([float(tb[k]["fert"]) for k in keys])
            rels = []
            for task in ("math", "translate", "chat"):
                if (task, "en") in tb:
                    en = float(tb[(task, "en")][col])
                    rr = [float(tb[k][col]) / en for k in keys if k[0] == task and k[1] != "en"]
                    rels.append(f"{task} {np.median(rr):.2f}" if rr else f"{task} -")
            if f == "DFlash":
                diff = "—"
            else:
                d = b - np.array(boots["DFlash"])
                dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
                diff = f"{c[1] - fit(base, keys, col)[1]:+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
            lines.append(f"| {f} | {taus.mean():.2f} | {c[1]:+.2f} [{lo:+.2f}, {hi:+.2f}] | {c[2]:+.2f} | {spearman(taus, fert):+.2f} | "
                         f"{'; '.join(rels)} | {diff} |")
        lines.append("")
out = "\n".join(lines) + "\n"
open(f"{R}/drafter_comparison{'' if FAMILY == 'qwen' else '_' + FAMILY}.md", "w", encoding="utf-8").write(out)
print(out)
