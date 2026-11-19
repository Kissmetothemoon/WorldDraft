"""Summarise the same-anchor reveal intervention (I0 / reveal word start / reveal previous token / reveal own start).

usage: reveal_summary.py <tag> ; reads $LEXDRAFT_ROOT/work/<tag>/reveal_*.pkl
Per in-word token record: (k, j, m, c0, c_start, c_prev, c_self, start_ok_I0, prev_ok_I0, q_start, bpiece)
"""
import glob
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, lang_ok, pload

tag = sys.argv[1]
rng = np.random.default_rng(0)
by_lang = defaultdict(list)
n_seq = defaultdict(int)
for f in sorted(glob.glob(f"{ROOT}/work/{tag}/reveal_*.pkl")):
    for si, r in enumerate(pload(f)):
        out = r["output_ids"]
        g4 = [tuple(out[i:i + 4]) for i in range(max(0, len(out) - 3))]
        if not lang_ok(r["text_out"], r["lang"]) or (1 - len(set(g4)) / max(1, len(g4))) > 0.5:
            continue
        n_seq[r["lang"]] += 1
        sid = hash((f, si)) % (2 ** 31)
        for x in r["rec"]:
            by_lang[r["lang"]].append((sid,) + tuple(x))


def boot(a, fn, reps=300):
    seqs = a[:, 0]
    uniq, inv = np.unique(seqs, return_inverse=True)
    groups = [np.nonzero(inv == u)[0] for u in range(len(uniq))]
    est = fn(a)
    bs = []
    for _ in range(reps):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        bs.append(fn(a[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


# columns after sid: 1 k, 2 j, 3 m, 4 c0, 5 c_start, 6 c_prev, 7 c_self, 8 start_ok, 9 prev_ok, 10 q_start, 11 bpiece
eff = lambda a: ((1 - a[:, 4]) - (1 - a[:, 5])).mean() - ((1 - a[:, 4]) - (1 - a[:, 6])).mean()
lines = [f"# Reveal intervention ({tag})\n",
         "Anchor s = w-2 for a multi-token word starting at w; outcome = error of in-word tokens (teacher-forced, same offsets).",
         "gain_start = e(I0) - e(word start revealed); gain_prev = e(I0) - e(previous token revealed) (control: one revealed token, not the start).",
         "H1 effect = gain_start - gain_prev. The 'start already right' columns restrict to words whose start the drafter predicts",
         "correctly without help, so any remaining gain comes only from conditioning on the chosen word.\n",
         "| lang | seqs | in-word tokens | e(I0) | e(start revealed) | e(prev revealed) | H1 effect [95% CI] | H1 effect, start already right [95% CI] | e(own start revealed) |",
         "|---|---|---|---|---|---|---|---|---|"]
for l in sorted(by_lang):
    a = np.asarray(by_lang[l], dtype=np.float64)
    if len(a) < 50:
        continue
    e0, es, ep, ese = 1 - a[:, 4].mean(), 1 - a[:, 5].mean(), 1 - a[:, 6].mean(), 1 - a[:, 7].mean()
    h = boot(a, eff)
    ar = a[a[:, 8] == 1]
    hr = boot(ar, eff) if len(ar) >= 50 else (np.nan, np.nan, np.nan)
    lines.append(f"| {l} | {n_seq[l]} | {len(a)} | {e0:.3f} | {es:.3f} | {ep:.3f} | {h[0]:+.3f} [{h[1]:+.3f}, {h[2]:+.3f}] | "
                 f"{hr[0]:+.3f} [{hr[1]:+.3f}, {hr[2]:+.3f}] | {ese:.3f} |")
out = "\n".join(lines) + "\n"
open(f"{ROOT}/work/{tag}/reveal_report.md", "w").write(out)
print(out)
