"""Aggregate stage-0/1 results into tables and a gate report.

usage: analyze.py --tag q8 --tokenizer <target path> [--eagle-model <path>]
reads  $WORLDDRAFT_ROOT/work/<tag>/{tf_*.pkl,cand_*.pkl,eagle_k7_*.pkl,speed_*.pkl,val.pkl}
writes $WORLDDRAFT_ROOT/work/<tag>/report/{report.md,*.csv,token_records.npz}
"""
import argparse
import glob
import math
import os
from collections import defaultdict

import numpy as np

from common import LANGS, ROOT, classify_tokens, lang_ok, pload, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--eagle-model", default=None)
ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args()

W = f"{ROOT}/work/{args.tag}"
OUT = f"{W}/report"
os.makedirs(OUT, exist_ok=True)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(args.tokenizer)
bt = token_byte_table(tok)
EB = np.array([0.1, 0.5, 1.0, 2.0, 4.0])  # target-entropy bin edges (nats)

tf_files = sorted(glob.glob(f"{W}/tf_*.pkl"))
cand = {}
for f in sorted(glob.glob(f"{W}/cand_*.pkl")):
    for m in pload(f):
        cand[(m["id"], m["k"])] = m


def word_measures(m):
    """per-offset suffix divergence D_j, majority conflict C_j, start uncertainty u, length divergence L."""
    pi = np.array(m["pi"], dtype=np.float64)
    pi = pi / pi.sum()
    conts = m["conts"]
    mstar = m["wlens"][0]
    D, C = {}, {}
    for j in range(1, max(1, mstar)):
        if any(c is None or len(c) < j for c in conts):
            break
        toks = [c[j - 1] for c in conts]
        D[j] = float(sum(p for p, t in zip(pi[1:], toks[1:]) if t != toks[0]))
        agg = defaultdict(float)
        for p, t in zip(pi, toks):
            agg[t] += p
        C[j] = int(max(agg, key=agg.get) != toks[0])
    L = float(sum(p for p, wl in zip(pi[1:], m["wlens"][1:]) if wl != mstar))
    return D, C, float(1 - pi[0]), L


def lead_run(corr):
    ok = (corr == 1).astype(np.int32)
    return np.cumprod(ok, axis=1).sum(1)  # number of leading correct positions


def entry(corr, lead, a, k):
    """risk-set outcome for anchor row a, offset k (1-based): 1 error, 0 correct, -1 not in risk set."""
    if a < 0 or a >= corr.shape[0] or k < 1 or k > corr.shape[1]:
        return -1
    if lead[a] < k - 1:
        return -1
    v = corr[a, k - 1]
    return -1 if v == 2 else int(v == 0)


def chain_tau(corr, lead, K):
    R = corr.shape[0]
    a, accs = 0, []
    while a < R:
        acc = min(int(lead[a]), K) + 1
        accs.append(acc)
        a += acc
    return accs


conds = sorted({(t, l) for t in ("math", "translate", "chat") for l in LANGS})
cidx = {c: i for i, c in enumerate(conds)}
KMAX = 15
haz = np.zeros((len(conds), 5, KMAX + 1, len(EB) + 1, 2), dtype=np.int64)  # type: 0 other 1 ws 2 inw-known 3 inw-drafted 4 ws-post-multi
summ = defaultdict(lambda: defaultdict(float))
tok_rec = defaultdict(list)
post_rec = defaultdict(list)
seq_ok = {}

for f in tf_files:
    for si, r in enumerate(pload(f)):
        c = (r["task"], r["lang"])
        ci = cidx[c]
        out = r["output_ids"]
        T = len(out)
        ok = lang_ok(r["text_out"], r["lang"])
        g4 = [tuple(out[i:i + 4]) for i in range(max(0, T - 3))]
        degen = (1 - len(set(g4)) / max(1, len(g4))) > 0.5  # looping output (repetition makes drafting trivially easy)
        S = summ[c]
        S["n_seq"] += 1
        S["n_lang_ok"] += ok
        S["n_degen"] += degen
        if not ok or degen or r["corr"] is None or T < 8:
            continue
        S["n_used"] += 1
        corr = r["corr"]
        K = corr.shape[1]
        lead = lead_run(corr)
        cl = classify_tokens(out, bt, r["lang"])
        cls, wst, jj, wl, bp = (np.array(cl[x]) for x in ("cls", "wstart", "j", "wlen", "bpiece"))
        ent = np.array(r["ent"], dtype=np.float32)[:T]
        ebin = np.digitize(ent, EB)
        nwords = int((cls == 1).sum())
        S["tokens"] += T
        S["words"] += nwords
        S["inword"] += int((cls == 2).sum())
        S["bpiece"] += int(bp.sum())
        S["ent_sum"] += float(ent.sum())
        S["ndev"] += r.get("ndev", 0)
        for kk, key in ((KMAX, "tau15"), (7, "tau7")):
            accs = chain_tau(corr, lead, kk)
            S[key + "_acc"] += sum(accs)
            S[key + "_rounds"] += len(accs)
        # hazard tables over all anchors
        post_multi = np.zeros(T, dtype=bool)
        for t in range(1, T):
            if cls[t] == 1 and cls[t - 1] in (1, 2) and wl[t - 1] >= 2:
                post_multi[t] = True
        for a in range(corr.shape[0]):
            for k in range(1, min(K, int(lead[a]) + 1) + 1):
                t = a + k
                if t >= T:
                    break
                v = corr[a, k - 1]
                if v == 2:
                    break
                if cls[t] == 2:
                    ty = 2 if wst[t] <= a else 3
                elif cls[t] == 1:
                    ty = 4 if post_multi[t] else 1
                else:
                    ty = 0
                haz[ci, ty, k, ebin[t], int(v == 0)] += 1
        # token-level records for H1 (in-word tokens)
        for t in np.nonzero(cls == 2)[0]:
            w, j = int(wst[t]), int(jj[t])
            ek = entry(corr, lead, w, j)
            ed = entry(corr, lead, w - 1, j + 1)
            ec = entry(corr, lead, w + 1, j - 1) if j >= 2 else -1
            if ek < 0 and ed < 0:
                continue
            m = cand.get((r["id"], w))
            Dj = Cj = u = Lw = np.nan
            if m is not None:
                D, C, u, Lw = word_measures(m)
                Dj, Cj = D.get(j, np.nan), C.get(j, np.nan)
            dpr = r["dprob"]
            ds = float(dpr[w - 1, 0]) if 0 <= w - 1 < dpr.shape[0] else np.nan  # drafter top-1 prob at the drafted start
            tok_rec[c].append((len(seq_ok), t, j, wl[t], bp[t], ent[t], ek, ed, ec, Dj, Cj, u, Lw, ds))
        # H2: word start following a multi-token word
        for t in np.nonzero(post_multi)[0]:
            w = int(wst[t - 1])
            mlen = t - w
            m = cand.get((r["id"], w))
            Lw = word_measures(m)[3] if m is not None else np.nan
            post_rec[c].append((len(seq_ok), t, mlen, ent[t], entry(corr, lead, w - 1, mlen + 1),
                                entry(corr, lead, w, mlen), entry(corr, lead, t - 1, 1), entry(corr, lead, t - 2, 2), Lw,
                                entry(corr, lead, w + 1, mlen - 1) if mlen >= 2 else -1))
        seq_ok[(r["id"])] = True

cols = ["seq", "t", "j", "wlen", "bpiece", "ent", "e_known", "e_drafted", "e_ctrl", "D", "C", "u", "L", "dstart"]
pcols = ["seq", "t", "mlen", "ent", "e_prevdrafted", "e_startknown", "e_k1", "e_k2", "L", "e_w1"]
np.savez_compressed(f"{OUT}/token_records.npz",
                    **{f"inw|{t}|{l}": np.array(v, dtype=np.float64) for (t, l), v in tok_rec.items()},
                    **{f"post|{t}|{l}": np.array(v, dtype=np.float64) for (t, l), v in post_rec.items()},
                    inw_cols=np.array(cols), post_cols=np.array(pcols), haz=haz,
                    conds=np.array(["%s|%s" % c for c in conds]))

# ------------------------------------------------------------------ EAGLE-3 reference + draft vocab coverage
eagle = {}
for f in glob.glob(f"{W}/eagle_k7_*.pkl"):
    for key, v in pload(f).items():
        eagle[tuple(key.split(":"))] = v["tau"]
cover = {}
if args.eagle_model:
    try:
        from safetensors.torch import load_file
        sd = load_file(glob.glob(f"{args.eagle_model}/*.safetensors")[0])
        t2d = sd.get("t2d")
        if t2d is not None:
            inv = t2d.bool().numpy()
            for f in tf_files:
                for r in pload(f):
                    o = np.array(r["output_ids"])
                    o = o[o < len(inv)]
                    cv = summ[(r["task"], r["lang"])]
                    cv["cov_hit"] += float(inv[o].sum())
                    cv["cov_n"] += len(o)
    except Exception as e:  # noqa
        print("eagle vocab coverage failed:", e)


def ratio(a, b):
    return a / b if b else float("nan")


lines = []
P = lines.append
P(f"# Stage 0/1 report: {args.tag}\n")
P("## 1. Per-condition summary\n")
P("| task | lang | n / lang-ok / looping / used | tokens/word | in-word % | byte-piece % | mean H | DFlash tau@15 | DFlash tau@7 | EAGLE-3 tau@7 | Delta_par@7 | EAGLE vocab cov |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows_csv = []
for c in conds:
    S = summ[c]
    if not S["n_seq"]:
        continue
    t15, t7 = ratio(S["tau15_acc"], S["tau15_rounds"]), ratio(S["tau7_acc"], S["tau7_rounds"])
    te = eagle.get(c, float("nan"))
    en = summ[(c[0], "en")]
    t7en = ratio(en["tau7_acc"], en["tau7_rounds"])
    teen = eagle.get((c[0], "en"), float("nan"))
    dpar = 1 - (t7 / t7en) / (te / teen) if all(map(math.isfinite, [t7, t7en, te, teen])) else float("nan")
    cov = ratio(S["cov_hit"], S["cov_n"])
    row = dict(task=c[0], lang=c[1], n=int(S["n_seq"]), n_ok=int(S["n_lang_ok"]), n_degen=int(S["n_degen"]),
               n_used=int(S["n_used"]), fert=ratio(S["tokens"], S["words"]),
               inword=100 * ratio(S["inword"], S["tokens"]), bpiece=100 * ratio(S["bpiece"], S["tokens"]),
               H=ratio(S["ent_sum"], S["tokens"]), tau15=t15, tau7=t7, eagle7=te, dpar7=dpar, eagle_cov=cov)
    rows_csv.append(row)
    P(f"| {c[0]} | {c[1]} | {row['n']} / {row['n_ok']} / {row['n_degen']} / {row['n_used']} | {row['fert']:.2f} | {row['inword']:.1f} | {row['bpiece']:.1f} | {row['H']:.2f} | {t15:.2f} | {t7:.2f} | {te:.2f} | {dpar:+.2f} | {cov:.2f} |")
import csv

with open(f"{OUT}/summary.csv", "w", newline="", encoding="utf-8") as fh:
    if rows_csv:
        wr = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        wr.writeheader()
        wr.writerows(rows_csv)
P("\nAll statistics exclude responses in the wrong language/script and looping responses (4-gram repetition > 0.5).")
P("Delta_par compares DFlash and EAGLE-3 public checkpoints (different data/architecture/draft vocab): descriptive only.\n")

# ------------------------------------------------------------------ hazards
P("## 2. Rejection hazard by token type (all tasks pooled per language, offsets 1-15, entropy-standardised)\n")
P("Hazard = P(rejected at this position | all earlier block positions accepted). Entropy-standardised = weighted by the")
P("pooled entropy-bin distribution of in-word tokens, so types/languages are compared at equal target uncertainty.\n")
P("| lang | word-start | in-word (start known) | in-word (start drafted) | ws after multi-token word | ratio drafted/known |")
P("|---|---|---|---|---|---|")
langs_present = sorted({c[1] for c in conds if summ[c]["n_seq"]})
ref = haz[:, 2:4, :, :, :].sum((0, 1, 2, 4)).astype(float)
ref = ref / ref.sum()
for l in langs_present:
    ci = [cidx[c] for c in conds if c[1] == l]
    H = haz[ci].sum(0)  # (type, k, ebin, 2)
    vals = []
    for ty in (1, 2, 3, 4):
        e = H[ty].sum(0)  # (ebin, 2)
        rate = np.where(e.sum(1) > 0, e[:, 1] / np.maximum(e.sum(1), 1), np.nan)
        wsum = np.nansum(rate * ref) / np.sum(ref[~np.isnan(rate)]) if np.any(~np.isnan(rate)) else np.nan
        vals.append(wsum)
    P(f"| {l} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} | {ratio(vals[2], vals[1]):.2f} |")
P("\nNote: 'start drafted' tokens sit at larger block offsets than 'start known' ones; the DiD in section 3 removes that.\n")

# ------------------------------------------------------------------ H1 DiD
P("## 3. H1 test: does seeing the word start reduce in-word errors? (difference-in-differences, same tokens)\n")
P("A = e(t | anchor w-1, start drafted & accepted) - e(t | anchor w, start known)   [start status + one offset step]")
P("B = e(t | anchor w, start known) - e(t | anchor w+1, start known)                 [one offset step only]")
P("H1 effect = A - B (paired on tokens with j>=2 where all three outcomes exist). 95% CI: bootstrap over sequences.\n")
rng = np.random.default_rng(0)


def did(rec, mask=None):
    if len(rec) == 0:
        return np.nan, np.nan, np.nan, 0
    a = np.asarray(rec)
    if mask is not None:
        a = a[mask]
    sel = (a[:, 6] >= 0) & (a[:, 7] >= 0) & (a[:, 8] >= 0)
    a = a[sel]
    if len(a) < 30:
        return np.nan, np.nan, np.nan, len(a)
    val = (a[:, 7] - a[:, 6]) - (a[:, 6] - a[:, 8])
    seqs = a[:, 0].astype(int)
    uniq, inv = np.unique(seqs, return_inverse=True)
    s_sum = np.bincount(inv, weights=val)
    s_n = np.bincount(inv)
    est = s_sum.sum() / s_n.sum()
    bs = []
    for _ in range(args.boot):
        idx = rng.integers(0, len(uniq), len(uniq))
        bs.append(s_sum[idx].sum() / s_n[idx].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, len(a)


P("| lang | tokens/word | n tokens | e_known | e_drafted | H1 effect A-B [95% CI] | relative (A-B)/e_known | H1, drafter start conf<0.6 | H1, drafter start conf>=0.9 |")
P("|---|---|---|---|---|---|---|---|---|")
h1_rows = []
fert_l = {}
for l in langs_present:
    tk = sum(summ[c]["tokens"] for c in conds if c[1] == l)
    wd = sum(summ[c]["words"] for c in conds if c[1] == l)
    fert_l[l] = ratio(tk, wd)
for l in langs_present:
    rec = sum((tok_rec[c] for c in conds if c[1] == l), [])
    if not rec:
        continue
    a = np.asarray(rec)
    full = (a[:, 6] >= 0) & (a[:, 7] >= 0) & (a[:, 8] >= 0)
    est, lo, hi, n = did(rec)
    lowc = did(rec, ~np.isnan(a[:, 13]) & (a[:, 13] < 0.6))
    highc = did(rec, ~np.isnan(a[:, 13]) & (a[:, 13] >= 0.9))
    ek = a[full, 6].mean() if full.any() else np.nan
    ed = a[full, 7].mean() if full.any() else np.nan
    h1_rows.append((l, fert_l[l], est, lo, hi, ek))
    P(f"| {l} | {fert_l[l]:.2f} | {n} | {ek:.3f} | {ed:.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] | {est / ek if ek else float('nan'):+.2f} | "
      f"{lowc[0]:+.3f} (n={lowc[3]}) | {highc[0]:+.3f} (n={highc[3]}) |")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan"), int(m.sum())
    rx, ry = np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


if h1_rows:
    rho, nn = spearman([r[1] for r in h1_rows], [r[2] for r in h1_rows])
    rho2, _ = spearman([r[1] for r in h1_rows], [r[2] / r[5] if r[5] else np.nan for r in h1_rows])
    P(f"\nAcross languages: Spearman(tokens/word, H1 effect) = {rho:+.2f}; with relative effect = {rho2:+.2f} (n={nn}).\n")

# ------------------------------------------------------------------ within-language suffix-conflict association (gate 1)
P("## 4. Gate-1 check: within-language association of in-word error with conflict measures\n")
P("Population: in-word tokens whose word start was drafted in the same block and accepted (H1 risk set).")
P("Mantel-Haenszel risk ratio, strata = target-entropy bin x within-word index (1,2,3+). Each comparison needs n>=100 per group.")
P("Target-side conflict: D = target probability mass of top-4 alternative word starts whose j-th continuation differs (high D >= 0.1 vs low D < 0.02).")
P("Drafter-side conflict: drafter top-1 probability at the drafted word start (low < 0.6 vs high >= 0.9).\n")


def mh_ratio(a, expo, ctrl):
    eb = np.digitize(a[:, 5], EB)
    jb = np.minimum(a[:, 2], 3)
    key = eb * 10 + jb
    num = den = 0.0
    for b in np.unique(key):
        s_ = key == b
        x1, x0 = a[s_ & expo], a[s_ & ctrl]
        if len(x1) and len(x0):
            w = len(x1) * len(x0) / (len(x1) + len(x0))
            num += w * x1[:, 7].mean()
            den += w * x0[:, 7].mean()
    return num / den if den > 0 else np.nan


P("| lang | n hiD / loD | err hiD | err loD | MH ratio (D) | n lowConf / highConf | err lowConf | err highConf | MH ratio (drafter conf) |")
P("|---|---|---|---|---|---|---|---|---|")
gate1_D = gate1_conf = 0
for l in langs_present:
    rec = sum((tok_rec[c] for c in conds if c[1] == l), [])
    if not rec:
        continue
    a = np.asarray(rec)
    a = a[a[:, 7] >= 0]
    hiD = ~np.isnan(a[:, 9]) & (a[:, 9] >= 0.1)
    loD = ~np.isnan(a[:, 9]) & (a[:, 9] < 0.02)
    loC = ~np.isnan(a[:, 13]) & (a[:, 13] < 0.6)
    hiC = ~np.isnan(a[:, 13]) & (a[:, 13] >= 0.9)
    r_d = mh_ratio(a, hiD, loD) if hiD.sum() >= 100 and loD.sum() >= 100 else np.nan
    r_c = mh_ratio(a, loC, hiC) if loC.sum() >= 100 and hiC.sum() >= 100 else np.nan
    gate1_D += int(np.isfinite(r_d) and r_d >= 1.5)
    gate1_conf += int(np.isfinite(r_c) and r_c >= 1.5)
    m = lambda x: a[x, 7].mean() if x.any() else np.nan
    P(f"| {l} | {hiD.sum()} / {loD.sum()} | {m(hiD):.3f} | {m(loD):.3f} | {r_d:.2f} | {loC.sum()} / {hiC.sum()} | {m(loC):.3f} | {m(hiC):.3f} | {r_c:.2f} |")
P(f"\n**Gate 1 (H1 route): {gate1_D} languages with target-side MH ratio >= 1.5; {gate1_conf} languages with drafter-side MH ratio >= 1.5 (need >= 2).**")
P("Caveat: these are associations inside DFlash, not interventions; the DiD in section 3 is the within-model contrast.\n")

# ------------------------------------------------------------------ H2
P("## 5. H2 check: next word start after a multi-token word (start w, length m)\n")
P("A2 = e(u | anchor w-1: previous word incl. its start drafted) - e(u | anchor w: start known, rest drafted)")
P("B2 = e(u | anchor w) - e(u | anchor w+1)   [one more drafted in-word token, offset control; needs m>=2]")
P("H2 effect = A2 - B2 on tokens with all three outcomes. L = target mass of alternative starts with a different word length.\n")
P("| lang | n | e_k1 | e(anchor w+1) | e(anchor w) | e(anchor w-1) | H2 effect A2-B2 [95% CI] | H2, L>=0.1 | H2, L<0.02 |")
P("|---|---|---|---|---|---|---|---|---|")


def did2(a):
    sel = (a[:, 4] >= 0) & (a[:, 5] >= 0) & (a[:, 9] >= 0)
    b = a[sel]
    if len(b) < 30:
        return np.nan, np.nan, np.nan, len(b)
    val = (b[:, 4] - b[:, 5]) - (b[:, 5] - b[:, 9])
    uniq, inv = np.unique(b[:, 0].astype(int), return_inverse=True)
    s_sum, s_n = np.bincount(inv, weights=val), np.bincount(inv)
    bs = [s_sum[idx].sum() / s_n[idx].sum() for idx in (rng.integers(0, len(uniq), len(uniq)) for _ in range(args.boot))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return s_sum.sum() / s_n.sum(), lo, hi, len(b)


for l in langs_present:
    rec = sum((post_rec[c] for c in conds if c[1] == l), [])
    if not rec:
        continue
    a = np.asarray(rec)
    est, lo, hi, n = did2(a)
    if not np.isfinite(est):
        continue
    sel = (a[:, 4] >= 0) & (a[:, 5] >= 0) & (a[:, 9] >= 0)
    b = a[sel]
    hl = did2(a[~np.isnan(a[:, 8]) & (a[:, 8] >= 0.1)])
    ll = did2(a[~np.isnan(a[:, 8]) & (a[:, 8] < 0.02)])
    k1 = a[a[:, 6] >= 0, 6].mean()
    P(f"| {l} | {n} | {k1:.3f} | {b[:, 9].mean():.3f} | {b[:, 5].mean():.3f} | {b[:, 4].mean():.3f} | {est:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
      f"{hl[0]:+.3f} (n={hl[3]}) | {ll[0]:+.3f} (n={ll[3]}) |")

# ------------------------------------------------------------------ cross-condition correlations
P("\n## 5b. Cross-condition correlations (descriptive; each point = one task x language condition)\n")
if rows_csv:
    t15 = [r["tau15"] for r in rows_csv]
    for nm, key in (("tokens/word", "fert"), ("byte-piece %", "bpiece"), ("in-word %", "inword"), ("mean target entropy", "H")):
        rho, nn = spearman([r[key] for r in rows_csv], t15)
        P(f"- Spearman(DFlash tau@15, {nm}) = {rho:+.2f} (n={nn})")
    X = np.array([[1.0, math.log(r["fert"]), r["H"]] for r in rows_csv if r["fert"] > 0 and math.isfinite(r["tau15"])])
    y = np.array([math.log(r["tau15"]) for r in rows_csv if r["fert"] > 0 and math.isfinite(r["tau15"])])
    if len(y) > 5:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        P(f"- OLS log(tau@15) = {coef[0]:+.2f} {coef[1]:+.2f}*log(tokens/word) {coef[2]:+.2f}*H   (R^2 = {r2:.2f}, n={len(y)})")
P("")

# ------------------------------------------------------------------ stage 0 speed + validation
sp = sum((pload(f) for f in sorted(glob.glob(f"{W}/speed_*.pkl"))), [])
if sp:
    P("\n## 6. Stage 0: wall-clock on one NVIDIA L20 (batch 1, greedy, HF generate vs DFlash spec_generate)\n")
    P("| task | lang | n | AR tok/s | DFlash tok/s | speedup | tau | identical outputs |")
    P("|---|---|---|---|---|---|---|---|")
    g = defaultdict(list)
    for x in sp:
        g[(x["task"], x["lang"])].append(x)
    for k in sorted(g):
        xs = g[k]
        ar = sum(x["n_ar"] for x in xs) / sum(x["t_ar"] for x in xs)
        dfl = sum(x["n_sp"] for x in xs) / sum(x["t_sp"] for x in xs)
        tau = sum(sum(x["accs"]) for x in xs) / sum(len(x["accs"]) for x in xs)
        P(f"| {k[0]} | {k[1]} | {len(xs)} | {ar:.1f} | {dfl:.1f} | {dfl/ar:.2f}x | {tau:.2f} | {sum(x['identical'] for x in xs)}/{len(xs)} |")
    allar = sum(x["n_ar"] for x in sp) / sum(x["t_ar"] for x in sp)
    alld = sum(x["n_sp"] for x in sp) / sum(x["t_sp"] for x in sp)
    P(f"\nOverall: AR {allar:.1f} tok/s, DFlash {alld:.1f} tok/s, speedup {alld/allar:.2f}x\n")
if os.path.exists(f"{W}/val.pkl"):
    v = pload(f"{W}/val.pkl")
    P("## 7. Offline-reproduction validation\n")
    P(f"- anchor-matched acceptance agreement (offline vs real spec_generate): {np.mean([x['anchor_match'] for x in v]):.3f}")
    P(f"- token agreement batched implementation vs official DFlash forward: {np.mean([x['impl_token_agree'] for x in v]):.3f}")
    P(f"- tau real {np.mean([x['tau_real'] for x in v]):.2f} vs offline chain {np.mean([x['tau_off'] for x in v]):.2f} (n={len(v)})\n")
open(f"{OUT}/report.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
