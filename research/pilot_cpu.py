"""CPU pilots on existing records (Qwen3-8B evaluation trajectories, clean only).

P1 (route M1, word-slot drafting) feasibility:
  a. word/token structure: tokens per word (mean / p90), tokens straddling a word boundary, canonical re-encoding of words
  b. does block-drafter acceptance decay with TOKEN offset k or with WORD offset j?  hazard tables at word-start tokens
  c. hybrid ceiling: word starts drafted by the block drafter, in-word continuation tokens by an AR drafter that sees the
     word's own prefix (teacher-forced AR record) -> tau@15 vs block-only and AR-only on the same anchors
P5 (route M5, verification truncation): simulate fixed / confidence / word-aware truncation with the measured engine
  per-step cost ratios c(L) (prof_exact.md) -> predicted speedup per language at concurrency 1 / 8 / 32
usage: pilot_cpu.py <block_tag> <ar_tag> <gen_tag> <tokenizer> <out.md>
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

from common import ROOT, _segment, lang_ok, pload, token_byte_table, utf8_chars

btag, atag, gtag, tokp, outp = sys.argv[1:6]
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(tokp)
bt = token_byte_table(tok)
keep = set(json.load(open(f"{ROOT}/work/clean_ids_{gtag}.json")))
LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "ko", "ar", "th", "sw", "hi", "bn", "te", "am", "sr-Cyrl", "sr-Latn", "hi-Latn"]


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/work/{tag}/tf_*.pkl")):
        for r in pload(f):
            if r["id"] in keep and r.get("corr") is not None:
                out[r["id"]] = r
    return out


B, A = load(btag), load(atag)
ids_common = [i for i in B if i in A]
if len(sys.argv) > 6:
    ids_common = ids_common[: int(sys.argv[6])]
print("sequences", len(ids_common), flush=True)


def words_of(out_ids, lang):
    """per token: word index (-1 for whitespace/punct-only), is_word_start, straddles (covers bytes of 2 words)."""
    starts, buf = [], bytearray()
    for t in out_ids:
        starts.append(len(buf))
        buf += bt[t]
    ends = starts[1:] + [len(buf)]
    chars = utf8_chars(bytes(buf))
    text = "".join(c for _, _, c in chars)
    spans = _segment(text, lang)
    wb = [(chars[a][0], chars[b - 1][1]) for a, b in spans if a < len(chars) and b - 1 < len(chars)]
    n = len(out_ids)
    widx, wstart, strad = np.full(n, -1), np.zeros(n, bool), np.zeros(n, bool)
    import bisect
    wst = [x[0] for x in wb]
    for k in range(n):
        s0, e0 = starts[k], ends[k]
        j0 = bisect.bisect_right(wst, s0) - 1
        hit = []
        j = max(j0, 0)
        while j < len(wb) and wb[j][0] < e0:
            if wb[j][1] > s0:
                hit.append(j)
            j += 1
        if hit:
            widx[k] = hit[0]
            strad[k] = len(hit) > 1
    seen = set()
    for k in range(n):
        if widx[k] >= 0 and widx[k] not in seen:
            wstart[k] = True
            seen.add(widx[k])
    return widx, wstart, strad, wb, bytes(buf)


# cost ratios c_SD / c_AR per verification length (measured, averaged over languages; prof_exact.md, DFlash ml)
COST = {1: {3: 1.32, 7: 1.32, 15: 1.33}, 8: {3: 1.35, 7: 1.37, 15: 1.44}, 32: {3: 1.65, 7: 1.90, 15: 2.68}}


def cost(conc, L):
    pts = sorted(COST[conc].items())
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(L, xs, ys, left=ys[0] + (L - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])))


ST = defaultdict(lambda: defaultdict(float))
HZ = defaultdict(lambda: np.zeros((6, 16, 2)))  # lang -> word offset j (0..5+) x token offset k -> [n, accepted]
HZA = defaultdict(lambda: np.zeros((16, 2)))     # lang -> token offset k (all tokens), block
TR = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))  # lang -> policy -> [tokens, sum L, rounds]
thetas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
for n_i, sid in enumerate(ids_common):
    rb, ra = B[sid], A[sid]
    l = rb["lang"]
    out = rb["output_ids"]
    widx, wstart, strad, wb, buf = words_of(out, l)
    S = ST[l]
    S["tokens"] += len(out)
    S["words"] += len(wb)
    S["strad"] += strad.sum()
    wl = np.bincount(widx[widx >= 0]) if (widx >= 0).any() else np.array([0])
    S["wl_sum"] += wl.sum()
    S["wl_n"] += len(wl)
    S["wl_p90"] += np.percentile(wl, 90) if len(wl) else 0
    S["seqs"] += 1
    # canonical re-encoding of each word (bytes of its tokens, leading whitespace included)
    for j in range(len(wb)):
        tk = np.where(widx == j)[0]
        if len(tk) == 0 or strad[tk].any():
            continue
        wbytes = b"".join(bt[out[k]] for k in tk)
        try:
            txt = wbytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        S["canon_n"] += 1
        S["canon_ok"] += tok.encode(txt, add_special_tokens=False) == [out[k] for k in tk]
    cb, ca = rb["corr"], ra["corr"]
    R, K = cb.shape
    T = len(out)
    lead_b = np.cumprod((cb == 1).astype(np.int32), 1).sum(1)
    # hazard tables (block drafter): risk set = all earlier positions accepted
    for a in range(R):
        nw = 0
        for k in range(1, K + 1):
            t = a + k
            if t >= T or cb[a, k - 1] == 2:
                break
            if wstart[t]:
                nw += 1
            acc = cb[a, k - 1] == 1
            HZA[l][k] += [1, acc]
            if wstart[t]:
                HZ[l][min(nw - 1, 5), k] += [1, acc]
            if not acc:
                break
    # hybrid chain: word starts / other from the block drafter, in-word continuations from the AR drafter (one-step,
    # teacher-forced: row t-1, column 0 predicts token t)
    def hyb_ok(a, k):
        t = a + k
        if widx[t] >= 0 and not wstart[t]:
            return ca[t - 1, 0] if t - 1 < ca.shape[0] else 2
        return cb[a, k - 1]

    for name, fn in (("block", lambda a, k: cb[a, k - 1]), ("hybrid", hyb_ok),
                     ("AR", lambda a, k: ca[a, k - 1] if k - 1 < ca.shape[1] else 2)):
        a, acc_sum, rounds, seq_steps = 0, 0, 0, 0
        while a < R:
            n = 0
            for k in range(1, K + 1):
                if a + k >= T:
                    break
                v = fn(a, k)
                if v != 1:
                    break
                n += 1
                if name == "hybrid" and widx[a + k] >= 0 and not wstart[a + k]:
                    seq_steps += 1
            acc_sum += n + 1
            rounds += 1
            a += n + 1
        S[name + "_acc"] += acc_sum
        S[name + "_rounds"] += rounds
        if name == "hybrid":
            S["hyb_seq"] += seq_steps
    # truncation policies on the block drafter (dprob = drafter top-1 prob per position)
    p = rb["dprob"].astype(np.float64)
    surv = np.cumprod(p, 1)
    nxt_ws = np.zeros((R, K), bool)  # gold token at a+k+1 starts a word or is non-word (boundary after position k)
    for a in range(R):
        for k in range(1, K + 1):
            t = a + k + 1
            nxt_ws[a, k - 1] = (t >= T) or wstart[t] or widx[t] < 0
    a = 0
    pol = {f"fixed {L}": (lambda a, L=L: L) for L in (1, 2, 3, 5, 7, 10, 15)}
    for th in thetas:
        pol[f"conf {th}"] = (lambda a, th=th: max(1, int((surv[a] >= th).sum())))
        pol[f"conf {th} + word cut"] = (lambda a, th=th: _cut(a, max(1, int((surv[a] >= th).sum()))))

    def _cut(a, L):
        if L >= K:
            return L
        for m in range(L, 0, -1):
            if nxt_ws[a, m - 1]:
                return m
        return L

    for name, f in pol.items():
        a = 0
        while a < R:
            L = f(a)
            n = min(int(lead_b[a]), L)
            TR[l][name][0] += n + 1
            TR[l][name][1] += L
            TR[l][name][2] += 1
            a += n + 1
    if n_i % 500 == 0:
        print(f"[{n_i}/{len(ids_common)}]", flush=True)

L = [f"# CPU pilots: block={btag}, AR={atag}, trajectories={gtag} (clean)\n",
     "## P1a. Word structure of the target's outputs\n",
     "| lang | tokens/word | p90 tokens/word | tokens straddling a word boundary | words whose tokens = canonical re-encoding |",
     "|---|---|---|---|---|"]
for l in LANGS:
    S = ST.get(l)
    if not S:
        continue
    L.append(f"| {l} | {S['wl_sum']/max(1,S['wl_n']):.2f} | {S['wl_p90']/S['seqs']:.1f} | {100*S['strad']/S['tokens']:.2f}% | "
             f"{100*S['canon_ok']/max(1,S['canon_n']):.1f}% |")
L += ["", "## P1b. Block-drafter acceptance hazard at WORD-START tokens, by word offset j (rows) and token offset k (cols)",
      "(P(accept | all earlier drafted tokens accepted); n in brackets)\n"]
for l in ["en", "de", "zh", "th", "hi", "bn", "te", "sr-Cyrl"]:
    if l not in HZ:
        continue
    H = HZ[l]
    L.append(f"### {l}\n")
    L.append("| word offset j | " + " | ".join(f"k={k}" for k in (1, 2, 3, 4, 6, 8, 11, 15)) + " | pooled |")
    L.append("|---|" + "---|" * 9)
    for j in range(5):
        cells = [f"{H[j,k,1]/H[j,k,0]:.2f} ({int(H[j,k,0])})" if H[j, k, 0] >= 30 else "-" for k in (1, 2, 3, 4, 6, 8, 11, 15)]
        tot = H[j].sum(0)
        L.append(f"| {j+1} | " + " | ".join(cells) + f" | {tot[1]/max(1,tot[0]):.2f} |")
    L.append("")
L += ["## P1c. Hybrid ceiling (tau@15): block drafter for word starts, AR drafter (own word prefix) for in-word tokens\n",
      "| lang | block only | hybrid | AR only | hybrid / block | sequential in-word steps per round |", "|---|---|---|---|---|---|"]
for l in LANGS:
    S = ST.get(l)
    if not S:
        continue
    tb, th_, ta = (S[f"{n}_acc"] / S[f"{n}_rounds"] for n in ("block", "hybrid", "AR"))
    L.append(f"| {l} | {tb:.2f} | {th_:.2f} | {ta:.2f} | {th_/tb:.2f} | {S['hyb_seq']/S['hybrid_rounds']:.2f} |")
L += ["", "## P5. Verification truncation: predicted speedup = tokens per round / c(mean L), c from measured engine costs\n"]
for conc in (1, 8, 32):
    L.append(f"### concurrency {conc}\n")
    names = list(next(iter(TR.values())).keys())
    show = ["fixed 3", "fixed 7", "fixed 15"]
    L.append("| lang | fixed 3 | fixed 7 | fixed 15 | best fixed | best conf | best conf + word cut |")
    L.append("|---|---|---|---|---|---|---|")
    for l in LANGS:
        if l not in TR:
            continue
        sp = {n: (v[0] / v[2]) / cost(conc, v[1] / v[2]) for n, v in TR[l].items()}
        bf = max((n for n in sp if n.startswith("fixed")), key=sp.get)
        bc = max((n for n in sp if n.startswith("conf") and "word" not in n), key=sp.get)
        bw = max((n for n in sp if "word" in n), key=sp.get)
        L.append(f"| {l} | " + " | ".join(f"{sp[n]:.2f}" for n in show)
                 + f" | {sp[bf]:.2f} ({bf}) | {sp[bc]:.2f} ({bc}) | {sp[bw]:.2f} ({bw.replace(' + word cut','')}) |")
    L.append("")
txt = "\n".join(L) + "\n"
open(outp, "w", encoding="utf-8").write(txt)
print(txt)
