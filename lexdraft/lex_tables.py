"""Build GPU-ready LexDraft tables per language from monolingual text (same statistics as pilot_lex.py / sd_lex.py).

usage: lex_tables.py <tokenizer> "<glob of wiki_<lang>.jsonl>" <out_dir> [max_tokens_per_lang]
per language writes <out_dir>/<lang>.npz with
  ctx_keys  int64 sorted   in-word context key  n | w[-1]<<2 | w[-2]<<20 | w[-3]<<38  (only contexts with >= 2 observations)
  ctx_end   f32            log p(word ends | ctx)
  ctx_floor f32            log of the smoothed floor for unseen continuations
  ctx_top   int32 (Nc, 4)  top-4 continuations (-1 pad)
  pair_keys int64 sorted   ctx_index << 18 | next   (top-32 continuations per context)
  pair_lp   f32            log p(next | ctx)
"""
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

tokp, pat, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
cap = float(sys.argv[4]) if len(sys.argv) > 4 else 1e7
os.makedirs(out_dir, exist_ok=True)
LX, E = 32, 4


def tables(f):
    from transformers import AutoTokenizer
    from common import token_byte_table
    tok = AutoTokenizer.from_pretrained(tokp)
    bt = token_byte_table(tok)
    V = max(len(bt), 1 << 18)
    CONT = np.zeros(V, dtype=bool)
    for t, bs in enumerate(bt):
        if bs:
            CONT[t] = (bs[0] >= 0x80) or chr(bs[0]).isalnum()
    SW = np.array([CONT[t] or (bt[t][:1] == b" " and len(bt[t]) > 1) if t < len(bt) else False for t in range(V)])
    C = defaultdict(Counter)
    used, lang = 0, None
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        lang = r["lang"]
        out = r["output_ids"]
        used += len(out)
        if used > cap:
            break
        word = []
        for t in out:
            if CONT[t] and word:
                for n in range(1, 4):
                    if len(word) >= n:
                        C[(n,) + tuple(word[-n:])][t] += 1
                word.append(t)
            else:
                if word:
                    for n in range(1, 4):
                        if len(word) >= n:
                            C[(n,) + tuple(word[-n:])][-1] += 1
                word = [t] if SW[t] else []
    keys, ends, floors, tops, pk, pl = [], [], [], [], [], []
    items = []
    for ctx, c in C.items():
        tot = sum(c.values())
        if tot < 2:
            continue
        n, w = ctx[0], ctx[1:]
        key = n | (w[-1] << 2) | ((w[-2] if n >= 2 else 0) << 20) | ((w[-3] if n >= 3 else 0) << 38)
        items.append((key, c, tot))
    items.sort(key=lambda x: x[0])
    for i, (key, c, tot) in enumerate(items):
        keys.append(key)
        ends.append(math.log((c.get(-1, 0) + 0.5) / (tot + 1)))
        floors.append(math.log(0.5 / (tot + len(c))))
        cont = [(k, v) for k, v in c.most_common(LX + 1) if k != -1][:LX]
        tops.append([k for k, _ in cont[:E]] + [-1] * (E - min(E, len(cont))))
        for k, v in cont:
            pk.append((i << 18) | k)
            pl.append(math.log(v / tot))
    order = np.argsort(np.array(pk, dtype=np.int64), kind="stable")
    np.savez(f"{out_dir}/{lang}.npz", ctx_keys=np.array(keys, dtype=np.int64), ctx_end=np.array(ends, dtype=np.float32),
             ctx_floor=np.array(floors, dtype=np.float32), ctx_top=np.array(tops, dtype=np.int32),
             pair_keys=np.array(pk, dtype=np.int64)[order], pair_lp=np.array(pl, dtype=np.float32)[order])
    return lang, len(keys), len(pk), used


if __name__ == "__main__":
    files = sorted(glob.glob(pat))
    with Pool(len(files)) as p:
        for lang, nc, npair, used in p.imap_unordered(tables, files):
            print(lang, "contexts", nc, "pairs", npair, "tokens", used, flush=True)
