"""Prompts whose greedy target response is usable: language check passes and no looping (same filter as analyze.py).

usage: build_clean_prompts.py <gen_tag>   -> work/prompts_eval_clean_<gen_tag>.jsonl, work/clean_ids_<gen_tag>.json
"""
import glob
import json
import sys
from collections import Counter

from common import ROOT, jload, lang_ok

tag = sys.argv[1]
ok, stat = set(), Counter()
for f in sorted(glob.glob(f"{ROOT}/work/{tag}/gen_*.jsonl")):
    for r in jload(f):
        out = r["output_ids"]
        g4 = [tuple(out[i:i + 4]) for i in range(max(0, len(out) - 3))]
        degen = (1 - len(set(g4)) / max(1, len(g4))) > 0.5
        good = lang_ok(r["text_out"], r["lang"]) and not degen and len(out) >= 8
        stat[(r["task"], r["lang"], good)] += 1
        if good:
            ok.add(r["id"])
rows = [r for r in jload(f"{ROOT}/work/prompts_eval.jsonl") if r["id"] in ok]
with open(f"{ROOT}/work/prompts_eval_clean_{tag}.jsonl", "w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(sorted(ok), open(f"{ROOT}/work/clean_ids_{tag}.json", "w"))
for t, l in sorted({(t, l) for t, l, _ in stat}):
    print(t, l, "kept", stat[(t, l, True)], "dropped", stat[(t, l, False)])
print("clean prompts", len(rows))
