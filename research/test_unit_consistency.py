"""Consistency test: unit_targets (train_unit.py) -> flatten (eval_unit.py) must reproduce the gold continuation.
Also reports how much of the gold continuation a perfect unit-layout drafter could cover (coverage cap)."""
import json, sys, ast, types
import numpy as np
import torch
from collections import defaultdict
from common import ROOT, jload, classify_tokens, token_byte_table
from transformers import AutoTokenizer
src_t = open("train_unit.py", encoding="utf-8").read()
src_e = open("eval_unit.py", encoding="utf-8").read()
ns = {"torch": torch, "BS": 16}
def grab(src, name):
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.get_source_segment(src, n)
exec(grab(src_t, "LAYOUT"), ns)
tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-8B"); bt = token_byte_table(tok)
ns.update(EOU_ID=10**9, classify_tokens=classify_tokens, bt=bt, UNIT=True, args=types.SimpleNamespace(cap=0))
for f in ("units_of", "unit_targets"):
    exec(grab(src_t, f), ns)
for f in ("flatten", "_flatten"):
    exec(grab(src_e, f), ns)
ns["LAYOUT"] = ns["LAYOUT"]
keep = set(json.load(open(f"{ROOT}/work/clean_ids_q8.json")))
rows = [r for r in jload(f"{ROOT}/work/q8/gen_0.jsonl") if r["id"] in keep]
by = defaultdict(list)
for r in rows:
    by[r["lang"]].append(r)
for lang in ["en", "de", "zh", "th", "hi", "bn", "te", "am"]:
    ok = tot = 0; cov = []; trunc = 0
    for r in by[lang][:15]:
        P = len(r["prompt_ids"]); ids = r["prompt_ids"] + r["output_ids"]; N = len(ids)
        units = ns["units_of"](r["output_ids"], lang)
        A = list(range(P, N - 1))
        lab = ns["unit_targets"](ids, P, A, units, lang)
        for b, s in enumerate(A):
            row = [int(x) if x != -100 else 10**9 for x in lab[b].tolist()] + [10**9]
            fl = ns["flatten"](row, lang)
            gold = ids[s + 1: s + 1 + len(fl)]
            tot += 1; ok += fl == gold
            cov.append(len(fl))
    print(f"{lang}: flatten(gold labels) == gold continuation for {ok}/{tot} anchors; mean coverage {np.mean(cov):.1f} tokens; "
          f"P(coverage >= 8) {np.mean(np.array(cov) >= 8):.2f}")
