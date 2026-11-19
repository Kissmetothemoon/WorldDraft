"""Descriptive cross-drafter reference: EAGLE-3 (chain, K tokens) acceptance length per (task, lang) in vLLM."""
import argparse
import json
from collections import defaultdict

from common import MAX_NEW, build_messages, chat_ids, jload, pdump

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--eagle", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=7)
ap.add_argument("--per-group", type=int, default=100)
ap.add_argument("--groups", default="", help="comma list of task:lang to restrict (for sharding)")
args = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

tok = AutoTokenizer.from_pretrained(args.model)
groups = defaultdict(list)
for r in jload(args.prompts):
    groups[(r["task"], r["lang"])].append(r)
want = set(tuple(g.split(":")) for g in args.groups.split(",") if g) or set(groups)

llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096,
          speculative_config={"method": "eagle3", "model": args.eagle, "num_speculative_tokens": args.k},
          disable_log_stats=False, seed=0)


def spec_counters():
    vals = {}
    for m in llm.get_metrics():
        if "spec_decode" not in m.name:
            continue
        if hasattr(m, "values"):
            vals[m.name] = list(m.values)
        else:
            vals[m.name] = m.value
    return vals


res = {}
for key in sorted(want):
    rows = groups[key][: args.per_group]
    ps = [TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))) for r in rows]
    before = spec_counters()
    llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=MAX_NEW[key[0]]), use_tqdm=False)
    after = spec_counters()
    d = {}
    for n, v in after.items():
        b = before.get(n, 0 if not isinstance(v, list) else [0] * len(v))
        d[n] = [x - y for x, y in zip(v, b)] if isinstance(v, list) else v - b
    drafts = d.get("vllm:spec_decode_num_drafts", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens", 0)
    tau = 1 + acc / drafts if drafts else float("nan")
    res["%s:%s" % key] = {"tau": tau, "n": len(rows), "counters": d}
    print(key, f"tau={tau:.3f}", json.dumps(d.get("vllm:spec_decode_num_accepted_tokens_per_pos", [])), flush=True)
pdump(res, args.out)

# vLLM's engine shutdown can hang after the output is written: kill our own child processes
# (VLLM::EngineCore etc.) first so nothing is left holding GPU memory, then exit immediately.
import multiprocessing as _mp, os as _os, signal as _sig, sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
for _c in _mp.active_children():
    _c.kill()
try:
    import psutil as _ps
    for _c in _ps.Process().children(recursive=True):
        _c.kill()
except Exception:
    pass
_os._exit(0)
