"""Greedy target trajectories with vLLM (one shard per GPU)."""
import argparse
import os
import time

from common import MAX_NEW, build_messages, chat_ids, jdump, jload, shard

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--max-prompt", type=int, default=1024)
ap.add_argument("--max-new", type=int, default=0, help="override per-task max tokens")
ap.add_argument("--messages-field", default=None, help="use prebuilt messages field instead of task/lang template")
args = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

tok = AutoTokenizer.from_pretrained(args.model)
rows = shard(jload(args.prompts), args.shard, args.nshard)
prompts, keep = [], []
for r in rows:
    msgs = r[args.messages_field] if args.messages_field else build_messages(r["task"], r["lang"], r["text"])
    ids = chat_ids(tok, msgs)
    if len(ids) > args.max_prompt:
        continue
    r["prompt_ids"] = ids
    keep.append(r)
print(f"shard {args.shard}: {len(keep)}/{len(rows)} prompts", flush=True)

llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=4096,
          enable_prefix_caching=True, seed=0)
sps = [SamplingParams(temperature=0.0, max_tokens=args.max_new or MAX_NEW.get(r.get("task"), 512)) for r in keep]
t = time.time()
outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in keep], sps, use_tqdm=False)
ntok = 0
for r, o in zip(keep, outs):
    c = o.outputs[0]
    r["output_ids"] = list(c.token_ids)
    r["finish"] = c.finish_reason
    r["text_out"] = c.text
    ntok += len(c.token_ids)
print(f"shard {args.shard}: {ntok} tokens in {time.time()-t:.0f}s", flush=True)
jdump(keep, args.out)

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
