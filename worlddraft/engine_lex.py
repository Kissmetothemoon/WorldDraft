"""Optimised-engine benchmark (vLLM nightly, V2 runner) with WorldDraft inside the DFlash drafter (greedy).

The DFlash speculator's sample_draft is replaced by: drafter logits -> Triton WorldDraft walk (lex_triton.py) using the
request's last 12 accepted tokens (req_states.all_token_ids / total_len, written before propose()). Tables live in
fixed GPU buffers, so the drafter's CUDA graphs stay valid; the language and lambda are switched per (task, lang) group.
Same protocol as engine_bench.py: per group, chunks of `conc` requests; tokens/s, engine tau, generated token ids.
--lex-dir '' runs the unmodified engine (any --method).
"""
import argparse
import json
import os
import time
from collections import defaultdict

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import numpy as np
import torch

from common import MAX_NEW, build_messages, chat_ids, jload, pdump, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--method", default="none")
ap.add_argument("--spec-model", default=None)
ap.add_argument("--k", type=int, default=15)
ap.add_argument("--verify-sched", default="")
ap.add_argument("--lex-dir", default="")
ap.add_argument("--lams", default="", help="lang:lam,... (missing languages: off)")
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conc", default="1,8,32")
ap.add_argument("--per-group", type=int, default=32)
ap.add_argument("--tasks", default="math,translate,chat")
ap.add_argument("--langs", default="")
args = ap.parse_args()

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(args.model)
LEX = {"tabs": None, "rs": None}
if args.lex_dir:
    from vllm.v1.worker.gpu import model_runner as MR
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as DS
    from lex_triton import LexTables, lex_walk

    bt = token_byte_table(tok)
    _st = MR.GPUModelRunner.sample_tokens

    def sample_tokens(self, *a, **kw):
        LEX["rs"] = self.req_states
        return _st(self, *a, **kw)

    MR.GPUModelRunner.sample_tokens = sample_tokens
    _cap = MR.GPUModelRunner.capture_model

    def capture_model(self, *a, **kw):  # the drafter's CUDA graphs must be captured WITH the WorldDraft path
        LEX["rs"] = self.req_states
        return _cap(self, *a, **kw)

    MR.GPUModelRunner.capture_model = capture_model
    V0 = max(len(bt), 151936)
    cont = np.zeros(V0, dtype=bool)
    for t, bs in enumerate(bt):
        if bs:
            cont[t] = (bs[0] >= 0x80) or chr(bs[0]).isalnum()
    sw = np.array([cont[t] or (t < len(bt) and bt[t][:1] == b" " and len(bt[t]) > 1) for t in range(V0)])
    LEX["tabs"] = LexTables(args.lex_dir, cont, sw, "cuda:0")
    LEX["tabs"].load("en", 0.0)
    _sd = DS.DFlashSpeculator.sample_draft
    HIST = 12

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits,
                     spec_step_idx=0):
        tabs, rs = LEX["tabs"], LEX["rs"]
        if tabs is None or rs is None:
            return _sd(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits,
                       spec_step_idx)
        logits = self.compute_draft_logits(hidden_states, spec_step_idx)
        S = self.num_speculative_steps
        n = logits.shape[0] // S
        im = idx_mapping.view(n, S)[:, 0].clamp(min=0)
        tot = rs.total_len.gpu[im].long()
        pos = tot.unsqueeze(1) - HIST + torch.arange(HIST, device=logits.device).unsqueeze(0)
        hist = rs.all_token_ids.gpu[im.unsqueeze(1), pos.clamp(min=0)].to(torch.int32)
        hist = torch.where(pos >= 0, hist, torch.full_like(hist, -1))
        out = lex_walk(logits.view(n, S, -1), hist, tabs)
        return out.view(-1).to(torch.int64)

    DS.DFlashSpeculator.sample_draft = sample_draft

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

conc = [int(c) for c in args.conc.split(",")]
spec = None
if args.method != "none":
    spec = {"method": args.method, "model": args.spec_model, "num_speculative_tokens": args.k}
    if args.verify_sched:
        spec["num_speculative_tokens_per_batch_size"] = [tuple(int(x) for x in p.replace(":", "-").split("-"))
                                                         for p in args.verify_sched.split(",")]
llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.80, max_model_len=4096, max_num_seqs=max(conc),
          enable_prefix_caching=False, seed=0, speculative_config=spec, disable_log_stats=False)
LAMS = {k: float(v) for k, v in (x.split(":") for x in args.lams.split(","))} if args.lams else {}


def counters():
    v = {}
    for m in llm.get_metrics():
        if "spec_decode" in m.name and not hasattr(m, "values"):
            v[m.name] = m.value
    return v


groups = defaultdict(list)
tasks = set(args.tasks.split(","))
langs = set(args.langs.split(",")) if args.langs else None
for r in jload(args.prompts):
    if r["task"] in tasks and (langs is None or r["lang"] in langs):
        groups[(r["task"], r["lang"])].append(r)
llm.generate([TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages("chat", "en", "Hello!")))],
             SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
res = []
for key in sorted(groups):
    if LEX["tabs"] is not None:
        LEX["tabs"].load(key[1], LAMS.get(key[1], 0.0))
    rows = groups[key][: args.per_group]
    ps = [TokensPrompt(prompt_token_ids=chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))) for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW.get(key[0], 2048))
    for c in conc:
        before = counters()
        t, ntok, outs_all = time.time(), 0, []
        for i in range(0, len(ps), c):
            outs = llm.generate(ps[i:i + c], sp, use_tqdm=False)
            ntok += sum(len(o.outputs[0].token_ids) for o in outs)
            outs_all += [list(o.outputs[0].token_ids) for o in outs]
        dt = time.time() - t
        after = counters()
        d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        drafts = d.get("vllm:spec_decode_num_drafts", 0)
        tau = 1 + d.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts if drafts else float("nan")
        res.append({"task": key[0], "lang": key[1], "conc": c, "n": len(rows), "ids": [r["id"] for r in rows], "tokens": ntok,
                    "time": dt, "tps": ntok / dt, "tau": tau, "out_ids": outs_all})
        print(json.dumps({k: res[-1][k] for k in ("task", "lang", "conc", "tokens", "tps", "tau")}), flush=True)
pdump({"args": vars(args), "res": res}, args.out)

import multiprocessing as _mp, os as _os, sys as _sys
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
