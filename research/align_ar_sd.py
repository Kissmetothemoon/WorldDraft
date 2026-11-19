"""G3 baseline alignment: our ar_sd_generate vs HF's official assisted generation (same target/draft, constant K).

Checks: (1) outputs vs AR greedy, (2) tokens/s vs HF assisted generation, (3) real-loop tau vs offline tau computed
from one teacher-forced pass of both models over our own output (validates ar_offline.py).
"""
import argparse
import time

import numpy as np
import torch

from common import build_messages, chat_ids, jload, pdump, shard
from sd_lib import ar_sd_generate

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--K", type=int, default=5)
ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
args = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
stop = sorted({tok.eos_token_id} | {t for t in [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")]
                                    if isinstance(t, int) and 0 <= t < len(tok) and t != tok.unk_token_id})
K = args.K


def hf_assisted(pid):
    return target.generate(pid, assistant_model=draft, do_sample=False, max_new_tokens=args.max_new, eos_token_id=stop,
                           pad_token_id=stop[0], num_assistant_tokens=K, num_assistant_tokens_schedule="constant",
                           assistant_confidence_threshold=0.0)[0, pid.shape[1]:].tolist()


@torch.inference_mode()
def offline_tau(pid_list, gen):
    ids = torch.tensor([pid_list + gen], device=dev)
    P, N = len(pid_list), ids.shape[1]
    gold = target(ids).logits[0, P - 1:N - 1].argmax(-1)
    pred = draft(ids).logits[0, P - 1:N - 1].argmax(-1)
    match = (pred == gold).cpu().numpy()
    a, accs = 0, []
    while a < len(match):
        run = 0
        while run < K and a + run < len(match) and match[a + run]:
            run += 1
        accs.append(run + 1)
        a += run + 1
    return accs


rows = shard(jload(args.prompts), args.shard, args.nshard)
w = torch.tensor([chat_ids(tok, build_messages("chat", "en", "Hello!"))], device=dev)
target.generate(w, max_new_tokens=8, do_sample=False, pad_token_id=stop[0])
hf_assisted(w)
ar_sd_generate(target, draft, w, 16, stop, K)
rep = []
for i, r in enumerate(rows):
    pl = chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))
    pid = torch.tensor([pl], device=dev)
    res = {"id": r["id"]}
    for name, fn in (("ar", lambda: target.generate(pid, max_new_tokens=args.max_new, do_sample=False, eos_token_id=stop,
                                                      pad_token_id=stop[0])[0, pid.shape[1]:].tolist()),
                     ("hf_assisted", lambda: hf_assisted(pid)),
                     ("ours", lambda: ar_sd_generate(target, draft, pid, args.max_new, stop, K))):
        torch.cuda.synchronize(); t = time.time()
        o = fn()
        torch.cuda.synchronize()
        if name == "ours":
            o, accs = o
            res["accs"] = accs
        res[name] = {"n": len(o), "t": time.time() - t, "ids": o}
    res["offline_accs"] = offline_tau(pl, res["ours"]["ids"])
    rep.append(res)
    if i % 10 == 0:
        print(f"[{i}/{len(rows)}] ar {res['ar']['n']/res['ar']['t']:.1f} hf {res['hf_assisted']['n']/res['hf_assisted']['t']:.1f} "
              f"ours {res['ours']['n']/res['ours']['t']:.1f} tok/s; tau real {np.mean(res['accs']):.2f} offline {np.mean(res['offline_accs']):.2f}", flush=True)
pdump(rep, args.out)
print("done", flush=True)
