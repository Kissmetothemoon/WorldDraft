"""Validate the Triton WorldDraft walk against the python reference on real DFlash logits; time it.
usage: test_lex_triton.py <target> <draft> <table_dir> <gen jsonl> <langs comma> <lam>"""
import json
import sys
import time

import numpy as np
import torch

from common import token_byte_table
from lex_triton import LexTables, lex_walk, lex_walk_ref

tp, dp, tdir, genf, langs, lam = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(","), float(sys.argv[6])
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(tp)
bt = token_byte_table(tok)
target = AutoModelForCausalLM.from_pretrained(tp, torch_dtype=torch.bfloat16, device_map=dev).eval()
draft = AutoModel.from_pretrained(dp, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
V = target.lm_head.out_features
cont = np.zeros(V, dtype=bool)
for t, bs in enumerate(bt):
    if bs:
        cont[t] = (bs[0] >= 0x80) or chr(bs[0]).isalnum()
sw = np.array([cont[t] or (t < len(bt) and bt[t][:1] == b" " and len(bt[t]) > 1) for t in range(V)])
tabs = LexTables(tdir, cont, sw, dev)
BS, MASK, LIDS = draft.block_size, draft.mask_token_id, draft.target_layer_ids
rows = [json.loads(l) for l in open(genf)]
for lang in langs:
    tabs.load(lang, lam)
    rs = [r for r in rows if r["lang"] == lang][:4]
    L, H = [], []
    with torch.inference_mode():
        for r in rs:
            ids_l = r["prompt_ids"] + r["output_ids"]
            ids = torch.tensor([ids_l], device=dev)
            out = target(ids, output_hidden_states=True)
            feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1)
            P = len(r["prompt_ids"])
            for s0 in range(P + 2, len(ids_l) - BS, max(1, (len(ids_l) - P) // 16))[:16]:
                blk = torch.full((1, BS), MASK, dtype=torch.long, device=dev)
                blk[0, 0] = ids[0, s0]
                hid = draft(target_hidden=feats[:, :s0], noise_embedding=target.model.embed_tokens(blk),
                            position_ids=torch.arange(s0 + BS, device=dev).unsqueeze(0), is_causal=False)
                L.append(target.lm_head(hid[:, 1:, :])[0])
                H.append(ids_l[max(0, s0 - 11):s0 + 1])
    logits = torch.stack(L)
    hist = torch.full((len(H), 12), -1, dtype=torch.int32)
    for i, h in enumerate(H):
        hist[i, 12 - len(h):] = torch.tensor(h, dtype=torch.int32)
    hist = hist.to(dev)
    a = lex_walk(logits, hist, tabs).cpu()
    b = lex_walk_ref(logits, hist, tabs)
    base = logits.argmax(-1).cpu()
    mm = (a != b).nonzero().tolist()[:3]
    for bi, si in mm:
        lp = torch.log_softmax(logits[bi, si].float(), -1)
        print("   mismatch", bi, si, "kernel", int(a[bi, si]), tok.decode([int(a[bi, si])]), float(lp[a[bi, si]]),
              "ref", int(b[bi, si]), tok.decode([int(b[bi, si])]), float(lp[b[bi, si]]), "prev", a[bi, :si].tolist() == b[bi, :si].tolist(),
              "hist", [tok.decode([t]) for t in hist[bi].tolist() if t >= 0][-4:])
    print(f"{lang}: B={len(H)} kernel==ref {(a == b).float().mean():.4f}; changed vs argmax {(a != base).float().mean():.3f}", flush=True)
    for B in (1, 8, 32):
        lg, hs = logits[:B].contiguous(), hist[:B].contiguous()
        for _ in range(3):
            lex_walk(lg, hs, tabs)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            lex_walk(lg, hs, tabs)
        torch.cuda.synchronize()
        print(f"   B={B}: lex_walk {1000*(time.time()-t)/50:.3f} ms per call (incl. topk + logsumexp)", flush=True)
