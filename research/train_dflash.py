"""Fine-tune a released DFlash drafter on target-generated data (H0 data-matching test).

Objective = the standard DFlash block objective: for random anchors s in a response, block = [x_s, MASK x (B-1)],
context = frozen target features of positions < s; cross-entropy on x_{s+1..s+B-1} (exp. decaying position weights).
All blocks of one sequence share one attention call (block b sees context < s_b and its own block only).

torchrun --nproc_per_node 4 train_dflash.py --target ... --init ... --data 'glob' --out ...
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--data", required=True, help="glob of gen_vllm.py jsonl files")
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=4, help="sequences per optimizer step per GPU")
ap.add_argument("--anchors", type=int, default=32)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--decay", type=float, default=7.0, help="position weight exp(-(k-1)/decay)")
ap.add_argument("--max-seqs", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reveal-p", type=float, default=0.0,
                help="DFlash-mix: fraction of blocks with revealed in-block tokens (true tokens, excluded from the loss)")
args = ap.parse_args()

ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = torch.device("cuda", torch.cuda.current_device())
random.seed(args.seed + rank)
torch.manual_seed(args.seed)

from transformers import AutoModel, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import rotate_half

target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map={"": dev},
                                              attn_implementation="sdpa").eval()
for p in target.parameters():
    p.requires_grad_(False)
draft = AutoModel.from_pretrained(args.init, trust_remote_code=True, torch_dtype=torch.float32, device_map={"": dev})
draft.train()
LIDS, BS, MASK = draft.target_layer_ids, draft.block_size, draft.mask_token_id
cfg = draft.config
HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads

# ------------------------------------------------------------------ data
rows = []
for f in sorted(glob.glob(args.data)):
    with open(f, encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            ids = r["prompt_ids"] + r["output_ids"]
            if len(r["output_ids"]) < 8:
                continue
            rows.append((ids[: args.max_len], len(r["prompt_ids"])))
random.Random(args.seed).shuffle(rows)
if args.max_seqs:
    rows = rows[: args.max_seqs]
mine = rows[rank::world]
n_steps = int(len(mine) * args.epochs) // args.accum
if args.max_steps:
    n_steps = min(n_steps, args.max_steps)
if ddp:  # ranks can hold different numbers of sequences; a step-count mismatch deadlocks the final barrier
    t_steps = torch.tensor([n_steps], device=dev)
    dist.all_reduce(t_steps, op=dist.ReduceOp.MIN)
    n_steps = int(t_steps.item())
if rank == 0:
    print(f"{len(rows)} sequences, {world} GPUs, {n_steps} optimizer steps", flush=True)

params = [p for p in draft.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(1, min(100, n_steps // 20))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, n_steps))))))
posw = torch.exp(-torch.arange(BS - 1, device=dev, dtype=torch.float32) / args.decay)


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def seq_loss(ids_list, P):
    ids = torch.tensor([ids_list], device=dev)
    N = ids.shape[1]
    with torch.no_grad():
        out = target.model(ids, output_hidden_states=True)
        feats = torch.cat([out.hidden_states[l + 1] for l in LIDS], -1).float()
    cand = list(range(P, N - 1))
    A = sorted(random.sample(cand, min(args.anchors, len(cand))))
    At = torch.tensor(A, device=dev)
    nb = len(A)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        th = draft.hidden_norm(draft.fc(feats))  # (1,N,H)
        cos_c, sin_c = draft.rotary_emb(th, torch.arange(N, device=dev).unsqueeze(0))
        blk = torch.full((nb, BS), MASK, dtype=torch.long, device=dev)
        blk[:, 0] = ids[0, At]
        gidx0 = At.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0)
        revealed = torch.zeros(nb, BS - 1, dtype=torch.bool, device=dev)
        if args.reveal_p > 0:
            for b in range(nb):
                if random.random() >= args.reveal_p:
                    continue
                if random.random() < 0.5:  # random subset (covers "reveal the word start only")
                    k = random.randint(1, 3)
                    pos = random.sample(range(BS - 2), k)
                    revealed[b, pos] = True
                else:  # left prefix of random length
                    revealed[b, : random.randint(1, BS - 2)] = True
            revealed &= gidx0 < N
            blk[:, 1:] = torch.where(revealed, ids[0, gidx0.clamp(max=N - 1)], blk[:, 1:])
        with torch.no_grad():
            h = target.model.embed_tokens(blk).float()
        h = h.reshape(1, nb * BS, -1)
        pos_n = (At.unsqueeze(1) + torch.arange(BS, device=dev).unsqueeze(0)).reshape(1, -1)
        cos_n, sin_n = draft.rotary_emb(h, pos_n)
        qb = torch.arange(nb, device=dev).repeat_interleave(BS)  # block id of each query
        m_ctx = torch.arange(N, device=dev).unsqueeze(0) < At[qb].unsqueeze(1)  # (nb*BS, N)
        m_blk = qb.unsqueeze(1) == qb.unsqueeze(0)  # (nb*BS, nb*BS)
        mask = torch.cat([m_ctx, m_blk], 1)[None, None]
        for layer in draft.layers:
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(1, -1, NH, HD)).transpose(1, 2)
            kn = at.k_norm(at.k_proj(x).view(1, -1, NKV, HD)).transpose(1, 2)
            vn = at.v_proj(x).view(1, -1, NKV, HD).transpose(1, 2)
            q, kn = rope(q, cos_n, sin_n), rope(kn, cos_n, sin_n)
            kc = rope(at.k_norm(at.k_proj(th).view(1, N, NKV, HD)).transpose(1, 2), cos_c, sin_c)
            vc = at.v_proj(th).view(1, N, NKV, HD).transpose(1, 2)
            K, V = torch.cat([kc, kn], 2), torch.cat([vc, vn], 2)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, scale=at.scaling, enable_gqa=True)
            h = h + at.o_proj(o.transpose(1, 2).reshape(1, -1, NH * HD))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hb = draft.norm(h).view(nb, BS, -1)[:, 1:, :]
        logits = target.lm_head(hb.to(torch.bfloat16)).float()  # (nb, BS-1, V)
    if os.environ.get("XCHECK"):
        # compare against the official DFlash forward (no cache, full context) for a few anchors
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            agree = tot = 0
            for b in range(min(6, nb)):
                s0 = A[b]
                blk1 = torch.full((1, BS), MASK, dtype=torch.long, device=dev)
                blk1[0, 0] = ids[0, s0]
                hid = draft(target_hidden=feats[:, :s0], noise_embedding=target.model.embed_tokens(blk1).float(),
                            position_ids=torch.arange(s0 + BS, device=dev).unsqueeze(0), is_causal=False)
                ref = target.lm_head(hid[:, 1:, :].to(torch.bfloat16)).argmax(-1)[0]
                agree += int((ref == logits[b].argmax(-1)).sum())
                tot += BS - 1
        print(f"XCHECK train-forward vs official argmax agreement: {agree}/{tot}", flush=True)
    gidx = At.unsqueeze(1) + torch.arange(1, BS, device=dev).unsqueeze(0)
    valid = gidx < N
    labels = ids[0, gidx.clamp(max=N - 1)]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view(nb, BS - 1)
    valid = valid & ~revealed
    w = posw.unsqueeze(0) * valid
    loss = (ce * w).sum() / w.sum()
    acc = ((logits.argmax(-1) == labels) & valid).float().sum(0) / valid.float().sum(0).clamp(min=1)
    return loss, acc


t0, it = time.time(), 0
ema_loss, ema_acc = None, None
for step in range(n_steps):
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for k in range(args.accum):
        ids_list, P = mine[(step * args.accum + k) % len(mine)]
        loss, acc = seq_loss(ids_list, P)
        (loss / args.accum).backward()
        tot += loss.item() / args.accum
        ema_acc = acc if ema_acc is None else 0.98 * ema_acc + 0.02 * acc
    if ddp:
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    ema_loss = tot if ema_loss is None else 0.98 * ema_loss + 0.02 * tot
    if rank == 0 and (step % 25 == 0 or step == n_steps - 1):
        el = time.time() - t0
        print(f"step {step}/{n_steps} loss {ema_loss:.4f} acc@1..4 {[round(x, 3) for x in ema_acc[:4].tolist()]} "
              f"lr {sched.get_last_lr()[0]:.2e} {el/60:.1f}min eta {el/(step+1)*(n_steps-step-1)/60:.1f}min", flush=True)

if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    draft.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    for f in glob.glob(f"{args.init}/*.py") + [f"{args.init}/config.json"]:
        shutil.copy(f, args.out)
    open(f"{args.out}/TRAINED_OK", "w").write(json.dumps(vars(args)))
    print("saved", args.out, flush=True)
if ddp:
    dist.barrier()
    dist.destroy_process_group()
