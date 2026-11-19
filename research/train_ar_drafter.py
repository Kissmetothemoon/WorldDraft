"""Distil a small autoregressive draft model on target-generated responses (sequence-level KD, response tokens only).

Matched counterpart of train_dflash.py for the AR-vs-block comparison: same data files, same epochs.
torchrun-style: python -m torch.distributed.run --nproc_per_node 4 train_ar_drafter.py --model ... --data 'glob' --out ...
"""
import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--max-seqs", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--grad-ckpt", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(args.seed)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
if args.grad_ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # reentrant ckpt breaks DDP
model.config.use_cache = False
model.train()
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
core = model.module if ddp else model

rows = []
for f in sorted(glob.glob(args.data)):
    with open(f, encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            if len(r["output_ids"]) < 8:
                continue
            ids = (r["prompt_ids"] + r["output_ids"])[: args.max_len]
            rows.append((ids, len(r["prompt_ids"])))
random.Random(args.seed).shuffle(rows)
if args.max_seqs:
    rows = rows[: args.max_seqs]
mine = rows[rank::world]
n_steps = int(len(mine) * args.epochs) // args.accum
if args.max_steps:
    n_steps = min(n_steps, args.max_steps)
if ddp:  # equal step counts on all ranks (a mismatch deadlocks the collectives at the end)
    t = torch.tensor([n_steps], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    n_steps = int(t.item())
if rank == 0:
    print(f"{len(rows)} sequences, {world} GPUs, {n_steps} optimizer steps", flush=True)

opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(1, min(100, n_steps // 20))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, n_steps))))))

t0, ema = time.time(), None
for step in range(n_steps):
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for k in range(args.accum):
        ids, P = mine[(step * args.accum + k) % len(mine)]
        x = torch.tensor([ids], device=dev)
        labels = x.clone()
        labels[:, :P] = -100
        ctx = model.no_sync() if (ddp and k < args.accum - 1) else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=labels).loss
            (loss / args.accum).backward()
        tot += loss.item() / args.accum
    torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
    opt.step()
    sched.step()
    ema = tot if ema is None else 0.98 * ema + 0.02 * tot
    if rank == 0 and (step % 50 == 0 or step == n_steps - 1):
        el = time.time() - t0
        print(f"step {step}/{n_steps} loss {ema:.4f} lr {sched.get_last_lr()[0]:.2e} {el/60:.1f}min "
              f"eta {el/(step+1)*(n_steps-step-1)/60:.1f}min", flush=True)

if rank == 0:
    os.makedirs(args.out, exist_ok=True)
    core.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    open(f"{args.out}/TRAINED_OK", "w").write(json.dumps(vars(args)))
    print("saved", args.out, flush=True)
if ddp:
    dist.barrier()
    dist.destroy_process_group()
