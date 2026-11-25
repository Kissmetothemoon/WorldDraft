"""Real speculative decoding (HF, batch 1) with DFlash + lexical prior: greedy and temperature sampling, lossless.

Per round the drafter gives logits for the block; the draft is built left to right over the block positions:
candidates = drafter top-16 + the lexicon's top-E continuations of the CURRENT word (accepted prefix + tokens already
drafted in this block), score = log q_drafter / T + lam * log p_lex, and
  greedy  : take the argmax (target verifies with argmax)
  sample  : draw from q_t = softmax(score) over the candidates (explicit draft distribution) and verify with standard
            speculative sampling (accept w.p. min(1, p/q), else resample from norm(max(p - q, 0))) -> output ~ target.
Reports tau and wall time per prompt, and the time spent in the lexical walk.
"""
import argparse
import glob
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np
import torch

from common import MAX_NEW, build_messages, chat_ids, jload, pdump, shard, token_byte_table

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--draft", required=True)
ap.add_argument("--lex", required=True)
ap.add_argument("--lex-max-tokens", type=float, default=1e7)
ap.add_argument("--lam", type=float, default=0.75)
ap.add_argument("--off-langs", default="en,de,fr,es,zh", help="languages where the prior is switched off")
ap.add_argument("--prompts", required=True)
ap.add_argument("--clean-ids", required=True)
ap.add_argument("--per-group", type=int, default=12)
ap.add_argument("--temps", default="0,1")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="speed", choices=["speed", "lossless"])
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--lossless-max-new", type=int, default=48)
ap.add_argument("--save-seq", action="store_true", help="also store generated ids (output-identity check)")
args = ap.parse_args()

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.target)
bt = token_byte_table(tok)
target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
draft = AutoModel.from_pretrained(args.draft, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=dev).eval()
LIDS, BS, MASK = draft.target_layer_ids, draft.block_size, draft.mask_token_id
VOC = target.lm_head.out_features
TOPK, E, LX = 16, 4, 32
END = -1
CONT = np.zeros(VOC, dtype=bool)
for t, bs in enumerate(bt):
    if bs:
        CONT[t] = (bs[0] >= 0x80) or chr(bs[0]).isalnum()


def starts_word(t):
    return CONT[t] or (bt[t][:1] == b" " and len(bt[t]) > 1)


# ---------------- lexicon
t0 = time.time()
C = defaultdict(Counter)
used = Counter()
for f in sorted(glob.glob(args.lex)):
    for r in jload(f):
        lang = r.get("lang") or "en"
        if used[lang] >= args.lex_max_tokens:
            continue
        out = r["output_ids"]
        used[lang] += len(out)
        word = []
        for t in out:
            if CONT[t] and word:
                for n in range(1, 4):
                    if len(word) >= n:
                        C[(lang, tuple(word[-n:]))][t] += 1
                word.append(t)
            else:
                if word:
                    for n in range(1, 4):
                        if len(word) >= n:
                            C[(lang, tuple(word[-n:]))][END] += 1
                word = [t] if starts_word(t) else []
CACHE = {}


def lex_dist(lang, word):
    key = (lang, tuple(word[-3:]))
    if key in CACHE:
        return CACHE[key]
    res = None
    for n in (3, 2, 1):
        if len(word) >= n:
            c = C.get((lang, tuple(word[-n:])))
            if c is not None:
                tot = sum(c.values())
                if tot >= 2:
                    items = [(k, v) for k, v in c.most_common(LX + 1) if k != END][:LX]
                    res = ({k: math.log(v / tot) for k, v in items}, math.log((c.get(END, 0) + 0.5) / (tot + 1)),
                           math.log(0.5 / (tot + len(c))), [k for k, _ in items[:E]])
                    break
    CACHE[key] = res
    return res


print(f"lexicon {len(C)} contexts in {time.time()-t0:.0f}s", flush=True)
OFF = set(args.off_langs.split(","))


def word_of(seq):
    """current word (token list) at the end of seq"""
    w = []
    for t in reversed(seq[-12:]):
        w.append(t)
        if not CONT[t]:
            break
    w.reverse()
    if not w or not starts_word(w[0]):
        return []
    return w


@torch.inference_mode()
def generate(input_ids, max_new, stop_ids, lang, lam, T):
    use_lex = lam > 0 and lang not in OFF
    n_in = input_ids.shape[1]
    max_len = n_in + max_new
    out = torch.full((1, max_len + BS), MASK, dtype=torch.long, device=dev)
    pos = torch.arange(out.shape[1], device=dev).unsqueeze(0)
    kv_t, kv_d = DynamicCache(), DynamicCache()
    o = target(input_ids, position_ids=pos[:, :n_in], past_key_values=kv_t, use_cache=True, logits_to_keep=1, output_hidden_states=True)
    out[:, :n_in] = input_ids
    if T == 0:
        out[:, n_in] = o.logits[0, -1].argmax()
    else:
        out[:, n_in] = torch.multinomial(torch.softmax(o.logits[0, -1].float() / T, -1), 1)[0]
    th = torch.cat([o.hidden_states[l + 1] for l in LIDS], -1)
    seq = input_ids[0].tolist() + [int(out[0, n_in])]
    accs, start, t_lex = [], n_in, 0.0
    while start < max_len:
        b = out[:, start:start + BS].clone()
        dl = target.lm_head(draft(target_hidden=th, noise_embedding=target.model.embed_tokens(b),
                                  position_ids=pos[:, kv_d.get_seq_length():start + BS], past_key_values=kv_d,
                                  use_cache=True, is_causal=False)[:, -BS + 1:, :])[0].float()
        kv_d.crop(start)
        lq = torch.log_softmax(dl / max(T, 1e-6) if T > 0 else dl, -1)  # (BS-1, V)
        qs = []  # full-vocab draft distributions (sampling only)
        if not use_lex:
            if T == 0:
                b[0, 1:] = dl.argmax(-1)
            else:
                q = lq.exp()
                b[0, 1:] = torch.multinomial(q, 1)[:, 0]
                qs = q
        else:
            tl0 = time.time()
            tv, ti = lq.topk(64, -1)
            tv, ti = tv.tolist(), ti.tolist()  # one device->host copy per round
            ctx = list(seq)
            drafted, qrows = [], []
            for k in range(BS - 1):
                lpk = dict(zip(ti[k], tv[k]))
                floor_k = tv[k][-1]
                w = word_of(ctx)
                d = lex_dist(lang, w) if w else None
                cand = ti[k][:TOPK]
                if d:
                    extra = [x for x in d[3] if x not in cand]
                    cand = cand + extra
                    sc = [lpk.get(c, floor_k) + lam * (d[0].get(c, d[2]) if CONT[c] else d[1]) for c in cand]
                else:
                    sc = [lpk[c] for c in cand]
                if T == 0:
                    x = cand[max(range(len(cand)), key=sc.__getitem__)]
                else:
                    m = max(sc)
                    e = [math.exp(v - m) for v in sc]
                    z = sum(e)
                    pr = [v / z for v in e]
                    x = cand[int(np.random.choice(len(cand), p=pr))]
                    qrows.append((cand, pr))
                drafted.append(x)
                ctx.append(x)
            b[0, 1:] = torch.tensor(drafted, device=dev)
            t_lex += time.time() - tl0
        o = target(b, position_ids=pos[:, start:start + BS], past_key_values=kv_t, use_cache=True, output_hidden_states=True)
        if T == 0:
            post = o.logits[0].argmax(-1)
            a = int((b[0, 1:] == post[:-1]).cumprod(0).sum().item())
            nxt = int(post[a])
        else:
            p = torch.softmax(o.logits[0].float() / T, -1)  # (BS, V): p[k] = target dist for position k+1
            a, nxt = 0, None
            pc = p.cpu() if use_lex else None
            for k in range(BS - 1):
                x = int(b[0, k + 1])
                if use_lex:
                    cand, pr = qrows[k]
                    qx = pr[cand.index(x)]
                    px = float(pc[k, x])
                else:
                    qx, px = float(qs[k, x]), float(p[k, x])
                if np.random.rand() < min(1.0, px / max(qx, 1e-20)):
                    a += 1
                    continue
                if use_lex:
                    qrow = torch.zeros(VOC)
                    qrow[torch.tensor(cand)] = torch.tensor(pr)
                    res = (pc[k] - qrow).clamp(min=0).to(dev)
                else:
                    res = (p[k] - qs[k]).clamp(min=0)
                nxt = int(torch.multinomial(res / res.sum(), 1)) if res.sum() > 0 else int(torch.multinomial(p[k], 1))
                break
            if nxt is None:
                nxt = int(torch.multinomial(p[BS - 1], 1))
        out[:, start:start + a + 1] = b[:, :a + 1]
        out[:, start + a + 1] = nxt
        seq = seq[: start + 1] if False else seq
        seq = out[0, :start + a + 2].tolist()
        start += a + 1
        kv_t.crop(start)
        th = torch.cat([o.hidden_states[l + 1] for l in LIDS], -1)[:, :a + 1, :]
        accs.append(a + 1)
        if any(s in out[0, n_in:start].tolist() for s in stop_ids):
            break
    return accs, start - n_in, t_lex, out[0, n_in:start].tolist()


stop = sorted({tok.eos_token_id} | {t for t in [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")]
                                    if isinstance(t, int) and 0 <= t < len(tok) and t != tok.unk_token_id})
keep = set(json.load(open(args.clean_ids)))
groups = defaultdict(list)
for r in jload(args.prompts):
    if r["id"] in keep:
        groups[(r["task"], r["lang"])].append(r)
sel = shard([r for k in sorted(groups) for r in groups[k][: args.per_group]], args.shard, args.nshard)
temps = [float(x) for x in args.temps.split(",")]
if args.mode == "lossless":
    # empirical check that speculative sampling with the WorldDraft proposal leaves the output distribution unchanged
    rep = []
    for i, r in enumerate(sel):
        pid = torch.tensor([chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))], device=dev)
        rec = {"id": r["id"], "task": r["task"], "lang": r["lang"], "m": {"target": [], "base": [], "lex": []}}
        for j in range(args.samples):
            torch.manual_seed(10_000 * i + j)
            np.random.seed(10_000 * i + j)
            g = target.generate(pid, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=args.lossless_max_new,
                                eos_token_id=stop, pad_token_id=stop[0])[0, pid.shape[1]:].tolist()
            rec["m"]["target"].append(g)
            for name, lam in (("base", 0.0), ("lex", args.lam)):
                torch.manual_seed(20_000 * i + j)
                np.random.seed(20_000 * i + j)
                _, _, _, seq = generate(pid, args.lossless_max_new, stop, r["lang"], lam, 1.0)
                rec["m"][name].append(seq[: args.lossless_max_new])
        # target log-likelihood of every sample (one teacher-forced pass per sample)
        rec["nll"] = {}
        for name, seqs in rec["m"].items():
            v = []
            for q in seqs:
                if not q:
                    continue
                ids = torch.tensor([pid[0].tolist() + q], device=dev)
                with torch.inference_mode():
                    lg = target(ids).logits[0, pid.shape[1] - 1:-1].float()
                v.append(float(-torch.log_softmax(lg, -1).gather(-1, ids[0, pid.shape[1]:, None]).mean()))
            rec["nll"][name] = v
        rep.append(rec)
        if i % 5 == 0:
            print(i, r["id"], {k: round(float(np.mean(v)), 3) for k, v in rec["nll"].items()}, flush=True)
    pdump(rep, args.out)
    print("done", len(rep), flush=True)
    raise SystemExit
warm = torch.tensor([chat_ids(tok, build_messages("chat", "en", "Hello!"))], device=dev)
generate(warm, 32, stop, "en", 0.0, 0.0)
rep = []
for i, r in enumerate(sel):
    pid = torch.tensor([chat_ids(tok, build_messages(r["task"], r["lang"], r["text"]))], device=dev)
    mx = MAX_NEW[r["task"]]
    rec = {"id": r["id"], "task": r["task"], "lang": r["lang"], "m": {}}
    for T in temps:
        for name, lam in (("base", 0.0), ("lex", args.lam)):
            torch.manual_seed(1234 + i)
            np.random.seed(1234 + i)
            torch.cuda.synchronize()
            t = time.time()
            accs, n, tl, seq = generate(pid, mx, stop, r["lang"], lam, T)
            torch.cuda.synchronize()
            rec["m"][f"{name}_T{T:g}"] = {"accs": accs, "n": n, "t": time.time() - t, "t_lex": tl}
            if args.save_seq:
                rec["m"][f"{name}_T{T:g}"]["seq"] = seq
    rep.append(rec)
    if i % 10 == 0:
        print(i, r["id"], {k: round(sum(v["accs"]) / len(v["accs"]), 2) for k, v in rec["m"].items()}, flush=True)
pdump(rep, args.out)
print("done", len(rep), flush=True)
