"""Greedy speculative decoding loops used for wall-clock measurements.

dflash_generate : DFlash block drafting (copy of the official spec_generate, returns acceptance lengths)
ar_sd_generate  : vanilla speculative decoding with an independent AR draft model (chain of K greedy drafts)
Both are lossless under greedy decoding (outputs equal target greedy decoding up to bf16 near-ties).
"""
import torch
from transformers import DynamicCache


@torch.inference_mode()
def dflash_generate(target, draft, input_ids, max_new_tokens, stop_ids):
    dev = input_ids.device
    lids, bs, mask = draft.target_layer_ids, draft.block_size, draft.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    out = torch.full((1, max_len + bs), mask, dtype=torch.long, device=dev)
    pos = torch.arange(out.shape[1], device=dev).unsqueeze(0)
    kv_t, kv_d = DynamicCache(), DynamicCache()
    o = target(input_ids, position_ids=pos[:, :n_in], past_key_values=kv_t, use_cache=True, logits_to_keep=1,
               output_hidden_states=True)
    out[:, :n_in] = input_ids
    out[:, n_in] = o.logits.argmax(-1)[0, -1]
    th = torch.cat([o.hidden_states[l + 1] for l in lids], -1)
    accs, start = [], n_in
    while start < max_len:
        b = out[:, start:start + bs].clone()
        dl = target.lm_head(draft(target_hidden=th, noise_embedding=target.model.embed_tokens(b),
                                  position_ids=pos[:, kv_d.get_seq_length():start + bs], past_key_values=kv_d,
                                  use_cache=True, is_causal=False)[:, -bs + 1:, :])
        kv_d.crop(start)
        b[:, 1:] = dl.argmax(-1)
        o = target(b, position_ids=pos[:, start:start + bs], past_key_values=kv_t, use_cache=True,
                   output_hidden_states=True)
        post = o.logits.argmax(-1)
        a = int((b[:, 1:] == post[:, :-1]).cumprod(1).sum(1)[0].item())
        out[:, start:start + a + 1] = b[:, :a + 1]
        out[:, start + a + 1] = post[:, a]
        start += a + 1
        kv_t.crop(start)
        th = torch.cat([o.hidden_states[l + 1] for l in lids], -1)[:, :a + 1, :]
        accs.append(a + 1)
        if any(s in out[0, n_in:start].tolist() for s in stop_ids):
            break
    return _trim(out[0, n_in:start].tolist(), stop_ids), accs


@torch.inference_mode()
def ar_sd_generate(target, draft, input_ids, max_new_tokens, stop_ids, K):
    """seq = accepted tokens; target cache covers seq[:-1] after each round (last token = bonus, unfed);
    the draft cache covers a prefix of seq and is fed whatever it has not seen yet."""
    dev = input_ids.device
    seq = input_ids[0].tolist()
    n_in = len(seq)
    kv_t, kv_d = DynamicCache(), DynamicCache()
    o = target(input_ids, past_key_values=kv_t, use_cache=True, logits_to_keep=1)
    seq.append(int(o.logits[0, -1].argmax()))
    accs = []
    while len(seq) - n_in < max_new_tokens:
        # draft K tokens greedily
        drafts = []
        feed = seq[kv_d.get_seq_length():]
        for _ in range(K):
            d = draft(torch.tensor([feed], device=dev), past_key_values=kv_d, use_cache=True, logits_to_keep=1)
            nxt = int(d.logits[0, -1].argmax())
            drafts.append(nxt)
            feed = [nxt]
        # verify: target sees the unfed bonus token + K drafts
        vt = torch.tensor([[seq[-1]] + drafts], device=dev)
        o = target(vt, past_key_values=kv_t, use_cache=True)
        pred = o.logits[0].argmax(-1).tolist()  # pred[i] = target token after (bonus, drafts[:i])
        a = 0
        while a < K and drafts[a] == pred[a]:
            a += 1
        seq += drafts[:a] + [pred[a]]
        accs.append(a + 1)
        kv_t.crop(len(seq) - 1)
        kv_d.crop(min(kv_d.get_seq_length(), len(seq) - 1))
        if any(s in seq[n_in:] for s in stop_ids):
            break
    return _trim(seq[n_in:n_in + max_new_tokens], stop_ids), accs


def _trim(gen, stop_ids):
    for k, t in enumerate(gen):
        if t in stop_ids:
            return gen[:k + 1]
    return gen
