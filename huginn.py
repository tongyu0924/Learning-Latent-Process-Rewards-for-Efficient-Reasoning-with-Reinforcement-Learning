"""Huginn loading + patches: left-padded batched attention, MC-dropout in the core, latent capture."""
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_ID = 65509
STOP_IDS = {65504, 65505, 65508}
SYSTEM = "You are a helpful assistant."

model = None
tokenizer = None
_rope = None
_orig_core = None
_PAD = {"mask": None}
_DROP = {"p": 0.0}
_CAP = {"on": False, "buf": []}


# ---------------- patches ----------------
def _attn_forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
    B, S, E = x.shape
    q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
    q = q.view(B, S, self.n_head, self.head_dim)
    k = k.view(B, S, self.n_kv_heads, self.head_dim)
    v = v.view(B, S, self.n_kv_heads, self.head_dim)
    if self.config.qk_bias:
        qb, kb = self.qk_bias.split(1, dim=0)
        q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
    q, k = _rope(q, k, freqs_cis=freqs_cis)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, block_idx)
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    L = k.shape[2]
    qpos = torch.arange(L - S, L, device=q.device)
    kpos = torch.arange(L, device=q.device)
    kpm = _PAD["mask"]
    if kpm is None:
        if S == L:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=(kpos[None] <= qpos[:, None])[None, None])
    else:
        m = (kpos[None] <= qpos[:, None])[None, None] & kpm[:, :L][:, None, None, :]
        m = m | (kpos[None] == qpos[:, None])[None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    return self.proj(y.transpose(1, 2).reshape(B, S, E))


def _dropout_hook(module, inputs, output):
    return F.dropout(output, _DROP["p"], training=True) if _DROP["p"] > 0 else output


def _capturing_core(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
    out, bidx = _orig_core(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step)
    if _CAP["on"]:
        if int(current_step) == 0:
            _CAP["buf"].append([x.detach()])
        _CAP["buf"][-1].append(out.detach())
    return out, bidx


def load_model(model_id):
    global model, tokenizer, _rope, _orig_core
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if tokenizer.eos_token_id is not None:
        STOP_IDS.add(tokenizer.eos_token_id)

    _rope = sys.modules[type(model).__module__].apply_rotary_emb_complex_like
    type(model.transformer.core_block[0].attn).forward = _attn_forward
    for blk in model.transformer.core_block:
        blk.attn.register_forward_hook(_dropout_hook)
        blk.mlp.register_forward_hook(_dropout_hook)
    _orig_core = model.core_block_forward
    model.core_block_forward = _capturing_core
    return model, tokenizer


# ---------------- context managers ----------------
@contextmanager
def pad_mask(mask):
    _PAD["mask"] = mask
    try:
        yield
    finally:
        _PAD["mask"] = None


@contextmanager
def mc_dropout(p):
    old, _DROP["p"] = _DROP["p"], p
    try:
        yield
    finally:
        _DROP["p"] = old


@contextmanager
def fixed_rng(seed):
    devs = [torch.cuda.current_device()] if DEVICE == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        yield


class RecurrentKVCache:
    """One KV slot per block / recurrence step."""
    def __init__(self):
        self.k, self.v = {}, {}

    def update(self, k, v, step_idx, lookup_strategy=None):
        i = int(step_idx)
        if i in self.k:
            self.k[i] = torch.cat([self.k[i], k], 2)
            self.v[i] = torch.cat([self.v[i], v], 2)
        else:
            self.k[i], self.v[i] = k, v
        return self.k[i], self.v[i]

    def get_seq_length(self, step_idx=0):
        return self.k[0].shape[2] if 0 in self.k else 0


# ---------------- core ops ----------------
def core_step(s, e, freqs, k):
    """One recurrent transition s_k = R(e, s_{k-1}) without caching."""
    return _orig_core(s, e, freqs, None, None, torch.tensor(1, dtype=torch.long), k)[0]


@torch.no_grad()
def prelude_embed(ids):
    freqs = model.freqs_cis[:, :ids.shape[1]]
    x = model.transformer.wte(ids)
    if model.emb_scale != 1:
        x = x * model.emb_scale
    bidx = torch.tensor(-1, dtype=torch.long)
    for blk in model.transformer.prelude:
        bidx += 1
        x = blk(x, freqs, bidx, None, None)
    return x, freqs


# ---------------- generation ----------------
def build_prompt(text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(s, add_special_tokens=False)


def decode(tok_ids):
    ids = [t for t in tok_ids if t not in STOP_IDS and t != PAD_ID]
    return tokenizer.decode(ids, skip_special_tokens=True).split("<|")[0].strip()


def _sample(logits, temperature, top_p):
    probs = torch.softmax(logits.float() / max(temperature, 1e-5), -1)
    sp, si = probs.sort(-1, descending=True)
    sp[(sp.cumsum(-1) - sp) > top_p] = 0
    return si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))


@torch.no_grad()
def generate(prompts, num_steps, max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0, capture=False):
    """Batched left-padded generation. With capture=True also returns the latent trajectory
    states[i]: [P_i + n_i - 1, r+1, d] on CPU (positions = input tokens, depths 0..r)."""
    B, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    valid = torch.zeros((B, L), dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p)
        valid[i, L - len(p):] = True
    ids, valid = ids.to(DEVICE), valid.to(DEVICE)
    cache, outs, done, step_states = RecurrentKVCache(), [[] for _ in range(B)], [False] * B, []
    cur, pos = ids, torch.arange(L, device=DEVICE)
    for t in range(max_new_tokens):
        _CAP["on"], _CAP["buf"] = capture, []
        try:
            with pad_mask(valid):
                out = model(input_ids=cur, num_steps=num_steps, past_key_values=cache, use_cache=True,
                            cache_position=pos, output_details={"return_logits": True, "return_latents": False,
                                                                "return_head": False, "return_stats": False})
        finally:
            _CAP["on"] = False
        if capture:
            step_states.append(torch.stack(_CAP["buf"][-1], 2).cpu())
        logits = out.logits[:, -1]
        del out
        nxt = _sample(logits, temperature, top_p) if do_sample else logits.argmax(-1, keepdim=True)
        for i, tok in enumerate(nxt[:, 0].tolist()):
            if not done[i]:
                outs[i].append(tok)
                done[i] = tok in STOP_IDS
        if all(done):
            break
        cur = nxt
        valid = torch.cat([valid, torch.ones((B, 1), dtype=torch.bool, device=DEVICE)], 1)
        pos = torch.tensor([L + t], device=DEVICE)
    if not capture:
        return outs
    allS = torch.cat(step_states, 1)
    states = [allS[i, L - len(p):L + len(outs[i]) - 1].clone() for i, p in enumerate(prompts)]
    return outs, states


def generate_chunked(prompts, num_steps, max_new_tokens, batch_size, capture=False, **kw):
    bs = max(1, batch_size * 8 // max(8, num_steps))
    outs, states = [], []
    for i in range(0, len(prompts), bs):
        res = generate(prompts[i:i + bs], num_steps, max_new_tokens, capture=capture, **kw)
        if capture:
            outs += res[0]
            states += res[1]
        else:
            outs += res
    return (outs, states) if capture else outs
