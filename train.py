"""Reward-based latent RL on Huginn: SLPO surrogate + RLOO / latent-value baseline + step cost."""
import argparse
import json
import math
import os
import random
import time

import torch

import huginn as H
from config import get_config
from data import GSM_SUFFIX, extract_pred_number, load_eval_sets, load_gsm8k, norm_num, num_equal, score_item

CFG = None
DEVICE = H.DEVICE
LOG2PI = math.log(2 * math.pi)

VARIANTS = {
    "outcome":         dict(name="RL-Outcome",  kind="grpo", baseline="mean",  lam=0.0),
    "slpo":            dict(name="SLPO",        kind="slpo", baseline="rloo",  lam=0.0),
    "slpo_value":      dict(name="SLPO+V",      kind="slpo", baseline="value", lam=0.0),
    "slpo_value_cost": dict(name="SLPO+V+cost", kind="slpo", baseline="value", lam=None),  # None -> cfg
}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------- reward ----------------
def correct_reward(text, gold):
    return 1.0 if num_equal(extract_pred_number(text), norm_num(gold)) else 0.0


def shaped_reward(correct, n_tokens, lam):
    # r' = r - lam * T / T_max,  T = n_tokens * r
    return correct - lam * (n_tokens * CFG.r) / (CFG.train_max_new_tokens * CFG.r)


def advantages(R, kind):
    G = len(R)
    if kind == "rloo" and G > 1:
        tot = sum(R)
        return [x - (tot - x) / (G - 1) for x in R]
    m = sum(R) / G
    return [x - m for x in R]


class LatentValueHead(torch.nn.Module):
    """V(x, z_{<t}) from s_{t-1, r}; used only as a baseline."""
    def __init__(self, d, h):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, h), torch.nn.GELU(),
                                       torch.nn.Linear(h, 1))

    def forward(self, s):
        return self.net(s.float()).squeeze(-1)


# ---------------- SLPO ----------------
def _pack(prompt_ids, comps, states):
    P, r = len(prompt_ids), CFG.r
    T, b = P + max(len(c) for c in comps) - 1, len(comps)
    ids = torch.full((b, T), H.PAD_ID, dtype=torch.long)
    tgt = torch.full((b, T), H.PAD_ID, dtype=torch.long)
    mask = torch.zeros((b, T))
    S = torch.zeros((b, T, r + 1, states[0].shape[-1]), dtype=states[0].dtype)
    for i, c in enumerate(comps):
        seq, n = prompt_ids + c, P + len(c) - 1
        ids[i, :n], tgt[i, :n] = torch.tensor(seq[:-1]), torch.tensor(seq[1:])
        mask[i, P - 1:n] = 1.0
        S[i, :n] = states[i][:n]
    return ids.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE), S


def _value_feats(S):
    f = S[:, :, CFG.r].to(DEVICE)
    return torch.cat([f[:, :1], f[:, :-1]], 1)          # position t sees s_{t-1, r}


def _value_step(vhead, vopt, feats, Rp, mask):
    loss = (((vhead(feats) - Rp[:, None]) ** 2) * mask).sum() / mask.sum()
    vopt.zero_grad(set_to_none=True)
    loss.backward()
    vopt.step()
    return float(loss)


def slpo_backward(prompt_ids, comps, states, Rp, A_rloo, n_norm, vhead=None, vopt=None, use_value=False):
    ids, tgt, mask, S = _pack(prompt_ids, comps, states)
    b, T = ids.shape
    denom = mask.sum(1).clamp(min=1)
    Rp = torch.tensor(Rp, device=DEVICE)
    out = {"vloss": 0.0}

    # advantage A[b, T]
    if vhead is not None:
        feats = _value_feats(S)
        with torch.no_grad():
            vb = vhead(feats)
        out["vloss"] = _value_step(vhead, vopt, feats, Rp, mask)
    if use_value:
        adv = Rp[:, None] - vb
        out["base"] = float((vb * mask).sum() / mask.sum())
    else:
        A = torch.tensor(A_rloo, device=DEVICE)
        adv = A[:, None].expand(b, T)
        out["base"] = float((Rp - A).mean())

    e, freqs = H.prelude_embed(ids)
    lat_log, tok_log = 0.0, 0.0
    for k in range(1, CFG.r + 1):
        x_in = S[:, :, k - 1].to(DEVICE)
        h = S[:, :, k].to(DEVICE).float()
        with H.mc_dropout(CFG.slpo_dropout):
            Z = torch.stack([H.core_step(x_in, e, freqs, k - 1).float() for _ in range(CFG.slpo_K)])
        mu, var = Z.mean(0), Z.var(0, unbiased=False) + CFG.slpo_var_eps
        dens = -0.5 * ((h - mu) ** 2 / var + torch.log(var) + LOG2PI)
        dens = dens.mean(-1) if CFG.slpo_dim_mean else dens.sum(-1)
        term = CFG.slpo_lambda_latent * dens
        lat_log += float(((dens * mask).sum(1) / denom).mean())

        if k == CFG.r and CFG.slpo_token_term:
            logits = H.model.predict_from_latents(mu.to(x_in.dtype)).logits
            tlp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
            term = term + tlp
            tok_log = float(((tlp * mask).sum(1) / denom).mean())
            del logits, tlp

        loss = -((adv * term * mask).sum(1) / denom).sum() / n_norm
        loss.backward()
        del Z, mu, var, dens, h, x_in, loss, term
    out.update(lat=lat_log / CFG.r, tok=tok_log)
    return out


def value_only(prompt_ids, comps, states, Rp, vhead, vopt):
    _, _, mask, S = _pack(prompt_ids, comps, states)
    return _value_step(vhead, vopt, _value_feats(S), torch.tensor(Rp, device=DEVICE), mask)


# ---------------- token-level GRPO baseline ----------------
def token_logps(prompt_ids, comps, num_steps, seed):
    P, T = len(prompt_ids), len(prompt_ids) + max(len(c) for c in comps)
    ids = torch.full((len(comps), T), H.PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(comps), T - 1))
    for i, c in enumerate(comps):
        ids[i, :P + len(c)] = torch.tensor(prompt_ids + c)
        mask[i, P - 1:P - 1 + len(c)] = 1.0
    ids, mask = ids.to(DEVICE), mask.to(DEVICE)
    with H.fixed_rng(seed):
        logits = H.model(input_ids=ids, num_steps=num_steps).logits[:, :-1]
    lp = logits.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return lp, mask


def grpo_backward(peft_model, prompt_ids, comps, adv, n_norm, beta):
    seed = random.randint(0, 2 ** 31 - 1)
    adv = torch.tensor(adv, device=DEVICE).unsqueeze(1)
    with torch.no_grad(), peft_model.disable_adapter():
        ref_lp, _ = token_logps(prompt_ids, comps, CFG.r, seed)
    lp, mask = token_logps(prompt_ids, comps, (0, CFG.r), seed)
    ratio = torch.exp(lp - lp.detach())
    surr = torch.min(ratio * adv, ratio.clamp(1 - CFG.clip_eps, 1 + CFG.clip_eps) * adv)
    d = ref_lp - lp
    kl = torch.exp(d) - d - 1
    seq = ((-surr + beta * kl) * mask).sum(1) / mask.sum(1).clamp(min=1)
    (seq.sum() / n_norm).backward()
    return float(((kl * mask).sum() / mask.sum()).detach())


# ---------------- training loop ----------------
def train_variant(peft_model, variant, train_items):
    spec = VARIANTS[variant]
    is_slpo = spec["kind"] == "slpo"
    lam = CFG.step_cost_lambda if spec["lam"] is None else spec["lam"]
    vhead = vopt = None
    if spec["baseline"] == "value":
        vhead = LatentValueHead(H.model.config.n_embd, CFG.value_hidden).to(DEVICE)
        vopt = torch.optim.AdamW(vhead.parameters(), lr=CFG.value_lr)

    peft_model.set_adapter(variant)
    params = [p for n, p in peft_model.named_parameters() if p.requires_grad and f".{variant}." in n]
    log(f"### {variant} | baseline={spec['baseline']} lambda={lam} | trainable={sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=0.0)
    beta, hist, step = CFG.beta_init, [], 0
    per_update = CFG.prompts_per_micro * CFG.grad_accum
    H.model.gradient_checkpointing = not is_slpo
    avg = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731

    for epoch in range(CFG.num_epochs):
        order = random.sample(range(len(train_items)), len(train_items))
        for u0 in range(0, len(order), per_update):
            t0 = time.time()
            upd = [train_items[i] for i in order[u0:u0 + per_update]]
            n_norm = len(upd) * CFG.G
            use_value = vhead is not None and step >= CFG.value_warmup_updates
            st = {k: [] for k in ("R", "acc", "T", "kl", "lat", "tok", "vloss", "base")}
            opt.zero_grad(set_to_none=True)

            for m0 in range(0, len(upd), CFG.prompts_per_micro):
                micro = upd[m0:m0 + CFG.prompts_per_micro]
                p_ids = [H.build_prompt(it["question"] + GSM_SUFFIX) for it in micro]
                flat = [p for p in p_ids for _ in range(CFG.G)]
                kw = dict(do_sample=True, temperature=CFG.temperature, top_p=CFG.top_p)
                if is_slpo:
                    with H.mc_dropout(CFG.slpo_dropout):
                        comps_all, states_all = H.generate_chunked(flat, CFG.r, CFG.train_max_new_tokens,
                                                                   CFG.rollout_batch_size, capture=True, **kw)
                else:
                    comps_all = H.generate_chunked(flat, CFG.r, CFG.train_max_new_tokens,
                                                   CFG.rollout_batch_size, **kw)
                    states_all = None

                for j, it in enumerate(micro):
                    sl = slice(j * CFG.G, (j + 1) * CFG.G)
                    comps = comps_all[sl]
                    corr = [correct_reward(H.decode(c), it["answer"]) for c in comps]
                    R = [shaped_reward(corr[g], len(comps[g]), lam) for g in range(CFG.G)]
                    A = advantages(R, "mean" if spec["kind"] == "grpo" else "rloo")
                    st["R"] += R
                    st["acc"] += corr
                    st["T"] += [len(c) * CFG.r for c in comps]

                    nonempty = [g for g in range(CFG.G) if comps[g]]
                    keep = nonempty if use_value else [g for g in nonempty if abs(A[g]) > 1e-8]
                    for s0 in range(0, len(nonempty if not keep else keep), CFG.loss_micro_bs):
                        if not keep:   # no policy signal: still fit V
                            if vhead is None:
                                break
                            gi = nonempty[s0:s0 + CFG.loss_micro_bs]
                            st["vloss"].append(value_only(p_ids[j], [comps[g] for g in gi],
                                                          [states_all[sl][g] for g in gi],
                                                          [R[g] for g in gi], vhead, vopt))
                            continue
                        gi = keep[s0:s0 + CFG.loss_micro_bs]
                        cb, ab = [comps[g] for g in gi], [A[g] for g in gi]
                        if is_slpo:
                            o = slpo_backward(p_ids[j], cb, [states_all[sl][g] for g in gi],
                                              [R[g] for g in gi], ab, n_norm, vhead, vopt, use_value)
                            for key in ("lat", "tok", "vloss", "base"):
                                st[key].append(o[key])
                        else:
                            st["kl"].append(grpo_backward(peft_model, p_ids[j], cb, ab, n_norm, beta))
                del states_all
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            gn = float(torch.nn.utils.clip_grad_norm_(params, CFG.max_grad_norm))
            opt.step()
            step += 1
            rec = dict(update=step, reward=avg(st["R"]), acc=avg(st["acc"]), steps=avg(st["T"]),
                       grad_norm=gn, sec=time.time() - t0)
            if is_slpo:
                rec.update(lat=avg(st["lat"]), tok=avg(st["tok"]), base=avg(st["base"]), vloss=avg(st["vloss"]))
            else:
                rec["kl"] = avg(st["kl"])
                beta = max(1e-4, beta * (1 + max(-0.2, min(0.2, rec["kl"] / CFG.kl_target - 1))))
            hist.append(rec)
            log(f"[{variant}] " + " ".join(f"{k}={v:.4g}" for k, v in rec.items()))

    H.model.gradient_checkpointing = False
    peft_model.save_pretrained(os.path.join(CFG.out_dir, f"lora_{variant}"), selected_adapters=[variant])
    if vhead is not None:
        torch.save(vhead.state_dict(), os.path.join(CFG.out_dir, f"value_{variant}.pt"))
    json.dump(hist, open(os.path.join(CFG.out_dir, f"train_{variant}.json"), "w"), indent=1)


# ---------------- evaluation ----------------
def evaluate(name, items, num_steps, tag):
    items = items[:CFG.eval_limit] if CFG.eval_limit else items
    enc = [H.build_prompt(it["prompt"]) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(enc[i]))
    correct, steps, records = 0, 0, []
    for b in range(0, len(order), CFG.eval_batch_size):
        idx = order[b:b + CFG.eval_batch_size]
        outs = H.generate_chunked([enc[i] for i in idx], num_steps, CFG.max_new_tokens[name], CFG.eval_batch_size)
        for i, o in zip(idx, outs):
            txt = H.decode(o)
            ok = score_item(name, txt, items[i])
            correct += ok
            steps += len(o) * num_steps
            records.append({"i": i, "correct": bool(ok), "steps": len(o) * num_steps, "output": txt})
    res = {"acc": 100 * correct / max(1, len(items)), "steps": steps / max(1, len(items))}
    with open(os.path.join(CFG.out_dir, f"eval_{tag}_{name}.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    log(f"== {tag} | {name} | r={num_steps} | acc={res['acc']:.2f} | steps={res['steps']:.0f}")
    return res


def print_table(results, cols):
    print(f"\n{'Model':<20}" + "".join(f"{c + ' acc/steps':>24}" for c in cols))
    for name, res in results.items():
        print(f"{name:<20}" + "".join(f"{res[c]['acc']:>14.2f} / {res[c]['steps']:>7.0f}" for c in cols))


# ---------------- main ----------------
def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper", choices=["paper", "quick"])
    ap.add_argument("--variants", nargs="*", default=None, choices=list(VARIANTS))
    args = ap.parse_args()
    CFG = get_config(args.preset)
    if args.variants:
        CFG.train_variants = tuple(args.variants)
    os.makedirs(CFG.out_dir, exist_ok=True)
    random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)

    H.load_model(CFG.model_id)
    gsm_train, _ = load_gsm8k()
    train_items = [gsm_train[i] for i in random.Random(CFG.seed).sample(range(len(gsm_train)), CFG.n_train)]
    eval_sets = load_eval_sets(CFG.eval_datasets)
    log("eval sets:", {k: len(v) for k, v in eval_sets.items()})

    results = {}
    run_eval = lambda tag, r: {n: evaluate(n, it, r, tag) for n, it in eval_sets.items()}  # noqa: E731
    if CFG.eval_sft_r8:
        results[f"SFT-r{CFG.r}"] = run_eval(f"sft_r{CFG.r}", CFG.r)
    if CFG.eval_sft_r32:
        results[f"SFT-r{CFG.r_deep}"] = run_eval(f"sft_r{CFG.r_deep}", CFG.r_deep)

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout, bias="none",
                      target_modules=r".*transformer\.core_block\.\d+\.(attn\.Wqkv|attn\.proj|mlp\.fc|mlp\.proj)")
    peft_model = None
    for v in CFG.train_variants:
        if peft_model is None:
            peft_model = get_peft_model(H.model, lcfg, adapter_name=v)
        else:
            peft_model.add_adapter(v, lcfg)

    for v in CFG.train_variants:
        train_variant(peft_model, v, train_items)
        peft_model.set_adapter(v)
        results[f"{VARIANTS[v]['name']}-r{CFG.r}"] = run_eval(f"{v}_r{CFG.r}", CFG.r)

    print_table(results, list(eval_sets))
    json.dump(results, open(os.path.join(CFG.out_dir, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
