"""SWIFT-style latent scorer: reads Huginn's refinement states h_{j,1..r}, learns per-token gates and scores,
aggregates them into one trajectory score S, and is trained with BCE on final-answer correctness."""
import json
import os
import random
import time
from collections import Counter

import torch
import torch.nn.functional as F

import huginn as H
from data import GSM_SUFFIX, extract_answer, gsm_correct, score_item

DEVICE = H.DEVICE


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class LatentScorer(torch.nn.Module):
    def __init__(self, d, r, eps=1e-8):
        super().__init__()
        self.r, self.d, self.eps = r, d, eps
        self.proj = torch.nn.Linear(r * d, 2)        # -> (gate logit, score)

    def features(self, states, P, lens):
        """u_{i,j} = [h_{j,1}; ...; h_{j,r}] for every answer token j. Returns u [B, n, rd], mask [B, n]."""
        n_max = max(lens)
        u = torch.zeros((len(states), n_max, self.r * self.d))
        mask = torch.zeros((len(states), n_max))
        for i, (s, n) in enumerate(zip(states, lens)):
            u[i, :n] = s[P - 1:P - 1 + n, 1:self.r + 1].float().reshape(n, -1)
            mask[i, :n] = 1.0
        return u.to(DEVICE), mask.to(DEVICE)

    def forward(self, u, mask):
        out = self.proj(u)
        g = torch.sigmoid(out[..., 0]) * mask
        s = out[..., 1]
        return (g * s).sum(1) / g.sum(1).clamp(min=self.eps)    # S_i (a logit)

    @torch.no_grad()
    def score(self, states, P, lens, bs=8):
        """S_i for a group of trajectories sharing one prompt of length P."""
        out = []
        for i in range(0, len(states), bs):
            u, m = self.features(states[i:i + bs], P, lens[i:i + bs])
            out += self(u, m).float().cpu().tolist()
        return out


# ---------------- data collection ----------------
def _rollouts(cfg, items, prompt_key, G, dropout):
    """Sampled rollouts with latent capture. Yields (item, prompt_ids, comps, states) per problem."""
    p_ids = [H.build_prompt(it[prompt_key] + (GSM_SUFFIX if prompt_key == "question" else "")) for it in items]
    flat = [p for p in p_ids for _ in range(G)]
    with H.mc_dropout(dropout):
        comps, states = H.generate_chunked(flat, cfg.r, cfg.train_max_new_tokens, cfg.rollout_batch_size,
                                           capture=True, do_sample=True, temperature=cfg.temperature,
                                           top_p=cfg.top_p)
    for j, it in enumerate(items):
        sl = slice(j * G, (j + 1) * G)
        yield it, p_ids[j], comps[sl], states[sl]


def _auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y > 0.5]
    neg = [s for s, y in zip(scores, labels) if y <= 0.5]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


# ---------------- training ----------------
def train_scorer(cfg, train_items, val_items):
    sc = LatentScorer(H.model.config.n_embd, cfg.r).to(DEVICE)
    opt = torch.optim.AdamW(sc.parameters(), lr=cfg.scorer_lr, weight_decay=cfg.scorer_wd)
    n_pos = n_neg = 1
    best, best_state = float("inf"), None
    for ep in range(cfg.scorer_epochs):
        order = random.sample(train_items, len(train_items))
        for b0 in range(0, len(order), cfg.scorer_prompts_per_batch):
            batch = order[b0:b0 + cfg.scorer_prompts_per_batch]
            drop = cfg.slpo_dropout if random.random() < cfg.scorer_dropout_mix else 0.0
            data = []
            for it, p, comps, states in _rollouts(cfg, batch, "question", cfg.scorer_G, drop):
                y = [gsm_correct(H.decode(c), it["answer"]) for c in comps]
                n_pos += sum(y)
                n_neg += len(y) - sum(y)
                data.append((p, comps, states, y))
            pos_w = torch.tensor(min(n_neg / n_pos, cfg.scorer_max_pos_weight), device=DEVICE)
            losses = []
            for _ in range(cfg.scorer_inner_steps):
                for p, comps, states, y in data:
                    for i in range(0, len(comps), cfg.scorer_micro_bs):
                        u, m = sc.features(states[i:i + cfg.scorer_micro_bs], len(p),
                                           [len(c) for c in comps[i:i + cfg.scorer_micro_bs]])
                        S = sc(u, m)
                        t = torch.tensor(y[i:i + cfg.scorer_micro_bs], device=DEVICE)
                        loss = F.binary_cross_entropy_with_logits(S, t, pos_weight=pos_w)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        losses.append(float(loss))
            del data
            log(f"[scorer] ep {ep} {b0 + len(batch)}/{len(order)} bce={sum(losses) / len(losses):.4f} "
                f"pos_rate={n_pos / (n_pos + n_neg):.3f}")
        val = validate_scorer(cfg, sc, val_items)
        if val["bce"] < best:
            best, best_state = val["bce"], {k: v.detach().clone() for k, v in sc.state_dict().items()}
    if best_state is not None:
        sc.load_state_dict(best_state)
    sc.eval()
    torch.save(sc.state_dict(), os.path.join(cfg.out_dir, "latent_scorer.pt"))
    return sc


@torch.no_grad()
def validate_scorer(cfg, sc, items):
    """AUC / BCE on held-out problems, plus within-group best-of-G vs. random pick vs. oracle."""
    S_all, y_all, bon, rand, oracle = [], [], 0, 0.0, 0
    for b0 in range(0, len(items), cfg.scorer_prompts_per_batch):
        for it, p, comps, states in _rollouts(cfg, items[b0:b0 + cfg.scorer_prompts_per_batch], "question",
                                              cfg.scorer_G, 0.0):
            y = [gsm_correct(H.decode(c), it["answer"]) for c in comps]
            S = sc.score(states, len(p), [len(c) for c in comps])
            S_all += S
            y_all += y
            bon += y[max(range(len(S)), key=S.__getitem__)]
            rand += sum(y) / len(y)
            oracle += max(y)
    n = max(1, len(items))
    St, yt = torch.tensor(S_all), torch.tensor(y_all)
    res = {"auc": _auc(S_all, y_all), "bce": float(F.binary_cross_entropy_with_logits(St, yt)),
           "bon_acc": 100 * bon / n, "random_acc": 100 * rand / n, "oracle_acc": 100 * oracle / n}
    log("[scorer] val " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))
    return res


def load_scorer(cfg, path):
    sc = LatentScorer(H.model.config.n_embd, cfg.r).to(DEVICE)
    sc.load_state_dict(torch.load(path, map_location=DEVICE))
    return sc.eval()


# ---------------- best-of-N evaluation ----------------
@torch.no_grad()
def evaluate_best_of_n(cfg, sc, name, items, tag):
    """Sample N answers per problem (dropout off), pick by scorer. Also reports majority vote, mean, oracle."""
    items = items[:cfg.bon_limit] if cfg.bon_limit else items
    N = cfg.bon_n
    hit = {"scorer": 0, "majority": 0, "mean": 0.0, "oracle": 0}
    records = []
    for b0 in range(0, len(items), cfg.bon_prompts_per_batch):
        batch = items[b0:b0 + cfg.bon_prompts_per_batch]
        p_ids = [H.build_prompt(it["prompt"]) for it in batch]
        flat = [p for p in p_ids for _ in range(N)]
        comps, states = H.generate_chunked(flat, cfg.r, cfg.max_new_tokens[name], cfg.rollout_batch_size,
                                           capture=True, do_sample=True, temperature=cfg.temperature,
                                           top_p=cfg.top_p)
        for j, it in enumerate(batch):
            sl = slice(j * N, (j + 1) * N)
            cs, ss = comps[sl], states[sl]
            texts = [H.decode(c) for c in cs]
            ok = [bool(score_item(name, t, it)) for t in texts]
            S = sc.score(ss, len(p_ids[j]), [len(c) for c in cs])
            pick = max(range(N), key=S.__getitem__)
            answers = [extract_answer(name, t) for t in texts]
            votes = Counter(a for a in answers if a is not None)
            maj = answers.index(votes.most_common(1)[0][0]) if votes else 0
            hit["scorer"] += ok[pick]
            hit["majority"] += ok[maj]
            hit["mean"] += sum(ok) / N
            hit["oracle"] += any(ok)
            records.append({"i": b0 + j, "correct": ok, "S": S, "pick": pick, "majority": maj})
        del comps, states
    n = max(1, len(items))
    res = {k: 100 * v / n for k, v in hit.items()}
    with open(os.path.join(cfg.out_dir, f"bon_{tag}_{name}.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    log(f"== BoN@{N} {tag} | {name} | " + " ".join(f"{k}={v:.2f}" for k, v in res.items()))
    return res
