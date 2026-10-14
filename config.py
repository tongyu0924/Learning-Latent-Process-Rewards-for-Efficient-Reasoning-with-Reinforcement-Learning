import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model_id: str = "tomg-group-umd/huginn-0125"
    seed: int = 42
    out_dir: str = "/content/outputs" if os.path.isdir("/content") else "./outputs"

    # experiments
    eval_sft_r8: bool = True
    eval_sft_r32: bool = True
    train_variants: tuple = ("outcome", "slpo", "slpo_value", "slpo_value_cost", "slpo_value_cost_scorer")
    eval_datasets: tuple = ("gsm8k", "mathqa", "math")
    eval_limit: int | None = None
    eval_batch_size: int = 16          # at r=8; scaled down automatically for deeper r
    rollout_batch_size: int = 32

    # recurrence depth
    r: int = 8
    r_deep: int = 32

    # data
    n_train: int = 500
    num_epochs: int = 1

    # RL
    G: int = 8
    prompts_per_micro: int = 8
    grad_accum: int = 4                # 32 prompts per update
    lr: float = 2e-6
    max_grad_norm: float = 1.0
    loss_micro_bs: int = 4

    # token-level GRPO baseline
    clip_eps: float = 0.2
    beta_init: float = 0.1
    kl_target: float = 0.1

    # SLPO surrogate
    slpo_dropout: float = 0.1
    slpo_K: int = 4
    slpo_var_eps: float = 1e-4
    slpo_lambda_latent: float = 1.0
    slpo_token_term: bool = True
    slpo_dim_mean: bool = True

    # reward / baseline
    step_cost_lambda: float = 0.1
    value_hidden: int = 256
    value_lr: float = 1e-4
    value_warmup_updates: int = 2

    # latent scorer (SWIFT-style)
    use_scorer: bool = True
    scorer_path: str | None = None     # load a trained scorer instead of training one
    scorer_n_train: int = 1000         # GSM8K train problems, disjoint from the RL subset
    scorer_n_val: int = 200
    scorer_G: int = 8
    scorer_epochs: int = 2             # each epoch draws fresh rollouts
    scorer_prompts_per_batch: int = 8
    scorer_inner_steps: int = 4
    scorer_micro_bs: int = 8
    scorer_lr: float = 1e-4
    scorer_wd: float = 1e-5
    scorer_max_pos_weight: float = 10.0
    scorer_dropout_mix: float = 0.5    # fraction of scorer batches sampled with MC-dropout (matches RL rollouts)
    scorer_eta: float = 0.2            # weight of sigma(S) in the RL reward

    # best-of-N evaluation with the scorer
    bon_n: int = 8
    bon_datasets: tuple = ("gsm8k",)
    bon_limit: int | None = 500
    bon_prompts_per_batch: int = 4
    bon_after_variants: tuple = ("slpo_value_cost", "slpo_value_cost_scorer")

    # LoRA (recurrent core only)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # generation
    train_max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: dict = field(default_factory=lambda: {"gsm8k": 256, "mathqa": 256, "math": 512})


def get_config(preset="paper"):
    cfg = Config()
    if preset == "quick":
        cfg.n_train = 16
        cfg.G = 4
        cfg.prompts_per_micro = 4
        cfg.grad_accum = 1
        cfg.eval_limit = 32
        cfg.eval_datasets = ("gsm8k",)
        cfg.r_deep = 16
        cfg.train_max_new_tokens = 160
        cfg.scorer_n_train = 16
        cfg.scorer_n_val = 8
        cfg.scorer_G = 4
        cfg.scorer_epochs = 1
        cfg.bon_n = 4
        cfg.bon_limit = 16
    return cfg
