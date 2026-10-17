import os
from dataclasses import dataclass, field


@dataclass
class Config:
    preset: str = "colab"
    model_id: str = "tomg-group-umd/huginn-0125"
    seed: int = 42
    out_dir: str = "/content/outputs" if os.path.isdir("/content") else "./outputs"

    # experiments
    eval_sft_r8: bool = True
    eval_sft_r32: bool = False
    train_variants: tuple = ("slpo_value_cost",)
    eval_datasets: tuple = ("gsm8k",)
    eval_limit: int | None = 300
    eval_batch_size: int = 16          # at r=8; scaled down automatically for deeper r
    rollout_batch_size: int = 32

    # recurrence depth
    r: int = 8
    r_deep: int = 32

    # data
    n_train: int = 128
    num_epochs: int = 2
    # curriculum filter: one greedy pass of the base model over `filter_pool` GSM8K train problems,
    # train only on the ones it solves. At r=8 the base model gets ~15%, so on random problems most
    # rollout groups are all-wrong and carry no signal. Cached in out_dir and shared by all variants.
    filter_train: bool = True
    filter_pool: int = 768

    # RL
    G: int = 8
    prompts_per_micro: int = 4
    grad_accum: int = 2                # prompts per update = prompts_per_micro * grad_accum
    lr: float = 3e-5                   # LoRA starts at B=0: 2e-6 barely moves it in a short run
    lr_warmup_updates: int = 3
    max_grad_norm: float = 1.0
    loss_micro_bs: int = 4
    skip_all_wrong: bool = True        # no policy update on groups with zero correct rollouts (V is still fit)

    # token-level GRPO baseline
    clip_eps: float = 0.2
    beta_init: float = 0.1
    kl_target: float = 0.1

    # SLPO surrogate
    slpo_dropout: float = 0.05         # Huginn was pretrained without dropout; 0.1 hurts rollouts
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

    # latent scorer
    use_scorer: bool = False
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
    train_max_new_tokens: int = 384    # 256 truncates many GSM8K solutions before the final answer
    temperature: float = 0.7           # T=1.0 + dropout gives ~3-5% correct rollouts at r=8
    top_p: float = 0.95
    max_new_tokens: dict = field(default_factory=lambda: {"gsm8k": 384, "mathqa": 384, "math": 512})

    # monitoring: greedy accuracy on the last `monitor_n` GSM8K test problems (disjoint from the
    # reported eval slice) every `monitor_every` updates; 0 disables it
    monitor_every: int = 8
    monitor_n: int = 100


def get_config(preset="colab"):
    cfg = Config(preset=preset)
    if preset == "paper":       # original settings (many GPU hours)
        cfg.eval_sft_r32 = True
        cfg.train_variants = ("outcome", "slpo", "slpo_value", "slpo_value_cost", "slpo_value_cost_scorer")
        cfg.eval_datasets = ("gsm8k", "mathqa", "math")
        cfg.eval_limit = None
        cfg.n_train = 500
        cfg.num_epochs = 1
        cfg.filter_train = False
        cfg.prompts_per_micro = 8
        cfg.grad_accum = 4
        cfg.lr = 2e-6
        cfg.lr_warmup_updates = 0
        cfg.skip_all_wrong = False
        cfg.slpo_dropout = 0.1
        cfg.use_scorer = True
        cfg.train_max_new_tokens = 256
        cfg.temperature = 1.0
        cfg.max_new_tokens = {"gsm8k": 256, "mathqa": 256, "math": 512}
        cfg.monitor_every = 0
    elif preset == "quick":     # smoke test
        cfg.n_train = 8
        cfg.num_epochs = 1
        cfg.filter_pool = 24
        cfg.G = 4
        cfg.prompts_per_micro = 4
        cfg.grad_accum = 1
        cfg.eval_limit = 16
        cfg.r_deep = 16
        cfg.train_max_new_tokens = 128
        cfg.max_new_tokens = {"gsm8k": 160, "mathqa": 160, "math": 256}
        cfg.value_warmup_updates = 1
        cfg.monitor_every = 1
        cfg.monitor_n = 8
        cfg.scorer_n_train = 16
        cfg.scorer_n_val = 8
        cfg.scorer_G = 4
        cfg.scorer_epochs = 1
        cfg.bon_n = 4
        cfg.bon_limit = 16
    elif preset != "colab":
        raise ValueError(preset)
    return cfg
