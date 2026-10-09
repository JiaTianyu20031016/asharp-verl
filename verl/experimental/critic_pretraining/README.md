# Fixed-policy critic pretraining (experimental)

This workflow separates policy rollout collection from critic optimization. It is
intended for PPO variants such as VAPO that initialize a critic on fixed-policy
Monte-Carlo returns before online actor-critic training.

## 1. Collect fixed-policy trajectories

```bash
python -m verl.experimental.critic_pretraining.collect_policy_rollouts \
  --model-path /data2/jty/models/Qwen2.5-Math-1.5B-Instruct \
  --input-files /path/to/gsm8k/train.parquet \
  --output-path /path/to/rollouts.parquet \
  --n 8 \
  --max-response-length 1024 \
  --temperature 1.0
```

The output has one row per sampled trajectory and preserves exact
`prompt_ids` and `response_ids`. `--n` is the number of responses per
prompt. A policy LoRA adapter can be supplied with
`--lora-adapter-path /path/to/adapter`. The built-in verl reward dispatcher is
used by default; custom reward code is supported through
`--reward-function-path` and `--reward-function-name`.

## 2. Pretrain the critic

LoRA:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m verl.experimental.critic_pretraining.pretrain \
  --model-path /data2/jty/models/Qwen2.5-Math-1.5B-Instruct \
  --train-files /path/to/rollouts.parquet \
  --output-dir /path/to/critic-lora \
  --lora-rank 32 \
  --lora-alpha 32 \
  --mixed-precision bf16
```

Full-parameter:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m verl.experimental.critic_pretraining.pretrain \
  --model-path /data2/jty/models/Qwen2.5-Math-1.5B-Instruct \
  --train-files /path/to/rollouts.parquet \
  --output-dir /path/to/critic-full \
  --lora-rank 0 \
  --mixed-precision bf16
```

By default, full training keeps FP32 master weights while using BF16 autocast;
LoRA loads the frozen base in BF16. Use `--model-dtype` to override this.

The training target at every response action is the discounted terminal reward;
`--gamma 1` gives the undiscounted Monte-Carlo target used by VAPO-style
critic initialization. Values are aligned to the token preceding each response
action, matching verl's PPO critic slicing convention.

The LoRA export contains the adapter and the trained scalar `score` head. The
full export is a Hugging Face `AutoModelForTokenClassification` checkpoint.
Every output directory contains `verl_critic_pretraining.json` with the exact
PPO overrides required to load it.

## Evaluate on fixed holdout trajectories

`verl.experimental.critic_pretraining.evaluate` evaluates token values using
the same preceding-token alignment as PPO. It reports EV, MSE, bias, AUC,
calibration, relative-position metrics, and prompt-start prediction quality.

```bash
python -m verl.experimental.critic_pretraining.evaluate \
  --base-model /path/to/base-token-classification-model \
  --adapter /path/to/critic-adapter \
  --rollout-files /path/to/fixed-holdout-rollouts.parquet \
  --output-dir /path/to/evaluation
```

To compare the exact whitened GAE advantages that PPO would construct from the
same fixed trajectories and critic values, add
`--advantage-lambdas 0.95 0.99 1.0`. The resulting
`advantage_by_lambda.json` reports early/late means and correct-minus-incorrect
credit gaps without modifying either model.

## 3. Load the critic in PPO

For a LoRA export:

```text
critic.model.path=/data2/jty/models/Qwen2.5-Math-1.5B-Instruct
critic.model.lora_rank=32
critic.model.lora_adapter_path=/path/to/critic-lora
```

For a full export:

```text
critic.model.path=/path/to/critic-full
critic.model.lora_rank=0
```

Use independent GAE lambdas as follows:

```text
algorithm.adv_estimator=gae
algorithm.lam_policy=0.95
algorithm.lam_critic=1.0
```

Either role-specific lambda may be `null`, in which case it falls back to
`algorithm.lam`. Thus existing configurations retain their old behavior.
`lam_critic=1.0` produces Monte-Carlo critic returns while
`lam_policy<1.0` keeps the lower-variance policy advantage estimator.

## Operational constraints

- Use a short temporary directory, for example `TMPDIR=/tmp/verl`. Ray's Unix
  socket path must remain below 108 bytes. The vLLM weight-transfer IPC code in
  this checkout also respects `TMPDIR`.
- In async rollout mode, `data.train_batch_size * rollout.n` must be divisible
  by the total number of agent-loop workers. For a small two-GPU smoke test,
  `actor_rollout_ref.rollout.agent.num_workers=1` gives two workers.
- Rollouts must be collected from the exact frozen policy checkpoint or adapter
  intended for critic initialization. Changing tokenizer/chat template or the
  policy between collection and critic pretraining invalidates that assumption.
- `torchrun` uses data-parallel critic pretraining. The 1.5B model fits
  comfortably on A800 80GB GPUs for both LoRA and full-parameter modes.
