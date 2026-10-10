#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/jty/verl-asharp
PYTHON_BIN=/data1/jty/anaconda3/envs/asharp-verl/bin/python
STAGE=${1:?usage: run.sh warmup|ppo [Hydra overrides...]}
shift

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR=/data1/jty/tmp/ray
export TMPDIR=/data1/jty/tmp
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODEL=/data2/Common_LLM_Base/Qwen/Qwen2.5-3B-Instruct
DATA=/data1/jty/data/gsm8k_ppo_terminal
OUT=/data1/jty/outputs/ppo_lora32_gsm8k_qwen25_3b
REWARD="${ROOT}/experiments/qwen25_math_lora32/reward_math_verify.py"
mkdir -p "${OUT}"

case "${STAGE}" in
  warmup)
    RUN_NAME=qwen25_3b_gsm8k_critic_warmup
    TOTAL_EPOCHS=8
    CRITIC_WARMUP=1000000
    CRITIC_ADAPTER=null
    VAL_BEFORE_TRAIN=False
    ;;
  ppo)
    RUN_NAME=qwen25_3b_gsm8k_ppo
    TOTAL_EPOCHS=15
    CRITIC_WARMUP=0
    CRITIC_ADAPTER="${OUT}/critic_warmup_adapter"
    test -s "${CRITIC_ADAPTER}/adapter_model.safetensors" || {
      echo "Missing warmed critic adapter: ${CRITIC_ADAPTER}" >&2
      exit 1
    }
    VAL_BEFORE_TRAIN=True
    ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.lam_policy=0.99 \
  algorithm.lam_critic=1.0 \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="${REWARD}" \
  reward.custom_reward_function.name=compute_score \
  data.train_files="${DATA}/train.parquet" \
  data.val_files="${DATA}/test.parquet" \
  data.train_batch_size=96 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  data.seed=42 \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.model_init_seed=42 \
  actor_rollout_ref.actor.ppo_mini_batch_size=96 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  critic.model.path="${MODEL}" \
  critic.model.lora_rank=32 \
  critic.model.lora_alpha=32 \
  critic.model.lora_adapter_path="${CRITIC_ADAPTER}" \
  'critic.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=False \
  critic.optim.lr=1e-5 \
  critic.optim.lr_warmup_steps=10 \
  critic.optim.lr_scheduler_type=constant \
  critic.model_init_seed=42 \
  critic.ppo_mini_batch_size=96 \
  critic.ppo_micro_batch_size_per_gpu=8 \
  critic.ppo_epochs=1 \
  critic.shuffle=False \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  trainer.critic_warmup="${CRITIC_WARMUP}" \
  trainer.logger='["console"]' \
  trainer.project_name=qwen25_3b_gsm8k_ppo \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq=77 \
  trainer.save_freq=77 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${OUT}/${RUN_NAME}" \
  "$@"
