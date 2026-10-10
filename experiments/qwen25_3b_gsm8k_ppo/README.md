# Qwen2.5-3B-Instruct GSM8K LoRA-32 PPO

The durable two-stage run starts with `tmux new-session -d -s qwen25_3b_gsm8k_ppo 'bash /data2/jty/verl-asharp/experiments/qwen25_3b_gsm8k_ppo/pipeline.sh > /data1/jty/logs/ppo_lora32_gsm8k_qwen25_3b/pipeline.log 2>&1'`.

- `run.sh warmup` freezes the actor and trains only the critic for eight dataset epochs (616 batches). If the final 77-step average explained variance is below 0.25 or still rises more than 0.08 relative to the preceding epoch, `pipeline.sh` resumes the same run to 12 epochs. PPO starts only after that check passes.
- The selected critic checkpoint is exported with `verl.model_merger` to `/data1/jty/outputs/ppo_lora32_gsm8k_qwen25_3b/critic_warmup_adapter`. The saved adapter includes the trained score head.
- `run.sh ppo` initializes a new actor LoRA and loads that critic adapter. The PPO cap is 15 dataset epochs (1,155 batches), with validation and checkpoints every 77 batches.
- The logical rollout batch is 96 prompts × 4 responses. In this VERL code, a PPO mini-batch setting of 96 prompts expands to all 384 responses. With one PPO epoch, actor and critic each take one optimizer step per rollout batch.
- Actor and critic rank are 32. Learning rates are `5e-6` and `1e-5`, respectively. `gamma=1`, actor lambda is 0.99, critic lambda is 1. There is no KL reward or loss, and the reward is only the binary terminal score from `../qwen25_math_lora32/reward_math_verify.py`.
- Training data is `/data1/jty/data/gsm8k_ppo_terminal/{train,test}.parquet`; prompts require a final `\boxed{}` answer. The backbone is `/data2/Common_LLM_Base/Qwen/Qwen2.5-3B-Instruct/`.
- Only GPUs 0–3 are visible to the run. Checkpoints are under `/data1/jty/outputs/ppo_lora32_gsm8k_qwen25_3b`, logs under `/data1/jty/logs/ppo_lora32_gsm8k_qwen25_3b`, and Ray temporary files under `/data1/jty/tmp/ray`.

To inspect progress, run `tmux list-sessions` and `tail -f /data1/jty/logs/ppo_lora32_gsm8k_qwen25_3b/warmup.log`. The `warmup_summary.txt`, `warmup_selection.txt`, and `completed.txt` files record stage transitions.
