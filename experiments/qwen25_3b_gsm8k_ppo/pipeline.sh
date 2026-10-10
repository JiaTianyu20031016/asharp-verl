#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/jty/verl-asharp
EXPERIMENT="${ROOT}/experiments/qwen25_3b_gsm8k_ppo"
OUT=/data1/jty/outputs/ppo_lora32_gsm8k_qwen25_3b
LOGS=/data1/jty/logs/ppo_lora32_gsm8k_qwen25_3b
PYTHON_BIN=/data1/jty/anaconda3/envs/asharp-verl/bin/python
mkdir -p "${OUT}" "${LOGS}"
cd "${ROOT}"

summarize_warmup() {
  "${PYTHON_BIN}" "${EXPERIMENT}/summarize_warmup.py" "${LOGS}/warmup.log" "${LOGS}/warmup_extended.log" | tee "${LOGS}/warmup_summary.txt"
}

bash "${EXPERIMENT}/run.sh" warmup > "${LOGS}/warmup.log" 2>&1
if ! summarize_warmup; then
  bash "${EXPERIMENT}/run.sh" warmup trainer.total_epochs=12 > "${LOGS}/warmup_extended.log" 2>&1
  summarize_warmup
fi

WARMUP_DIR="${OUT}/qwen25_3b_gsm8k_critic_warmup"
STEP=$(find "${WARMUP_DIR}" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n | tail -1)
test -n "${STEP}"
test -d "${WARMUP_DIR}/global_step_${STEP}/critic"
"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${WARMUP_DIR}/global_step_${STEP}/critic" \
  --target_dir "${OUT}/critic_warmup_export" > "${LOGS}/export_critic.log" 2>&1
test -s "${OUT}/critic_warmup_export/lora_adapter/adapter_model.safetensors"
cp -a "${OUT}/critic_warmup_export/lora_adapter" "${OUT}/critic_warmup_adapter"
printf 'warmup_step=%s\nadapter=%s\n' "${STEP}" "${OUT}/critic_warmup_adapter" > "${OUT}/warmup_selection.txt"

bash "${EXPERIMENT}/run.sh" ppo > "${LOGS}/ppo.log" 2>&1
printf 'PPO completed at %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "${OUT}/completed.txt"
