#!/usr/bin/env bash
# Train the math agent with the verl (distributed GPU) backend.
#
# Prerequisites:
#   1. Install rllm with verl extras:      uv pip install -e ".[verl]"
#   2. Install this cookbook:               uv pip install --no-deps -e cookbooks/math
#   3. Pull the datasets:                   rllm dataset pull hendrycks_math && rllm dataset pull math500

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

MODEL_PATH=/sujin/Models/Qwen/Qwen3-0.6B

python -u train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    +algorithm.grpo.baseline=mean \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=2 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.rollout.n=2 \
    trainer.project_name=math_agent \
    trainer.experiment_name=qwen3-0.6b \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    rllm.gateway.port=9091 \
    trainer.logger=['console'] \
    "$@"