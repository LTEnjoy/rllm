#!/usr/bin/env bash
# Train the math agent with the verl (distributed GPU) backend.
#
# Prerequisites:
#   1. Install rllm with verl extras:      uv pip install -e ".[verl]"
#   2. Install this cookbook:               uv pip install --no-deps -e cookbooks/math
#   3. Pull the datasets:                   rllm dataset pull hendrycks_math && rllm dataset pull math500

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

export PYTHONWARNINGS="ignore"
export VLLM_LOGGING_LEVEL=WARN
export VERL_LOGGING_LEVEL=WARN
export NCCL_DEBUG=WARN

MODEL_PATH=/sujin/Models/Qwen/Qwen3-4B

python -u train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=gae \
    rllm.rejection_sample.min_trajs_per_group=1 \
    rllm.algorithm.use_rllm=true \
    data.train_batch_size=64 \
    data.val_batch_size=-1 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    +model.name=$MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    trainer.logger="['console']" \
    trainer.project_name=math_tool_agent \
    trainer.experiment_name=qwen3-4b-gsm8k-ppo \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    rllm.gateway.port=9091 \
    "$@"
