#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${SIM2REAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${SIM2REAL_REPO:-$(dirname "$TASK_ROOT")/VideoX-Fun}
MODEL=${SIM2REAL_MODEL:-$REPO/model/Wan2.1-Fun-V1.1-1.3B-Control}
export SIM2REAL_ROOT="$TASK_ROOT" SIM2REAL_REPO="$REPO" SIM2REAL_MODEL="$MODEL"
name=${1:?run name required}
workers=${2:-4}
steps=${3:-3000}
save_every=${4:-250}
resume=${5:-none}
memory=${6:-ddp}
metadata=${7:-metadata_train.json}
# Optional overrides; defaults reproduce Run A exactly.
frames=${SIM2REAL_FRAMES:-81}
batch=${SIM2REAL_BATCH:-1}
accum=${SIM2REAL_ACCUM:-2}
loader_workers=${SIM2REAL_LOADER_WORKERS:-8}
vae_batch=${SIM2REAL_VAE_MINI_BATCH:-1}
ckpt_limit=${SIM2REAL_CKPT_LIMIT:-12}
report_to=${SIM2REAL_REPORT_TO:-tensorboard}
port=${SIM2REAL_PORT:-29647}
noise_prob=${SIM2REAL_COND_NOISE_PROB:-0}
noise_sigma=${SIM2REAL_COND_NOISE_SIGMA:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SIM2REAL_GPUS:-4,5,6,7}}
export PYTHONPATH="$REPO:$TASK_ROOT/tools"
export SIM2REAL_AUDIT="$TASK_ROOT/runs/$name/audit"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VIDEOX_ATTENTION_TYPE=SDPA
mkdir -p "$TASK_ROOT/runs/$name"
cd "$REPO"
launch=(.venv/bin/accelerate launch --num_processes "$workers" --num_machines 1 --mixed_precision bf16 --main_process_port "$port")
if [ "$workers" -gt 1 ]; then launch+=(--multi_gpu); fi
extra=()
if [ "$resume" != none ]; then
  test -d "$TASK_ROOT/runs/$name/$resume/transformer"
  extra+=(--resume_from_checkpoint "$resume")
fi
if [ "$memory" = low_vram ]; then extra+=(--low_vram); fi
if [ "$noise_prob" != 0 ]; then extra+=(--cond_noise_prob "$noise_prob" --cond_noise_sigma_max "$noise_sigma"); fi
if [ "$report_to" = wandb ]; then extra+=(--tracker_project_name "${WANDB_PROJECT:-phy_wan}"); fi
if [ "$memory" != ddp ] && [ "$memory" != low_vram ]; then
  echo "Unsupported memory mode: $memory" >&2
  exit 2
fi
exec "${launch[@]}" "$TASK_ROOT/tools/train_entry.py" \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path "$MODEL" \
  --train_data_dir '' --train_data_meta "$TASK_ROOT/$metadata" \
  --enable_bucket --fix_sample_size 480 640 \
  --video_sample_size 640 --image_sample_size 640 --token_sample_size 640 \
  --video_sample_stride 1 --video_sample_n_frames "$frames" --video_repeat 1 \
  --train_batch_size "$batch" --gradient_accumulation_steps "$accum" --dataloader_num_workers "$loader_workers" \
  --num_train_epochs 1000 --max_train_steps "$steps" --checkpointing_steps "$save_every" \
  --checkpoints_total_limit "$ckpt_limit" --learning_rate 1e-5 --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 --adam_weight_decay 0.03 --adam_epsilon 1e-10 --max_grad_norm 0.05 \
  --mixed_precision bf16 --gradient_checkpointing --vae_mini_batch "$vae_batch" --uniform_sampling \
  --train_mode control_ref --control_ref_image first_frame --add_full_ref_image_in_self_attention \
  --trainable_modules . --seed 42 --report_to "$report_to" --output_dir "$TASK_ROOT/runs/$name" "${extra[@]}"
