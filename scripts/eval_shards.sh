#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${SIM2REAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${SIM2REAL_REPO:-$(dirname "$TASK_ROOT")/VideoX-Fun}
export SIM2REAL_ROOT="$TASK_ROOT" SIM2REAL_REPO="$REPO"
output=${1:?output directory required}
split=${2:-val}
ckpt=${3:-pretrained}
cfg=${4:-1}
shift=${5:-3}
first=${6:-full}
export PYTHONPATH="$REPO:$TASK_ROOT/tools" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VIDEOX_ATTENTION_TYPE=SDPA
mkdir -p "$output"
# Prepare the shared AlexNet cache once; never race downloads across GPU workers.
CUDA_VISIBLE_DEVICES='' "$REPO/.venv/bin/python" "$TASK_ROOT/tools/evaluate.py" --output "$output" --prepare_only
extra=()
if [ "$first" = first ]; then extra+=(--first_segment_only); fi
jobs=()
IFS=',' read -r -a devices <<< "${SIM2REAL_GPUS:-4,5,6,7}"
num_shards=${#devices[@]}
for shard in "${!devices[@]}"; do
  gpu=${devices[$shard]}
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    if [ "$ckpt" != static ]; then
      "$REPO/.venv/bin/python" "$TASK_ROOT/tools/infer.py" --output "$output" --split "$split" \
        --ckpt "$ckpt" --guidance_scale "$cfg" --shift "$shift" --shard "$shard" --num_shards "$num_shards" "${extra[@]}"
    fi
    static=()
    if [ "$ckpt" = static ]; then static+=(--static); fi
    "$REPO/.venv/bin/python" "$TASK_ROOT/tools/evaluate.py" --output "$output" --split "$split" \
      --shard "$shard" --num_shards "$num_shards" "${static[@]}"
  ) > "$output/shard_$shard.log" 2>&1 &
  jobs+=("$!")
done
status=0
for shard in "${!jobs[@]}"; do
  if ! wait "${jobs[$shard]}"; then
    status=1
    echo "Evaluation shard $shard failed: $output/shard_$shard.log" >&2
    tail -n 50 "$output/shard_$shard.log" >&2
  fi
done
if [ "$status" -ne 0 ]; then echo "Evaluation shard failed: $output" >&2; exit 1; fi
"$REPO/.venv/bin/python" "$TASK_ROOT/tools/evaluate.py" --output "$output" --split "$split" --aggregate_only
