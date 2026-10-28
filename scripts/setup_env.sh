#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${SIM2REAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${SIM2REAL_REPO:-$(dirname "$TASK_ROOT")/VideoX-Fun}
export SIM2REAL_ROOT="$TASK_ROOT" SIM2REAL_REPO="$REPO"
export PATH="$HOME/.local/bin:$PATH"
bash "$TASK_ROOT/scripts/bootstrap_upstream.sh"
mkdir -p "$TASK_ROOT"/{reports,logs,runs,eval,episodes}
cd "$REPO"
uv venv .venv --python 3.10 --allow-existing
torch_package=torch==2.7.1+cu128
if [ -f "$TASK_ROOT/downloads/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl" ]; then
  torch_package="$TASK_ROOT/downloads/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
fi
uv pip install --python .venv/bin/python --no-deps "$torch_package" torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
wheel_cache=${SIM2REAL_WHEEL_CACHE:-$TASK_ROOT/downloads/dependencies}
wheel_options=()
if [ -d "$wheel_cache" ]; then wheel_options+=(--find-links "$wheel_cache"); fi
if [ -f "$TASK_ROOT/configs/environment.lock.txt" ]; then
  uv pip install --python .venv/bin/python --index-url https://pypi.org/simple "${wheel_options[@]}" -c "$TASK_ROOT/configs/constraints.txt" -r "$TASK_ROOT/configs/environment.lock.txt"
else
  uv pip install --python .venv/bin/python --index-url https://pypi.org/simple "${wheel_options[@]}" -c "$TASK_ROOT/configs/constraints.txt" -r requirements.txt torchaudio==2.7.1 lpips 'peft>=0.17,<0.19' psutil pytest matplotlib
fi
uv pip list --python .venv/bin/python --format freeze > "$TASK_ROOT/reports/environment.lock.txt"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$REPO" .venv/bin/python scripts/wan2.1_fun/train_control.py --help > "$TASK_ROOT/reports/train_help.txt"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$REPO" .venv/bin/python -c 'from videox_fun.models import WanTransformer3DModel, AutoencoderKLWan, CLIPModel, WanT5EncoderModel; from videox_fun.pipeline import WanFunControlPipeline; import torch,decord; print("Model and pipeline imports passed:", torch.__version__, "CUDA build", torch.version.cuda)'
