# phy_wan：Wan2.1 UR5 仿真→真实视频

输入仿真 RGB 视频和真实首帧，重建同一 episode 的完整真实视频，保持 640×480、5 fps 和原始帧数。基于 Wan2.1-Fun-V1.1-1.3B-Control，计划仅执行 Run A 全量微调。

仓库提供数据准备、训练、分段推理和评估程序。数据、模型权重、检查报告和运行日志均保存在使用者自己的服务器，不上传到 Git。

## 1. 获取固定版本的上游代码

```bash
git clone https://github.com/LiweiCheng1203/phy_wan.git
cd phy_wan
bash scripts/bootstrap_upstream.sh
```

上游放在同级 `VideoX-Fun/`，固定提交 `13bb3c625e8b1e457d7124d92207b49e00e4a6fe`。已有代码可设置 `SIM2REAL_REPO`；脚本拒绝使用其他提交，避免训练接口随上游更新而变化。原仓库代码保持独立，不需要复制修改其训练程序。上游代码及模型分别遵循其自身许可。

## 2. 连接已经上传的数据和权重

```bash
cp .env.example .env
# 编辑 .env，填入服务器实际绝对路径，然后执行：
source .env
```

| 变量 | 说明 |
| --- | --- |
| `SIM2REAL_ROOT` | 本仓库绝对路径，配置、日志及运行输出保存在此 |
| `SIM2REAL_REPO` | 固定版本的 VideoX-Fun 目录 |
| `SIM2REAL_MODEL` | 完整的 Wan2.1-Fun-V1.1-1.3B-Control 权重目录 |
| `SIM2REAL_GPUS` | 本机 GPU 编号，例如 `0,1,2,3`；完整训练使用四卡 |
| `SIM2REAL_DATA` | 已处理数据目录，下一层为 `episode_XXXXXX/` |
| `SIM2REAL_SOURCE` | 原始回放目录，含各 episode 及共享 `assets/`；重建数据时需要 |
| `SIM2REAL_REAL_VIDEOS` | 可选，原始外部 RGB 视频目录，文件名为 `episode_XXXXXX.mp4`；用于重定位原始视频 |

每条处理后数据需要 `meta.json`、`real.mp4`、`sim.mp4`、`real_first81.mp4`、`sim_first81.mp4`、`first_frame.png` 和 `labels.npz`。
如未设置 `SIM2REAL_REAL_VIDEOS`，程序从回放 `prepared/manifest.json` 查找原始视频；相对路径按 manifest 所在目录解析。
模型评估使用原始视频解码帧，不能把有损转码的 `real.mp4` 当作原始参考。

随附 `split.json` 和 `exclusions.json` 对应配套数据：train 118 / val 10 / test 12。它们不包含服务器路径。使用其他数据时按 [数据构建说明](docs/data_zh.md) 建立自己的划分。

## 3. 安装环境并准备数据

需要 Linux、Python 3.10、`uv`、`ffmpeg/ffprobe`、Git 和支持 CUDA 12.8 的 NVIDIA 驱动。完整实验按四张 A100 80GB 配置，实际显存峰值需在本机冒烟验证。

```bash
# CPU/EGL 数据工具环境，不安装大规模训练依赖：
uv venv .venv-data --python 3.10
uv pip install --python .venv-data/bin/python -r configs/data_requirements.txt
.venv-data/bin/python tools/prepare_uploaded_data.py --decode_check --require_reference
```

校验文件哈希、编码、尺寸、帧率和解码帧数，并验证评估用原始视频哈希。全部通过后才生成本机绝对路径的 `metadata_*.json`，保持 `split.json` 不变。训练元数据共 1180 条，每条 episode 对应 7 个完整视频条目和 3 个首窗口条目。只检查训练数据时可暂时不传 `--require_reference`，之后评估前必须补齐原始视频。

若上传的是原始回放，而非上述处理后数据，按 [数据构建说明](docs/data_zh.md) 生成。

```bash
# 安装训练依赖并检查真实训练入口和模型类的导入；不会训练或加载模型权重：
bash scripts/setup_env.sh
"$SIM2REAL_REPO/.venv/bin/python" -m pytest -q tools/test_contracts.py tools/test_portability.py tools/test_training_runtime.py
```

PyTorch 固定 2.7.1 cu128，torchvision 固定 0.22.1。依赖版本见 `configs/environment.lock.txt`，兼容约束见 `configs/constraints.txt`。安装脚本检查训练入口和模型导入，默认 SDPA，不要求 flash-attn。数据渲染单独使用 MuJoCo 3.15.0 环境。

安装时预备 LPIPS 所需的 AlexNet 权重，默认缓存到项目 `.cache/torch/`；可通过 `SIM2REAL_TORCH_HOME` 指定自己的可写目录。评估入口会先在 CPU 检查该权重，再启动各 GPU 分片；失败分片的日志会直接显示在主日志中。

## 4. 训练与评估

以下命令会开始训练。先在 `.env` 设置可用的 `SIM2REAL_GPUS`，并确认其没有被其他任务占用。

```bash
mkdir -p logs
"$SIM2REAL_REPO/.venv/bin/python" tools/experiment.py > logs/experiment.log 2>&1
```

建议在 tmux 中运行。该入口依次执行单卡恢复冒烟、四卡恢复冒烟、验证集基线、Run A 3000 步、模型选择和测试；失败即停止。计算采用 bf16 混合精度，可训练参数及 AdamW 状态使用 FP32。DDP OOM 后支持 `--memory low_vram`；ZeRO-2 尚未实现。

训练包装入口补齐各 DDP 进程及训练专用生成器的随机状态，并为恢复后的外层 epoch 循环加入目标步数保护；固定版本的上游源文件保持原样。缺少专用随机状态的旧 checkpoint 不能用于恢复训练，但可加载推理。采样器沿用上游的预取回退策略，恢复不保证逐位重现未中断训练。

81 帧分段、重叠 1 帧，后段条件使用上一段生成末帧；保留模型原始首帧，不替换、不混合接缝。尾段重复控制末帧补齐到 4k+1，输出裁回原始帧数。按 episode 和段号稳定派生种子；每种 shift 重新构造 Flow Euler 调度器。

主要指标排除第 0 帧，报告全图/前景质量、蓝杯动作误差、失败率、分段漂移和首帧接缝。模型排序优先动作：`median + 0.25*p95`，生成检测失败惩罚 800px；动作分数在最优 2px 内时比较前景 LPIPS。逐 episode 等权汇总和 bootstrap 置信区间。详见 [实验说明](docs/experiment_zh.md)。

单独推理、指标定义及输出目录见 [实验说明](docs/experiment_zh.md)。所有报告及结果写入本机 `reports/`、`runs/`、`eval/`，均已被 `.gitignore` 排除。
