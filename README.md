# phy_wan：Wan2.1 UR5 仿真→真实视频

输入仿真 RGB 视频和真实首帧，重建同一 episode 的完整真实视频，保持 640×480、5 fps 和原始帧数。基于 Wan2.1-Fun-V1.1-1.3B-Control，计划仅执行 Run A 全量微调。

**当前阶段：程序与数据准备。尚未开始训练、模型推理评估或模型选择。**
仓库包含数据构建、质检、训练、分段推理、评估及实验编排代码。GPU 训练/推理的冒烟验证将在后续阶段执行，不能将当前提交视为完整实验已经通过。
已通过 11 项 CPU 迁移/分段检查，并对全部 140 条数据完成输出哈希、四路视频解码规格及原始参考视频哈希复验；验证范围和未完成项见 [准备状态](reports/preparation_status.json)。

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
| `SIM2REAL_DATA` | 已处理数据目录，下一层为 `episode_XXXXXX/` |
| `SIM2REAL_SOURCE` | 原始回放目录，含各 episode 及共享 `assets/`；重建数据时需要 |
| `SIM2REAL_REAL_VIDEOS` | 可选，原始外部 RGB 视频目录，文件名为 `episode_XXXXXX.mp4`；用于重定位原始视频 |

每条处理后数据需要 `meta.json`、`real.mp4`、`sim.mp4`、`real_first81.mp4`、`sim_first81.mp4`、`first_frame.png` 和 `labels.npz`。
如未设置 `SIM2REAL_REAL_VIDEOS`，程序从回放 `prepared/manifest.json` 查找原始视频；相对路径按 manifest 所在目录解析。
模型评估使用原始视频解码帧，不能把有损转码的 `real.mp4` 当作原始参考。

最终冻结数据共 **140 条：train 118 / val 10 / test 12**。从 152 条中排除 11 条接触失败，以及视觉复核失败的 `episode_000389`。详见 `split.json`、`exclusions.json`、`reports/qc_report.json` 和 `reports/qc_review.json`。复核图片和视频保留在数据服务器，不放入代码仓库。

## 3. 只准备环境和数据，不启动训练

需要 Linux、Python 3.10、`uv`、`ffmpeg/ffprobe`、Git；训练服务器需要支持 CUDA 的 NVIDIA 驱动。

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
"$SIM2REAL_REPO/.venv/bin/python" -m pytest -q tools/test_contracts.py tools/test_portability.py
```

PyTorch 固定 2.7.1 cu128，torchvision 固定 0.22.1；依赖约束见 `configs/constraints.txt`。安装成功后生成 `reports/environment.lock.txt`，并执行上游训练入口 `--help` 及模型/管线类导入检查。默认 SDPA，不要求 flash-attn。数据渲染单独使用 MuJoCo 3.15.0 环境。

## 4. 后续训练与评估

**以下命令会开始 GPU 实验，本阶段不执行。** 原计划使用物理 GPU 4–7；运行前确认其可用。

```bash
mkdir -p logs
"$SIM2REAL_REPO/.venv/bin/python" tools/experiment.py > logs/experiment.log 2>&1
```

建议在 tmux 中运行。该入口依次执行单卡 10→20 步恢复、四卡短程恢复、验证集基线、独立 Run A 3000 步、checkpoint 筛选、完整验证及冻结后测试；失败即停止。DDP OOM 后支持 `--memory low_vram`。ZeRO-2 尚未实现，若仍 OOM，应先实现并重新冒烟，不能自动降低分辨率或帧数。

81 帧分段、重叠 1 帧，后段条件使用上一段生成末帧；保留模型原始首帧，不替换、不混合接缝。尾段重复控制末帧补齐到 4k+1，输出裁回原始帧数。按 episode 和段号稳定派生种子；每种 shift 重新构造 Flow Euler 调度器。

主要指标排除第 0 帧，报告全图/前景质量、蓝杯动作误差、失败率、分段漂移和首帧接缝。模型排序优先动作：`median + 0.25*p95`，生成检测失败惩罚 800px；动作分数在最优 2px 内时比较前景 LPIPS。逐 episode 等权汇总和 bootstrap 置信区间。详见 [实验说明](docs/experiment_zh.md)。

仓库不包含视频数据、预训练权重、checkpoint、机器凭据或运行日志。推理/评估结果、最终权重及效果报告需在后续实验完成后生成。
