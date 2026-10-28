# Run A 实验约定

本文件说明后续实验行为，不表示已经完成训练、推理冒烟或效果验收。部署与路径配置见根目录 README。

## 训练

使用现有上游训练入口，由 `tools/train_entry.py` 增加运行时审计，保持上游文件不变。
Transformer 全参数更新；VAE、T5、CLIP 冻结。81 帧、480×640、stride 1，不做随机空间裁剪。
使用 control_ref、first_frame、full_ref 分支输入置零。四卡 DDP bf16，每卡 batch 1，累积 2。
可训练参数和 AdamW 状态保存为 FP32，计算使用 bf16 autocast；冻结模块使用 bf16。运行时包装器完成精度转换，避免小学习率更新被 bf16 参数存储舍入。
AdamW lr 1e-5、weight decay 0.03、epsilon 1e-10，constant with warmup，warmup 100，3000 个优化器更新步。
启用 gradient checkpointing、VAE mini-batch 1、uniform timestep sampling。每 250 步保存，最多保留 12 个 checkpoint，seed 42、8 workers/进程，无 EMA。
保留作者梯度裁剪实现：阈值从 0.25 衰减至 0.05，并带异常梯度处理。

`experiment.py` 顺序：单卡 10 步保存并恢复至 20 步 → 四卡 4 步保存并恢复至 6 步 → 静态与零样本验证基线 → 从预训练权重独立启动正式 3000 步 → checkpoint 筛选 → 完整 val 选择 → 冻结配置 → test → 报告。
`runs/stages/` 保存成功阶段，`reports/experiment_status.json` 保存当前状态。失败即停，不把冒烟权重用作正式起点。
DDP OOM 时使用 `--memory low_vram`；ZeRO-2 留待需要时实现和重新冒烟，不能直接声称该回退已经可运行。

## 单独推理

```bash
source .env
CUDA_VISIBLE_DEVICES="${SIM2REAL_GPUS%%,*}" "$SIM2REAL_REPO/.venv/bin/python" tools/infer.py \
  --ckpt pretrained --split val --shard 0 --num_shards 4 \
  --guidance_scale 1 --steps 50 --shift 3 --seed 42 --output "$SIM2REAL_ROOT/eval/example"
```

CLI 支持 `--first_segment_only`；`--ckpt` 可指向完整 checkpoint 或 Transformer safetensors。严格校验全部键名和形状。
四卡完整推理与指标通过 `bash scripts/eval_shards.sh OUTPUT val CKPT CFG SHIFT full` 启动。
训练和评估设备通过 `.env` 中 `SIM2REAL_GPUS` 配置，单卡冒烟使用列表中的第一张卡。

每段控制张量 `[1,3,L,480,640]`，首帧条件 `[1,3,1,480,640]`，clip_image 为同一帧 PIL RGB，ref_image=None。
关闭 TeaCache，完整模型加载。Flow Euler 每段重新构造并设置 shift，记录实际 timesteps/sigmas。
从帧 0 开始，每段最多 81 帧、相邻控制重叠 1 帧；送入模型长度补齐至 4k+1，模型输出截回实际长度，再删除后段首个重叠输出。
N=164 时产生第三个 4 帧短段，补齐到 5。后段条件取生成末帧 float 张量，在内存传递。
保留模型首帧，不替换真实帧、不混合接缝。原始输出为 float32 `frames.npy`，MP4 仅供查看。

## 指标与选择

主要指标排除第 0 帧；首帧单报。全图 PSNR、SSIM、LPIPS-Alex；标签 1–3 膨胀 16px 后计算前景 PSNR 和空间 LPIPS 图的掩码加权均值。
蓝杯在标签 2 膨胀 40px 区域内检测，HSV H=90–135、S≥60、V≥40，少于 20 像素无效。
真实不可检测帧不参与误差，但报告参考检测率；真实可检测而生成失败，排序误差记 800px，同时报告失败率及未惩罚有效帧统计。
先逐 episode 汇总，再等权平均。bootstrap 以 episode 为单位，固定 seed 0，10000 次。
分别报告 1–80 帧、81 帧以后、逐段指标以及接缝附近变化和视频。

对 500、1000、1500、2000、2500、3000 步 checkpoint 在 val 首段筛选。
动作分数 A=惩罚中位误差+0.25×惩罚 p95，距最优不超过 2px 的候选优先选择前景 LPIPS，平局再看动作分数和较早训练步数。
选两个 checkpoint 跑完整 val，CFG∈{1,3}、shift∈{3,5}；零样本也扫描相同四组。
冻结 `configs/final_selection.json` 后，test 对静态、最佳零样本和最佳微调模型各评估一次，不用 test 调参。

## 后续产物

- `reports/environment.lock.txt`：环境安装成功后导出，约束文件不能冒充实测完整锁定。
- `reports/provenance.json`：运行 `tools/provenance.py` 记录权重哈希、上游提交及硬件。
- `reports/final_report_zh.md`、`reports/final_results.json`：完成实验后的结果与达标状态。
- `best_transformer/`、`runs/run_a/checkpoint-3000/`：最佳权重和最后可恢复状态。
- `eval/test_*/episodes.csv`、三联视频、首帧/接缝记录、`runs/run_a/audit/`：评估与训练证据。

最终核验成功后，报告入口会清理本任务 Run A 未选 checkpoint，保留最佳和最后状态。
不自动追加 Run B。效果未达标时仍保留完整结果，并将结论限定为“基于已有仿真回放的真实视频重建”。
