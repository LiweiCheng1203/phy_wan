# 从原始回放重建数据

先在仓库根目录 `source .env`，配置回放、原始视频和输出目录。环境安装见 README。
`SIM2REAL_SOURCE` 下每个 episode 需有 `scene_portable.xml`、`replay.npz`、`metrics.json`、`calibration/cameras.json`、`prepared/timeline.json`、`prepared/manifest.json`；共享资产位于 `SIM2REAL_SOURCE/assets/`。

```bash
.venv-data/bin/python tools/build_dataset.py --inventory_only
# 四个独立终端/任务分别设置 GPU 和 shard。此处展示 GPU 4 / shard 0。
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 \
  .venv-data/bin/python tools/build_dataset.py --shard 0 --num_shards 4
# 其余使用 GPU 5、6、7 对应 shard 1、2、3；全部成功后：
.venv-data/bin/python tools/qc_alignment.py
```

MuJoCo EGL 设备编号与 CUDA 可见编号并不必然相同，在本机已经核实使用上述物理设备映射；更换服务器后需核实 EGL 实际 GPU。只回放保存状态并执行 `mj_forward`，不重新积分。时间为 `clip(pts + capture_state_offset_s)`，保留偏移原始精度，free/ball 旋转使用 Slerp。关闭文字、阴影和分割抗锯齿。

质检会生成每条叠加视频、8 帧接触表和蓝杯质心评分。对所有异常及固定随机 10 条复核，决策写入 `reports/qc_review.json`。
本仓库附带的复核和划分仅适用于当前这批数据；更换输入数据需要重新复核并建立独立实验目录，不能套用旧结论。

```bash
# 仅在当前数据的质检及复核完整时执行。拒绝覆盖不同划分。
.venv-data/bin/python tools/make_metadata.py --freeze
```

已冻结并已上传的处理后数据直接执行 `prepare_uploaded_data.py`，无需重新渲染或重新划分。迁移时不改写 `meta.json` 中历史输入哈希，运行路径通过环境变量解析。
