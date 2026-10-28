"""Verify final test artifacts, write Chinese report, and retain authorized checkpoints."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, REPO, ROOT, probe, read_json, sha256, write_json


def main():
    selection = read_json(ROOT / "configs/final_selection.json")
    split = read_json(ROOT / "split.json")
    summary = {name: read_json(ROOT / "eval" / name / "summary.json") for name in ["test_static", "test_pretrained", "test_best"]}
    for name in summary:
        assert summary[name]["episodes"] == 12
        for ep in split["test"]:
            folder = ROOT / "eval" / name / ep
            n = read_json(DATA / ep / "meta.json")["N"]
            metrics = read_json(folder / "metrics.json")
            assert metrics["N"] == n and len(metrics["frames"]) == n
            video = probe(folder / "generated.mp4")
            assert (video["width"], video["height"], video["r_frame_rate"], int(video["nb_read_frames"])) == (640, 480, "5/1", n)
            if name != "test_static":
                generation = read_json(folder / "generation.json")
                assert generation["N"] == n
                assert all(sha256(folder / f) == h for f, h in generation["output_sha256"].items())
                frames = np.load(folder / "frames.npy", mmap_mode="r")
                assert frames.shape == (n, 480, 640, 3)
    baseline = summary["test_pretrained"]["main"]
    final = summary["test_best"]["main"]
    static = summary["test_static"]["main"]
    outcome = {"engineering_passed": True,
        "action_better_than_pretrained": final["action_score"]["mean"] < baseline["action_score"]["mean"],
        "fg_lpips_better_than_both": final["fg_lpips"]["mean"] < min(baseline["fg_lpips"]["mean"], static["fg_lpips"]["mean"]),
        "detection_failure_not_worse": final["detection_failure_rate"]["mean"] <= baseline["detection_failure_rate"]["mean"],
        "selection": selection, "summaries": summary}
    write_json(ROOT / "reports/final_results.json", outcome)
    events = []
    audit = ROOT / "runs/run_a/audit/rank0.jsonl"
    for line in audit.read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "training":
            events.append(event)
    if events:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for key in ["train_loss", "loss"]:
            points = [(e["step"], e["values"][key]) for e in events if key in e["values"]]
            if points:
                axes[0].plot(*zip(*points), linewidth=.6, label=key)
        axes[0].set_ylabel("Loss")
        axes[1].plot([e["step"] for e in events], [e["peak_cuda_bytes"] / 2**30 for e in events])
        axes[1].set_ylabel("Peak allocated GiB")
        axes[1].set_xlabel("Optimizer update")
        fig.tight_layout()
        fig.savefig(ROOT / "reports/training_curve.png", dpi=160)
        plt.close(fig)
    def cell(name, metric, group="main"):
        value = summary[name][group][metric]
        return f"{value['mean']:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]"
    lines = ["# Wan2.1 UR5 仿真到真实视频重建实验报告", "",
        "工程检查：通过。模型效果按下表单独判定；未达标时不自动追加 Run B。", "",
        f"数据划分：train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}，固定种子 0。",
        f"最佳 checkpoint：{selection['best']['ckpt']}；CFG={selection['best']['cfg']}，shift={selection['best']['shift']}，50 步。",
        f"零样本配置：CFG={selection['pretrained']['cfg']}，shift={selection['pretrained']['shift']}，50 步。", "",
        "下表为 episode 等权平均及 episode bootstrap 95% 置信区间。主要指标均排除第 0 帧。", "",
        "| 指标 | 静态首帧 | 零样本 | 微调 |", "|---|---:|---:|---:|"]
    for metric in ["action_score", "penalized_centroid_median", "penalized_centroid_p95", "centroid_median", "detection_failure_rate", "reference_detection_rate", "psnr", "ssim", "lpips", "fg_psnr", "fg_lpips"]:
        lines.append("| " + metric + " | " + " | ".join(cell(name, metric) for name in summary) + " |")
    lines += ["", "## 效果判定", ""]
    for key in ["action_better_than_pretrained", "fg_lpips_better_than_both", "detection_failure_not_worse"]:
        lines.append(f"- {key}: {'通过' if outcome[key] else '未达标'}")
    lines += ["", "## 首帧与分段漂移", "", "所有原始生成首帧均保留；首帧 PSNR 只作为质量报告项。"]
    for name in summary:
        lines.append(f"- {name}: 首帧 PSNR {cell(name, 'psnr', 'first_frame')}；第一段动作分数 {cell(name, 'action_score', 'first_segment')}；后续段动作分数 {cell(name, 'action_score', 'later_segments')}。")
    lines += ["", "接缝原始变化量、真实变化量和接缝片段保存在逐条 metrics.json 及 seam_*.mp4 中。", "",
        "## 解释与限制", "", "模型条件是已有仿真回放和真实首帧；结论限定为真实视频重建，不代表独立动作预测能力。",
        "没有替换首帧、插帧或过渡混合；后续分段只使用前一段生成末帧。无 Run B。",
        "蓝杯 HSV 检测使用冻结阈值和仿真蓝杯膨胀 40px 的区域；真实不可检测帧不计位置误差，生成失败但真实可检测的帧惩罚 800px。",
        "训练保留作者从 0.25 衰减至 0.05 的梯度裁剪和异常梯度处理。", "",
        "## 逐条结果", "", "所有 test episode 的指标见 eval/test_*/episodes.csv；完整帧指标、三联视频和接缝片段在同名 episode 目录。", ""]
    (ROOT / "reports/final_report_zh.md").write_text("\n".join(lines))
    keep = {Path(selection["best"]["ckpt"]).name, "checkpoint-3000"}
    removed = []
    for ckpt in (ROOT / "runs/run_a").glob("checkpoint-*"):
        if ckpt.name not in keep:
            assert ckpt.parent.resolve() == (ROOT / "runs/run_a").resolve() and not ckpt.is_symlink()
            removed.append(str(ckpt))
            shutil.rmtree(ckpt)
    write_json(ROOT / "reports/checkpoint_retention.json", {"kept": sorted(keep), "removed": removed})
    best = ROOT / "best_transformer"
    if not best.exists():
        best.symlink_to(Path(selection["best"]["ckpt"]) / "transformer", target_is_directory=True)
    print(json.dumps({k: v for k, v in outcome.items() if k not in ["selection", "summaries"]}), flush=True)


if __name__ == "__main__":
    main()
