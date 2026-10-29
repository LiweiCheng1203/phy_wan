"""Resumable Run A orchestration: smoke, baselines, 3000 updates, selection, test."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

from common import HSV, REPO, ROOT, fingerprint, read_json, sha256, write_json


def run(stage, cmd, env=None, status=None):
    status = status or ROOT / "reports/experiment_status.json"
    done = ROOT / "runs/stages" / (stage + ".json")
    signature = fingerprint(cmd)
    if done.exists():
        assert read_json(done)["signature"] == signature, f"Stage command changed: {stage}"
        print("STAGE_REUSE", stage, flush=True)
        return
    write_json(status, {"stage": stage, "status": "running", "command": cmd, "started_unix": time.time()})
    print("STAGE_BEGIN", stage, cmd, flush=True)
    begin = time.monotonic()
    log = ROOT / "logs" / (stage + ".log")
    try:
        with open(log, "a") as f:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    except BaseException as exc:
        write_json(status, {"stage": stage, "status": "failed", "log": str(log), "error": str(exc)})
        print(f"STAGE_FAILED {stage}: {log}", flush=True)
        print("\n".join(log.read_text(errors="replace").splitlines()[-80:]), flush=True)
        raise
    write_json(done, {"signature": signature, "seconds": time.monotonic() - begin, "command": cmd, "log": str(log)})
    print("STAGE_DONE", stage, round(time.monotonic() - begin, 1), flush=True)


def rank_candidates(candidates):
    remaining = list(candidates)
    selected = []
    while remaining:
        best = min(r["action_score"] for r in remaining)
        band = [r for r in remaining if r["action_score"] <= best + 2]
        winner = min(band, key=lambda r: (r["fg_lpips"], r["action_score"], r.get("step", 0), r["name"]))
        selected.append(winner)
        remaining.remove(winner)
    return selected


def candidate(name, output, **info):
    summary = read_json(output / "summary.json")["main"]
    return {"name": name, "output": str(output), "action_score": summary["action_score"]["mean"],
            "fg_lpips": summary["fg_lpips"]["mean"], **info}


def evaluation(stage, output, split, ckpt, cfg=1, shift=3, first=False):
    run(stage, ["bash", str(ROOT / "scripts/eval_shards.sh"), str(output), split, str(ckpt), str(cfg), str(shift), "first" if first else "full"])


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--memory", choices=["ddp", "low_vram"], default="ddp")
    p.add_argument("--skip_smoke", action="store_true")
    args = p.parse_args()
    assert (ROOT / "split.json").exists() and (ROOT / "reports/environment.lock.txt").exists()
    python = str(REPO / ".venv/bin/python")
    train = str(ROOT / "scripts/train.sh")
    if not args.skip_smoke:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("SIM2REAL_GPUS", "4,5,6,7").split(",")[0]}
        run("smoke_single_10", ["bash", train, "smoke_single", "1", "10", "10", "none", args.memory, "metadata_smoke.json"], env)
        run("smoke_single_resume_20", ["bash", train, "smoke_single", "1", "20", "10", "checkpoint-10", args.memory, "metadata_smoke.json"], env)
        run("smoke_ddp_4", ["bash", train, "smoke_ddp", "4", "4", "2", "none", args.memory, "metadata_smoke.json"])
        run("smoke_ddp_resume_6", ["bash", train, "smoke_ddp", "4", "6", "2", "checkpoint-4", args.memory, "metadata_smoke.json"])
    evaluation("val_static", ROOT / "eval/val_static", "val", "static")
    # Zero-shot validation is completed before formal training starts.
    baseline = []
    for cfg in [1, 3]:
        for shift in [3, 5]:
            name = f"val_pretrained_cfg{cfg}_shift{shift}"
            output = ROOT / "eval" / name
            evaluation(name, output, "val", "pretrained", cfg, shift)
            baseline.append(candidate(name, output, ckpt="pretrained", cfg=cfg, shift=shift))
    run_a = ROOT / "runs/run_a"
    # Resume only this independent formal run, which starts from pretrained weights.
    if not (ROOT / "runs/stages/run_a_3000.json").exists():
        checkpoints = sorted(run_a.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1])) if run_a.exists() else []
        checkpoints = [c for c in checkpoints if (c / "optimizer.bin").exists() and (c / "transformer").exists()]
        resume = checkpoints[-1].name if checkpoints else "none"
        # Use a stable command for the completed stage record; resume is execution state.
        cmd = ["bash", train, "run_a", "4", "3000", "250", resume, args.memory]
        run("run_a_3000", cmd)
    screening = []
    for step in [500, 1000, 1500, 2000, 2500, 3000]:
        ckpt = run_a / f"checkpoint-{step}"
        name = f"val_screen_{step}"
        output = ROOT / "eval" / name
        evaluation(name, output, "val", ckpt, first=True)
        screening.append(candidate(name, output, ckpt=str(ckpt), step=step))
    top_two = rank_candidates(screening)[:2]
    write_json(ROOT / "reports/checkpoint_screening.json", {"candidates": screening, "selected": top_two})
    full = []
    for selected in top_two:
        for cfg in [1, 3]:
            for shift in [3, 5]:
                name = f"val_full_{selected['step']}_cfg{cfg}_shift{shift}"
                output = ROOT / "eval" / name
                evaluation(name, output, "val", selected["ckpt"], cfg, shift)
                full.append(candidate(name, output, ckpt=selected["ckpt"], step=selected["step"], cfg=cfg, shift=shift))
    final, zero = rank_candidates(full)[0], rank_candidates(baseline)[0]
    frozen = {"best": final, "pretrained": zero, "full_candidates": full, "baseline_candidates": baseline,
        "rule": "action_score=episode_macro(median_penalized+0.25*p95_penalized); within 2px choose fg_lpips, then action, then earlier step",
        "split_sha256": sha256(ROOT / "split.json"), "hsv": HSV,
        "inference_sha256": sha256(ROOT / "tools/infer.py"), "metric_sha256": sha256(ROOT / "tools/evaluate.py")}
    frozen_path = ROOT / "configs/final_selection.json"
    if frozen_path.exists():
        assert read_json(frozen_path) == frozen, "Final configuration already frozen"
    write_json(frozen_path, frozen)
    evaluation("test_static", ROOT / "eval/test_static", "test", "static")
    evaluation("test_pretrained", ROOT / "eval/test_pretrained", "test", "pretrained", zero["cfg"], zero["shift"])
    evaluation("test_best", ROOT / "eval/test_best", "test", final["ckpt"], final["cfg"], final["shift"])
    run("final_report", [python, str(ROOT / "tools/report.py")])
    write_json(ROOT / "reports/experiment_status.json", {"stage": "complete", "status": "complete", "selection": frozen})


if __name__ == "__main__":
    main()
