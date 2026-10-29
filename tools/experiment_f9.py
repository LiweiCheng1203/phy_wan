"""Run F9 orchestration: 9-frame windows with a noisy condition latent; batch probe, smoke, training.

Variant `f9` trains the external view at 480x640; `f9dual` trains external | wrist side by side at 480x1280.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

from common import REPO, ROOT, read_json, write_json
from experiment import run

TRAIN = str(ROOT / "scripts/train.sh")
GPUS = "4,5,6,7"
# Run A grew from 62GB in the 4-GPU smoke to 69GB in formal training; keep similar headroom.
PEAK_LIMIT_BYTES = 70e9
SETTINGS = {"SIM2REAL_FRAMES": "9", "SIM2REAL_VAE_MINI_BATCH": "4", "SIM2REAL_CKPT_LIMIT": "6",
            "SIM2REAL_PORT": "29651", "SIM2REAL_COND_NOISE_PROB": "0.75", "SIM2REAL_COND_NOISE_SIGMA": "0.3",
            "SIM2REAL_REPORT_TO": "wandb", "WANDB_PROJECT": "phy_wan",
            # The system profile points this at an unwritable shared directory.
            "WANDB_CACHE_DIR": str(ROOT / ".cache/wandb")}
# Fallbacks keep the smallest probe's global batch at half the activation memory.
VARIANTS = {"f9": {"env": {}, "probe": [16, 12, 8], "fallback": {"batch": 4, "accum": 2}},
            "f9dual": {"env": {"SIM2REAL_WIDTH": "1280"}, "probe": [8, 6, 4], "fallback": {"batch": 2, "accum": 2}}}


def command(variant, name, gpus, steps, save_every, resume, metadata, batch, accum, **env):
    # Settings travel in the command itself, so stage signatures cover every hyperparameter.
    settings = {**SETTINGS, **VARIANTS[variant]["env"], "CUDA_VISIBLE_DEVICES": gpus, "SIM2REAL_BATCH": str(batch),
                "SIM2REAL_ACCUM": str(accum), "WANDB_DIR": str(ROOT / "runs" / name), **env}
    return ["env", *[f"{k}={v}" for k, v in settings.items()], "bash", TRAIN, name,
            str(len(gpus.split(","))), str(steps), str(save_every), resume, "ddp", metadata]


def probe_batch(variant, status):
    probe = ROOT / f"reports/{variant}_batch_probe.json"
    if probe.exists():
        return read_json(probe)["selected"]
    results, selected = [], None
    for batch in VARIANTS[variant]["probe"]:
        name = f"probe_{variant}_bs{batch}"
        output = ROOT / "runs" / name
        if output.exists():
            shutil.rmtree(output)
        cmd = command(variant, name, GPUS, 3, 1000, "none", f"metadata_smoke_{variant}.json", batch, 1, WANDB_MODE="offline")
        log = ROOT / "logs" / f"{variant}_{name}.log"
        write_json(status, {"stage": f"{variant}_{name}", "status": "running", "command": cmd, "started_unix": time.time()})
        print("PROBE_BEGIN", batch, flush=True)
        begin = time.monotonic()
        with open(log, "w") as f:
            # An out-of-memory exit means this batch does not fit, not that the experiment failed.
            code = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=3600).returncode
        events = [json.loads(line) for path in sorted((output / "audit").glob("rank*.jsonl")) for line in path.open()]
        peaks = [e["peak_cuda_bytes"] for e in events if e["event"] == "completed"]
        timing = [e["values"]["time/sec_per_step"] for e in events
                  if e["event"] == "training" and "time/sec_per_step" in e["values"]]
        fits = code == 0 and len(peaks) == len(GPUS.split(",")) and max(peaks) <= PEAK_LIMIT_BYTES
        results.append({"batch": batch, "exit_code": code, "peak_gb": max(peaks) / 1e9 if peaks else None,
                        "sec_per_step": timing[-1] if timing else None, "fits": fits,
                        "seconds": time.monotonic() - begin, "log": str(log)})
        print("PROBE_RESULT", results[-1], flush=True)
        # Probe checkpoints are full training states (~19GB); keep only the audit and logs.
        for path in [*output.glob("checkpoint-*"), output / ".rank_random_states"]:
            if path.exists():
                shutil.rmtree(path)
        if fits:
            selected = {"batch": batch, "accum": 1}
            break
    selected = selected or VARIANTS[variant]["fallback"]
    write_json(probe, {"limit_gb": PEAK_LIMIT_BYTES / 1e9, "results": results, "selected": selected})
    return selected


def wandb_logged_in():
    check = "import sys, wandb; sys.exit(0 if wandb.login(verify=True) else 1)"
    return subprocess.run([str(REPO / ".venv/bin/python"), "-c", check], timeout=180, stdin=subprocess.DEVNULL,
                          env={**os.environ, "WANDB_SILENT": "true"}).returncode == 0


def latest_checkpoint(run_dir):
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1])) if run_dir.exists() else []
    checkpoints = [c for c in checkpoints if (c / "optimizer.bin").exists() and (c / "transformer").exists()]
    return checkpoints[-1].name if checkpoints else "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS), default="f9")
    p.add_argument("--until", choices=["probe", "smoke", "train"], default="train")
    args = p.parse_args()
    v = args.variant
    status = ROOT / f"reports/experiment_{v}_status.json"
    smoke, train = f"metadata_smoke_{v}.json", f"metadata_train_{v}.json"
    for name in [train, smoke]:
        assert (ROOT / name).exists(), f"Run tools/make_window_metadata.py first: {name}"
    selected = probe_batch(v, status)
    batch, accum = selected["batch"], selected["accum"]
    print("BATCH", selected, flush=True)
    if args.until == "probe":
        return
    offline = {"WANDB_MODE": "offline"}
    single = GPUS.split(",")[0]
    run(f"{v}_smoke_single_10", command(v, f"smoke_{v}_single", single, 10, 10, "none", smoke, batch, accum, **offline), status=status)
    run(f"{v}_smoke_single_resume_20", command(v, f"smoke_{v}_single", single, 20, 10, "checkpoint-10", smoke, batch, accum, **offline), status=status)
    run(f"{v}_smoke_ddp_4", command(v, f"smoke_{v}_ddp", GPUS, 4, 2, "none", smoke, batch, accum, **offline), status=status)
    run(f"{v}_smoke_ddp_resume_6", command(v, f"smoke_{v}_ddp", GPUS, 6, 2, "checkpoint-4", smoke, batch, accum, **offline), status=status)
    if args.until == "smoke":
        write_json(status, {"stage": "smoke", "status": "complete", "batch": selected})
        return
    if not wandb_logged_in():
        write_json(status, {"stage": f"{v}_run_3000", "status": "waiting_for_wandb_login"})
        raise SystemExit("wandb is not logged in on this machine; run `wandb login` and start again")
    run_dir = ROOT / f"runs/run_{v}"
    run_dir.mkdir(parents=True, exist_ok=True)
    id_file = run_dir / "wandb_run_id.txt"
    if not id_file.exists():
        id_file.write_text(f"run_{v}_{int(time.time())}\n")
    # A fixed run id with resume=allow continues the same wandb curves after an interruption.
    online = {"WANDB_MODE": "online", "WANDB_NAME": f"run_{v}", "WANDB_RUN_ID": id_file.read_text().strip(),
              "WANDB_RESUME": "allow", "WANDB_INIT_TIMEOUT": "180"}
    resume = latest_checkpoint(run_dir)
    run(f"{v}_run_3000", command(v, f"run_{v}", GPUS, 3000, 500, resume, train, batch, accum, **online), status=status)
    write_json(status, {"stage": "complete", "status": "complete", "batch": selected,
                        "checkpoints": sorted(c.name for c in run_dir.glob("checkpoint-*"))})


if __name__ == "__main__":
    main()
