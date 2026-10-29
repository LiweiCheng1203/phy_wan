"""Run F9 orchestration: 9-frame windows with a noisy condition latent; batch probe, smoke, training."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

from common import REPO, ROOT, read_json, write_json
from experiment import run

STATUS = ROOT / "reports/experiment_f9_status.json"
PROBE = ROOT / "reports/f9_batch_probe.json"
TRAIN = str(ROOT / "scripts/train.sh")
GPUS = "4,5,6,7"
PROBE_BATCHES = [16, 12, 8]
# Run A grew from 62GB in the 4-GPU smoke to 69GB in formal training; keep similar headroom.
PEAK_LIMIT_BYTES = 70e9
SETTINGS = {"SIM2REAL_FRAMES": "9", "SIM2REAL_VAE_MINI_BATCH": "4", "SIM2REAL_CKPT_LIMIT": "6",
            "SIM2REAL_PORT": "29651", "SIM2REAL_COND_NOISE_PROB": "0.75", "SIM2REAL_COND_NOISE_SIGMA": "0.3",
            "SIM2REAL_REPORT_TO": "wandb", "WANDB_PROJECT": "phy_wan",
            # The system profile points this at an unwritable shared directory.
            "WANDB_CACHE_DIR": str(ROOT / ".cache/wandb")}


def command(name, gpus, steps, save_every, resume, metadata, batch, accum, **env):
    # Settings travel in the command itself, so stage signatures cover every hyperparameter.
    settings = {**SETTINGS, "CUDA_VISIBLE_DEVICES": gpus, "SIM2REAL_BATCH": str(batch),
                "SIM2REAL_ACCUM": str(accum), "WANDB_DIR": str(ROOT / "runs" / name), **env}
    return ["env", *[f"{k}={v}" for k, v in settings.items()], "bash", TRAIN, name,
            str(len(gpus.split(","))), str(steps), str(save_every), resume, "ddp", metadata]


def probe_batch():
    if PROBE.exists():
        return read_json(PROBE)["selected"]
    results, selected = [], None
    for batch in PROBE_BATCHES:
        name = f"probe_f9_bs{batch}"
        output = ROOT / "runs" / name
        if output.exists():
            shutil.rmtree(output)
        cmd = command(name, GPUS, 3, 1000, "none", "metadata_smoke_f9.json", batch, 1, WANDB_MODE="offline")
        log = ROOT / "logs" / f"f9_{name}.log"
        write_json(STATUS, {"stage": f"f9_{name}", "status": "running", "command": cmd, "started_unix": time.time()})
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
    # Fallback keeps the global batch of the smallest probe at half the activation memory.
    selected = selected or {"batch": 4, "accum": 2}
    write_json(PROBE, {"limit_gb": PEAK_LIMIT_BYTES / 1e9, "results": results, "selected": selected})
    return selected


def wandb_logged_in():
    check = "import sys, wandb; sys.exit(0 if wandb.login(anonymous='never', verify=True) else 1)"
    return subprocess.run([str(REPO / ".venv/bin/python"), "-c", check], timeout=180,
                          env={**os.environ, "WANDB_SILENT": "true"}).returncode == 0


def latest_checkpoint(run_dir):
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1])) if run_dir.exists() else []
    checkpoints = [c for c in checkpoints if (c / "optimizer.bin").exists() and (c / "transformer").exists()]
    return checkpoints[-1].name if checkpoints else "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--until", choices=["probe", "smoke", "train"], default="train")
    args = p.parse_args()
    for name in ["metadata_train_f9.json", "metadata_smoke_f9.json"]:
        assert (ROOT / name).exists(), f"Run tools/make_window_metadata.py first: {name}"
    selected = probe_batch()
    batch, accum = selected["batch"], selected["accum"]
    print("BATCH", selected, flush=True)
    if args.until == "probe":
        return
    smoke, offline = "metadata_smoke_f9.json", {"WANDB_MODE": "offline"}
    single = GPUS.split(",")[0]
    run("f9_smoke_single_10", command("smoke_f9_single", single, 10, 10, "none", smoke, batch, accum, **offline), status=STATUS)
    run("f9_smoke_single_resume_20", command("smoke_f9_single", single, 20, 10, "checkpoint-10", smoke, batch, accum, **offline), status=STATUS)
    run("f9_smoke_ddp_4", command("smoke_f9_ddp", GPUS, 4, 2, "none", smoke, batch, accum, **offline), status=STATUS)
    run("f9_smoke_ddp_resume_6", command("smoke_f9_ddp", GPUS, 6, 2, "checkpoint-4", smoke, batch, accum, **offline), status=STATUS)
    if args.until == "smoke":
        write_json(STATUS, {"stage": "smoke", "status": "complete", "batch": selected})
        return
    if not wandb_logged_in():
        write_json(STATUS, {"stage": "f9_run_3000", "status": "waiting_for_wandb_login"})
        raise SystemExit("wandb is not logged in on this machine; run `wandb login` and start again")
    run_dir = ROOT / "runs/run_f9"
    run_dir.mkdir(parents=True, exist_ok=True)
    id_file = run_dir / "wandb_run_id.txt"
    if not id_file.exists():
        id_file.write_text(f"run_f9_{int(time.time())}\n")
    # A fixed run id with resume=allow continues the same wandb curves after an interruption.
    online = {"WANDB_MODE": "online", "WANDB_NAME": "run_f9", "WANDB_RUN_ID": id_file.read_text().strip(),
              "WANDB_RESUME": "allow", "WANDB_INIT_TIMEOUT": "180"}
    resume = latest_checkpoint(run_dir)
    run("f9_run_3000", command("run_f9", GPUS, 3000, 500, resume, "metadata_train_f9.json", batch, accum, **online), status=STATUS)
    write_json(STATUS, {"stage": "complete", "status": "complete", "batch": selected,
                        "checkpoints": sorted(c.name for c in run_dir.glob("checkpoint-*"))})


if __name__ == "__main__":
    main()
