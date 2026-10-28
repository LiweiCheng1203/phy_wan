"""Frame metrics followed by episode-equally-weighted aggregation and bootstrap CIs."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from common import DATA, ROOT, HSV, blue_center, decode, dilate, encode, fingerprint, read_json, reference_video, sha256, write_json


def metric_network(device="cpu"):
    import torch
    import lpips
    # Login shells may inherit another user's unwritable TORCH_HOME.
    cache = Path(os.environ.get("SIM2REAL_TORCH_HOME", str(ROOT / ".cache/torch"))).expanduser()
    hub = cache / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub))
    return lpips.LPIPS(net="alex", spatial=True).eval().to(device)


def average(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def summarize_frames(frames):
    result = {key: average([f[key] for f in frames]) for key in ["psnr", "ssim", "lpips", "fg_psnr", "fg_lpips"]}
    valid = [f["centroid_px"] for f in frames if f["centroid_px"] is not None]
    penalized = [f["penalized_centroid_px"] for f in frames if f["penalized_centroid_px"] is not None]
    for prefix, values in [("centroid", valid), ("penalized_centroid", penalized)]:
        result[prefix + "_median"] = float(np.median(values)) if values else None
        result[prefix + "_p95"] = float(np.percentile(values, 95)) if values else None
    ref_count = sum(f["reference_detected"] for f in frames)
    failed = sum(f["reference_detected"] and not f["generated_detected"] for f in frames)
    result.update(reference_detection_rate=ref_count / len(frames) if frames else None,
                  detection_failure_rate=failed / ref_count if ref_count else None,
                  reference_detected_frames=ref_count, detection_failed_frames=failed, N=len(frames))
    result["action_score"] = (result["penalized_centroid_median"] + 0.25 * result["penalized_centroid_p95"]) if penalized else None
    return result


def episode_metrics(folder, output, network, device, static=False):
    import torch
    import torch.nn.functional as F
    reference = reference_video(folder.name, read_json(folder / "meta.json"))
    real = decode(reference)
    labels = np.load(folder / "labels.npz")["labels"]
    if static:
        generated = np.broadcast_to(np.array(Image.open(folder / "first_frame.png"))[None], real.shape).astype(np.float32) / 255
        output.mkdir(parents=True, exist_ok=True)
        sim = decode(folder / "sim.mp4")
        export = (generated * 255).round().astype(np.uint8)
        encode(output / "generated.mp4", export)
        encode(output / "triptych.mp4", np.concatenate([real, sim, export], axis=2))
    else:
        generated = np.load(output / "frames.npy", mmap_mode="r")
    assert generated.shape[1:] == (480, 640, 3) and generated.shape[0] <= len(real)
    assert np.isfinite(generated).all() and generated.min() >= 0 and generated.max() <= 1
    n = len(generated)
    real = real[:n]
    mask = np.stack([dilate(label > 0, 16) for label in labels[:n]])
    distances, fg_distances = [], []
    for start in range(0, n, 4):
        target = torch.from_numpy(real[start:start + 4].copy()).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
        pred = torch.from_numpy(np.array(generated[start:start + 4])).permute(0, 3, 1, 2).to(device) * 2 - 1
        with torch.inference_mode():
            network.spatial = False
            global_distance = network(target, pred)
            network.spatial = True
            spatial = network(target, pred)
            m = torch.from_numpy(mask[start:start + 4].copy()).to(device).float()[:, None]
            if m.shape[-2:] != spatial.shape[-2:]:
                m = F.interpolate(m, size=spatial.shape[-2:], mode="area")
            assert torch.all(m.sum((1, 2, 3)) > 0)
            distances.extend(global_distance.flatten().cpu().tolist())
            fg_distances.extend(((spatial * m).sum((1, 2, 3)) / m.sum((1, 2, 3))).cpu().tolist())
    frame_stats = []
    for k in range(n):
        ref = real[k].astype(np.float32) / 255
        gen = np.asarray(generated[k])
        error = (ref - gen).astype(np.float64) ** 2
        a = blue_center(real[k], labels[k])
        b = blue_center((gen * 255).round().astype(np.uint8), labels[k])
        e = float(np.linalg.norm(a - b)) if a is not None and b is not None else None
        frame_stats.append({"frame": k, "psnr": float(-10 * np.log10(max(error.mean(), 1e-12))),
            "fg_psnr": float(-10 * np.log10(max(error[mask[k]].mean(), 1e-12))),
            "ssim": float(structural_similarity(ref, gen, data_range=1., channel_axis=2)),
            "lpips": distances[k], "fg_lpips": fg_distances[k], "centroid_px": e,
            "penalized_centroid_px": e if e is not None else (800. if a is not None else None),
            "reference_detected": a is not None, "generated_detected": b is not None})
    seam_indices = list(range(81, n, 80))
    seams = []
    for k in seam_indices:
        a, b = np.asarray(generated[k - 1]), np.asarray(generated[k])
        seam = {"first_nonoverlap_frame": k, "generated_mean_abs_change": float(np.abs(a - b).mean()),
                "real_mean_abs_change": float(np.abs(real[k].astype(np.float32) - real[k - 1]).mean() / 255),
                "before_metrics": frame_stats[k - 1], "after_metrics": frame_stats[k]}
        seams.append(seam)
        lo, hi = max(0, k - 5), min(n, k + 6)
        sim = decode(folder / "sim.mp4")[lo:hi]
        frames = (np.asarray(generated[lo:hi]) * 255).round().astype(np.uint8)
        encode(output / f"seam_{k}.mp4", np.concatenate([real[lo:hi], sim, frames], axis=2))
    result = {"episode": folder.name, "hsv": HSV, "N": n, "main": summarize_frames(frame_stats[1:]),
              "first_frame": frame_stats[0], "first_segment": summarize_frames(frame_stats[1:81]),
              "later_segments": summarize_frames(frame_stats[81:]), "seams": seams,
              "per_segment": [summarize_frames(frame_stats[max(1, start):min(n, start + 80)])
                              for start in range(1, n, 80)], "frames": frame_stats,
              "reference_sha256": sha256(reference)}
    write_json(output / "metrics.json", result)
    return result


def aggregate(results):
    rng = np.random.default_rng(0)
    groups = {}
    for group in ["main", "first_segment", "later_segments", "first_frame"]:
        keys = sorted(set().union(*(r[group].keys() for r in results)))
        stats = {}
        for key in keys:
            if key in ["frame", "reference_detected", "generated_detected"]:
                continue
            values = np.array([r[group].get(key) for r in results if r[group].get(key) is not None], dtype=float)
            if len(values):
                draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
                stats[key] = {"mean": float(values.mean()), "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "episodes": len(values)}
        groups[group] = stats
    return {"episodes": len(results), "weighting": "equal_episode", "bootstrap_seed": 0, "bootstrap_samples": 10000, **groups}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--static", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--aggregate_only", action="store_true")
    p.add_argument("--prepare_only", action="store_true", help="Prepare LPIPS weights on CPU before launching evaluation workers")
    args = p.parse_args()
    if args.prepare_only:
        metric_network()
        print("LPIPS_READY", flush=True)
        return
    episodes = read_json(ROOT / "split.json")[args.split]
    if not args.aggregate_only:
        import torch
        torch.set_num_threads(4)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        network = metric_network(device)
        for episode in episodes[args.shard::args.num_shards]:
            output = args.output / episode
            metrics = output / "metrics.json"
            # Metric changes invalidate cached results; frame provenance remains in generation.json.
            reference = reference_video(episode, read_json(DATA / episode / "meta.json"))
            signature = fingerprint({"code": sha256(__file__), "common": sha256(Path(__file__).with_name("common.py")),
                "generation": sha256(output / "generation.json") if not args.static else "static_first_frame",
                "reference": sha256(reference),
                "labels": sha256(DATA / episode / "labels.npz")})
            if metrics.exists() and read_json(metrics).get("metric_signature") == signature:
                print("REUSE_METRICS", episode, flush=True)
                continue
            result = episode_metrics(DATA / episode, output, network, device, args.static)
            result["metric_signature"] = signature
            write_json(metrics, result)
            print("METRICS", episode, result["main"], flush=True)
    if args.num_shards == 1 or args.aggregate_only:
        results = [read_json(args.output / e / "metrics.json") for e in episodes]
        write_json(args.output / "summary.json", aggregate(results))
        with open(args.output / "episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", *results[0]["main"]])
            writer.writeheader()
            writer.writerows({"episode": r["episode"], **r["main"]} for r in results)


if __name__ == "__main__":
    main()
