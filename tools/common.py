"""Shared video IO, frame indexing, and task metric definitions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(os.environ.get("SIM2REAL_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
DATA = Path(os.environ.get("SIM2REAL_DATA", str(ROOT / "episodes"))).resolve()
SOURCE = Path(os.environ.get("SIM2REAL_SOURCE", str(ROOT / "sources"))).resolve()
REPO = Path(os.environ.get("SIM2REAL_REPO", str(ROOT.parent / "VideoX-Fun"))).resolve()
MODEL = Path(os.environ.get("SIM2REAL_MODEL", str(REPO / "model/Wan2.1-Fun-V1.1-1.3B-Control"))).resolve()
PROMPT = "A real-world video of a UR5 robot arm with a two-finger gripper picking up the blue cup and putting it into the brown cup on a wooden table."
HSV = {"lower": [90, 60, 40], "upper": [135, 255, 255], "min_pixels": 20}


def read_json(path):
    return json.loads(Path(path).read_text())


def reference_video(episode, meta=None):
    """Resolve original RGB video without modifying historical input provenance."""
    override = os.environ.get("SIM2REAL_REAL_VIDEOS")
    if override:
        path = Path(override) / f"{episode}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    source = SOURCE / episode
    if not (source / "prepared/manifest.json").is_file() and meta and meta.get("source"):
        source = Path(meta["source"])
    manifest = source / "prepared/manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; set SIM2REAL_SOURCE or SIM2REAL_REAL_VIDEOS")
    path = Path(read_json(manifest)["videos"]["image"]["path"])
    if not path.is_absolute():
        path = manifest.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing original RGB video {path}; set SIM2REAL_REAL_VIDEOS")
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path)]))["streams"][0]


def decode(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
        "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, 480, 640, 3).copy()


def encode(path, frames, crf=12):
    path = Path(path)
    temp = path.with_name(path.stem + ".partial.mp4")
    height, width = frames.shape[1:3]
    p = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "5", "-i", "-", "-an", "-c:v", "libx264",
        "-threads", "1", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temp)], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        p.stdin.close()
        code = p.wait()
    if code:
        raise RuntimeError(f"ffmpeg writer exit {code}: {path}")
    info = probe(temp)
    assert (info["codec_name"], info["r_frame_rate"]) == ("h264", "5/1"), info
    assert (info["width"], info["height"], int(info["nb_read_frames"])) == (width, height, len(frames)), info
    temp.replace(path)
    return info


def segments(n):
    if n < 2:
        raise ValueError("At least two frames required")
    start = 0
    while start < n - 1:
        count = min(81, n - start)
        padded = 4 * ((count - 1 + 3) // 4) + 1
        yield start, count, padded
        start += count - 1


def segment_seed(seed, episode, index):
    return int.from_bytes(hashlib.sha256(f"{seed}:{episode}:{index}".encode()).digest()[:8], "little") % (2**63 - 1)


def dilate(mask, radius):
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)).astype(bool)


def centroid(mask, minimum=20):
    y, x = np.nonzero(mask)
    if len(x) < minimum:
        return None
    return np.array([x.mean(), y.mean()])


def blue_center(frame, label):
    roi = dilate(label == 2, 40)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(HSV["lower"], np.uint8), np.array(HSV["upper"], np.uint8)) > 0
    return centroid(mask & roi, HSV["min_pixels"])


def psnr(real, gen, mask=None):
    error = (real.astype(np.float64) / 255 - gen.astype(np.float64) / 255) ** 2
    mse = error[mask].mean() if mask is not None else error.mean()
    return float(-10 * np.log10(max(float(mse), 1e-12)))
