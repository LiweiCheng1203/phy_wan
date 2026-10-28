"""Replay saved states at real video timestamps; never run physics integration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from common import DATA, ROOT, SOURCE, decode, encode, fingerprint, psnr, read_json, reference_video, sha256, write_json

RENDER = {"version": 2, "fps": 5, "size": [480, 640], "offsamples": 0, "headlight_ambient": 0.45,
          "headlight_diffuse": 0.65, "light_ambient": 0.25, "hidden_groups": [3, 5],
          "shadows": False, "crf": 12}


def inventory():
    included, excluded = [], []
    for path in sorted(SOURCE.glob("episode_*")):
        metrics = read_json(path / "metrics.json")
        item = {"episode": path.name, "source": str(path), "contact_check_pass": metrics.get("contact_check_pass")}
        if item["contact_check_pass"] is True:
            included.append(item)
        else:
            excluded.append({**item, "reason": "contact_check_failed"})
    write_json(ROOT / "inventory.json", {"included": included, "excluded": excluded})
    write_json(ROOT / "exclusions.json", excluded)
    return included


def label_lookup(model, mujoco):
    blue = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue")
    brown = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "brown")
    arm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur_base")
    assert min(blue, brown, arm) > 0
    body_labels = np.zeros(model.nbody, np.uint8)
    for i in range(model.nbody):
        current = i
        while current:
            if current in (blue, brown, arm):
                body_labels[i] = {blue: 2, brown: 3, arm: 1}[current]
                break
            current = int(model.body_parentid[current])
    return body_labels[model.geom_bodyid]


def build(item):
    import mujoco

    begin = time.monotonic()
    source = Path(item["source"])
    out = DATA / item["episode"]
    out.mkdir(parents=True, exist_ok=True)
    real_path = reference_video(item["episode"], item)
    camera = read_json(source / "calibration/cameras.json")
    pts = np.array(read_json(source / "prepared/timeline.json")["image"]["pts_s"])
    inputs = {str(p): sha256(p) for p in [source / "scene_portable.xml", source / "replay.npz",
              source / "calibration/cameras.json", source / "prepared/timeline.json", real_path]}
    # Render resources are shared by the portable XML; include their manifest and hashes.
    assets = source.parent / "assets"
    inputs.update({str(p): sha256(p) for p in sorted(assets.rglob("*")) if p.is_file()})
    signature = fingerprint({"inputs": inputs, "render": RENDER, "code": sha256(__file__)})
    meta_path = out / "meta.json"
    if meta_path.exists():
        previous = read_json(meta_path)
        if previous.get("signature") == signature and all((out / name).is_file() and sha256(out / name) == value
                                                         for name, value in previous.get("output_sha256", {}).items()):
            print("REUSE", item["episode"], flush=True)
            return
    real = decode(real_path)
    assert len(real) == len(pts) >= 81
    assert np.allclose(np.diff(pts), 0.2, atol=1e-6)
    Image.fromarray(real[0]).save(out / "first_frame.png")
    model = mujoco.MjModel.from_xml_path(str(source / "scene_portable.xml"))
    # Multisampling blends object ID colors at silhouettes, corrupting segmentation.
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    with np.load(source / "replay.npz") as track:
        times, qpos = track["time"].copy(), track["qpos"].copy()
    assert np.all(np.diff(times) > 0)
    rotations = []
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            adr = int(model.jnt_qposadr[j]) + (3 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 0)
            rotations.append((adr, Slerp(times, Rotation.from_quat(qpos[:, adr:adr + 4][:, [1, 2, 3, 0]]))))
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.65
    model.light_ambient[:] = 0.25
    option = mujoco.MjvOption()
    option.geomgroup[[3, 5]] = 0
    lookup = label_lookup(model, mujoco)
    sim = np.empty_like(real)
    labels = np.zeros(real.shape[:3], np.uint8)
    offset = camera["external"]["capture_state_offset_s"]
    targets = np.clip(pts + offset, times[0], times[-1])
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        for k, target in enumerate(targets):
            j = int(np.clip(np.searchsorted(times, target) - 1, 0, len(times) - 2))
            alpha = (target - times[j]) / (times[j + 1] - times[j])
            data.qpos[:] = (1 - alpha) * qpos[j] + alpha * qpos[j + 1]
            for adr, slerp in rotations:
                data.qpos[adr:adr + 4] = slerp(target).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="external", scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            sim[k] = renderer.render()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            is_geom = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (segmentation[..., 0] >= 0)
            labels[k][is_geom] = lookup[segmentation[..., 0][is_geom]]
            renderer.disable_segmentation_rendering()
    finally:
        renderer.close()
    assert np.any(labels == 1) and np.any(labels == 2) and np.any(labels == 3)
    products = {}
    for name, frames in [("real.mp4", real), ("sim.mp4", sim), ("real_first81.mp4", real[:81]), ("sim_first81.mp4", sim[:81])]:
        products[name] = encode(out / name, frames)
    first_psnr = psnr(real[0], decode(out / "real.mp4")[0])
    assert first_psnr > 40, first_psnr
    np.savez_compressed(out / "labels.partial.npz", labels=labels)
    (out / "labels.partial.npz").replace(out / "labels.npz")
    output_names = [*products, "first_frame.png", "labels.npz"]
    write_json(meta_path, {**item, "N": len(real), "signature": signature, "inputs": inputs,
                          "render": RENDER, "timestamps": pts.tolist(), "sim_times": targets.tolist(),
                          "products": products, "first_frame_psnr": first_psnr,
                          "output_sha256": {name: sha256(out / name) for name in output_names},
                          "body_mapping": {model.body(i).name: i for i in range(model.nbody)},
                          "seconds": time.monotonic() - begin})
    print("BUILT", item["episode"], len(real), round(time.monotonic() - begin, 2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inventory_only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    items = inventory() if args.num_shards == 1 else read_json(ROOT / "inventory.json")["included"]
    if args.inventory_only:
        return
    for item in items[args.shard::args.num_shards][:args.limit]:
        build(item)


if __name__ == "__main__":
    main()
