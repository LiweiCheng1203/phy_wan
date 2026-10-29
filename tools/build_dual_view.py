"""Side-by-side external | wrist episodes for dual-view training.

The real wrist stream (Berkeley UR5 hand_image) stores B and R swapped; it is corrected to RGB here.
Simulated wrist renders are already RGB. Frames are written losslessly, so the pairing is exact.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from common import DATA, decode, encode, psnr, read_json, sha256, write_json

HEIGHT, WIDTH = 480, 1280


def build(episode, external, wrist, output):
    sources = {"external": Path(external) / episode, "wrist": Path(wrist) / episode}
    out = Path(output) / episode
    inputs = {f"{view}/{name}": str(sources[view] / name) for view in sources
              for name in ["real.mp4", "sim.mp4", "first_frame.png"]}
    input_hashes = {key: sha256(path) for key, path in inputs.items()}
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta["inputs"] == input_hashes and all(sha256(out / n) == h for n, h in meta["outputs"].items()):
            return episode, "reuse"
    out.mkdir(parents=True, exist_ok=True)
    n = read_json(sources["external"] / "meta.json")["N"]
    for kind in ["real", "sim"]:
        left = decode(sources["external"] / f"{kind}.mp4")
        right = decode(sources["wrist"] / f"{kind}.mp4")
        assert len(left) == len(right) == n, (episode, kind, len(left), len(right), n)
        if kind == "real":
            right = right[..., ::-1]
        frames = np.ascontiguousarray(np.concatenate([left, right], axis=2))
        encode(out / f"{kind}.mp4", frames, lossless=True)
        assert np.array_equal(decode(out / f"{kind}.mp4", HEIGHT, WIDTH), frames), (episode, kind)
        if kind == "real":
            real_first = frames[0]
    first = np.concatenate([np.array(Image.open(sources["external"] / "first_frame.png").convert("RGB")),
                            np.array(Image.open(sources["wrist"] / "first_frame.png").convert("RGB"))[..., ::-1]], axis=1)
    # The PNG and video come from the same decoded frame, so the channel correction must agree.
    quality = psnr(first, real_first)
    assert quality > 35, (episode, quality)
    Image.fromarray(np.ascontiguousarray(first)).save(out / "first_frame.png")
    write_json(meta_path, {"episode": episode, "N": n, "size": [HEIGHT, WIDTH], "layout": "left external | right wrist",
                           "wrist_real_rb_swapped": True, "first_frame_vs_video_psnr_db": quality,
                           "inputs": input_hashes, "input_paths": inputs,
                           "outputs": {name: sha256(out / name) for name in ["real.mp4", "sim.mp4", "first_frame.png"]}})
    return episode, "built"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--external", type=Path, default=DATA)
    p.add_argument("--wrist", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    episodes = sorted(d.name for d in args.external.iterdir() if d.name.startswith("episode_"))[:args.limit]
    wrist = sorted(d.name for d in args.wrist.iterdir() if d.name.startswith("episode_"))
    assert args.limit or episodes == wrist, "External and wrist episode sets differ"
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(build, episodes, *[[str(x)] * len(episodes) for x in [args.external, args.wrist, args.output]]))
    write_json(args.output / "build_summary.json", {"episodes": len(results), "status": dict(results)})
    print({status: sum(s == status for _, s in results) for status in ["built", "reuse"]})


if __name__ == "__main__":
    main()
