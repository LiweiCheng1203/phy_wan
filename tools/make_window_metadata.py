"""Short-window training metadata: whole train episodes, a random window start per sample."""
import argparse
from pathlib import Path

from common import DATA, DUAL_PROMPT, PROMPT, ROOT, read_json, write_json


def episode_rows(episodes, repeats, frames, data=DATA, prompt=PROMPT):
    rows = []
    for episode in episodes:
        folder = data / episode
        for name in ["real.mp4", "sim.mp4", "meta.json"]:
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        meta = read_json(folder / "meta.json")
        n = meta["N"]
        height, width = meta.get("size", [480, 640])
        if n < frames:
            raise ValueError(f"{episode} has {n} frames, fewer than the {frames}-frame window")
        # The upstream dataset draws a new uniform window start each time a row is loaded.
        rows += [{"file_path": str(folder / "real.mp4"), "control_file_path": str(folder / "sim.mp4"),
                  "text": prompt, "type": "video", "width": width, "height": height} for _ in range(repeats)]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="f9")
    p.add_argument("--frames", type=int, default=9)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--smoke_episodes", type=int, default=8)
    p.add_argument("--smoke_repeats", type=int, default=16)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--prompt", choices=["single", "dual"], default="single")
    args = p.parse_args()
    train = read_json(ROOT / "split.json")["train"]
    prompt = DUAL_PROMPT if args.prompt == "dual" else PROMPT
    rows = episode_rows(train, args.repeats, args.frames, args.data, prompt)
    smoke = episode_rows(train[:args.smoke_episodes], args.smoke_repeats, args.frames, args.data, prompt)
    write_json(ROOT / f"metadata_train_{args.name}.json", rows)
    write_json(ROOT / f"metadata_smoke_{args.name}.json", smoke)
    print({"episodes": len(train), "train_rows": len(rows), "smoke_rows": len(smoke)})


if __name__ == "__main__":
    main()
