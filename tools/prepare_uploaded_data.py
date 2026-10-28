"""Validate uploaded processed data and regenerate metadata without changing splits."""
import argparse
from pathlib import Path

from common import DATA, ROOT, PROMPT, probe, read_json, reference_video, sha256, write_json


def prepare(data, root, decode_check=False, require_reference=False):
    split_path = root / "split.json"
    split_hash = sha256(split_path)
    split = read_json(split_path)
    ids = [ep for key in ("train", "val", "test") for ep in split[key]]
    if len(set(ids)) != len(ids):
        raise ValueError("Split contains duplicate episodes")
    excluded = {row["episode"] for row in read_json(root / "exclusions.json")}
    if excluded & set(ids):
        raise ValueError("Split contains excluded episodes")
    required = {"real.mp4", "sim.mp4", "real_first81.mp4", "sim_first81.mp4", "first_frame.png", "labels.npz"}
    metadata, records = {}, []
    for key in ("train", "val", "test"):
        rows = []
        for episode in split[key]:
            folder = data / episode
            meta = read_json(folder / "meta.json")
            if meta["episode"] != episode or meta["N"] < 81:
                raise ValueError(f"Invalid episode metadata: {folder}")
            hashes = meta["output_sha256"]
            if not required <= hashes.keys():
                raise ValueError(f"Incomplete output hashes: {folder}")
            for name, expected in hashes.items():
                if Path(name).name != name:
                    raise ValueError(f"Expected output basename, got {name}")
                if sha256(folder / name) != expected:
                    raise ValueError(f"Output hash mismatch: {folder / name}")
            if decode_check:
                for name in sorted(required):
                    if name.endswith(".mp4"):
                        info = probe(folder / name)
                        n = 81 if "first81" in name else meta["N"]
                        if (info["codec_name"], info["width"], info["height"], info["r_frame_rate"], int(info["nb_read_frames"])) != ("h264", 640, 480, "5/1", n):
                            raise ValueError(f"Invalid video: {folder / name}: {info}")
            if require_reference:
                reference = reference_video(episode, meta)
                original_hashes = [value for path, value in meta["inputs"].items() if path.endswith(".mp4")]
                if len(original_hashes) != 1 or sha256(reference) != original_hashes[0]:
                    raise ValueError(f"Original reference video hash mismatch: {reference}")
            if key == "train":
                for first in [False] * 7 + [True] * 3:
                    suffix = "_first81" if first else ""
                    rows.append({"file_path": str(folder / f"real{suffix}.mp4"),
                                 "control_file_path": str(folder / f"sim{suffix}.mp4"),
                                 "text": PROMPT, "type": "video", "width": 640, "height": 480})
            else:
                rows.append({"episode": episode, "folder": str(folder), "N": meta["N"]})
            records.append({"episode": episode, "N": meta["N"], "split": key})
        metadata[key] = rows
    # Validate everything before replacing any operational metadata.
    if sha256(split_path) != split_hash:
        raise RuntimeError("Frozen split changed during validation")
    for key, rows in metadata.items():
        write_json(root / f"metadata_{key}.json", rows)
    write_json(root / "metadata_smoke.json", metadata["train"][:20])
    record = {"counts": {key: len(split[key]) for key in ("train", "val", "test")},
              "train_rows": len(metadata["train"]), "data": str(data), "split_sha256": split_hash,
              "output_hashes_verified": True, "videos_decoded": decode_check,
              "original_references_verified": require_reference, "episodes": records}
    write_json(root / "reports/uploaded_data_check.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode_check", action="store_true", help="Also decode/count all four videos per episode")
    parser.add_argument("--require_reference", action="store_true", help="Verify original RGB videos needed for evaluation")
    args = parser.parse_args()
    result = prepare(DATA, ROOT, args.decode_check, args.require_reference)
    print({key: result[key] for key in ("counts", "train_rows", "output_hashes_verified", "videos_decoded", "original_references_verified")})


if __name__ == "__main__":
    main()
