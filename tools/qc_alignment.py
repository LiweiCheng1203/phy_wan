"""Generate alignment measurements and auditable review contact sheets."""
import argparse
import csv
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import DATA, ROOT, HSV, blue_center, centroid, decode, encode, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--aggregate_only", action="store_true")
    args = p.parse_args()
    cv2.setNumThreads(1)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    records = []
    items = read_json(ROOT / "inventory.json")["included"][:args.limit]
    for item in ([] if args.aggregate_only else items[args.shard::args.num_shards]):
        folder = DATA / item["episode"]
        if not (folder / "meta.json").exists():
            raise RuntimeError(f"Dataset not complete: {folder}")
        real, sim = decode(folder / "real.mp4"), decode(folder / "sim.mp4")
        labels = np.load(folder / "labels.npz")["labels"]
        errors = []
        overlay = ((real.astype(np.float32) + sim) * 0.5).astype(np.uint8)
        for k in range(len(real)):
            a, b = blue_center(real[k], labels[k]), centroid(labels[k] == 2)
            if a is not None and b is not None:
                errors.append(float(np.linalg.norm(a - b)))
            for cls, color in [(1, (255, 255, 255)), (2, (0, 255, 255)), (3, (255, 0, 255))]:
                contours, _ = cv2.findContours((labels[k] == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay[k], contours, -1, color, 1)
        median = float(np.median(errors)) if errors else None
        record = {"episode": item["episode"], "median_px": median,
                  "p95_px": float(np.percentile(errors, 95)) if errors else None,
                  "coverage": len(errors) / len(real), "N": len(real)}
        record["flagged"] = median is None or median > 25 or record["coverage"] < 0.5
        records.append(record)
        encode(folder / "overlay_qc.mp4", overlay, crf=18)
        # Sample the full action, not just the static beginning.
        indices = np.linspace(0, len(real) - 1, 8, dtype=int)
        sheet = Image.new("RGB", (1280, 4 * 504))
        draw = ImageDraw.Draw(sheet)
        for j, k in enumerate(indices):
            x, y = (j % 2) * 640, (j // 2) * 504
            sheet.paste(Image.fromarray(overlay[k]), (x, y + 24))
            draw.text((x + 8, y + 5), f"{item['episode']} frame {k} | median={median}", fill="white")
        sheet.save(report_dir / f"{item['episode']}_qc.jpg", quality=90)
        write_json(report_dir / f"{item['episode']}_qc.json", record)
        print("QC", record, flush=True)
    if args.num_shards > 1 and not args.aggregate_only:
        return
    if args.aggregate_only:
        records = [read_json(report_dir / f"{item['episode']}_qc.json") for item in items]
    with open(report_dir / "qc_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    random_ids = random.Random(0).sample([r["episode"] for r in records], min(10, len(records)))
    review = sorted(set(random_ids + [r["episode"] for r in records if r["flagged"]]))
    write_json(report_dir / "qc_report.json", {"hsv": HSV, "episodes": records, "review_required": review})


if __name__ == "__main__":
    main()
