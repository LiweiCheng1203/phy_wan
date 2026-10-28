import argparse
import random

from common import DATA, ROOT, PROMPT, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    report = read_json(ROOT / "reports/qc_report.json")
    decisions = read_json(ROOT / "reports/qc_review.json")
    pending = set(report["review_required"]) - set(decisions)
    if pending:
        raise RuntimeError(f"Missing visual reviews: {sorted(pending)}")
    accepted = sorted(r["episode"] for r in report["episodes"] if decisions.get(r["episode"], {}).get("include", True))
    assert len(accepted) > 22
    random.Random(0).shuffle(accepted)
    split = {"seed": 0, "test": accepted[:12], "val": accepted[12:22], "train": accepted[22:]}
    path = ROOT / "split.json"
    if path.exists() and read_json(path) != split:
        raise RuntimeError("Refusing to overwrite a different frozen split")
    if not args.freeze:
        print(split)
        return
    write_json(path, split)
    excluded = list(read_json(ROOT / "inventory.json")["excluded"])
    excluded += [{"episode": episode, "reason": "visual_qc_failed", "review": decision}
                 for episode, decision in decisions.items() if not decision["include"]]
    write_json(ROOT / "exclusions.json", excluded)
    for name in ["train", "val", "test"]:
        rows = []
        for episode in split[name]:
            folder = DATA / episode
            if name == "train":
                for first in [False] * 7 + [True] * 3:
                    suffix = "_first81" if first else ""
                    rows.append({"file_path": str(folder / f"real{suffix}.mp4"), "control_file_path": str(folder / f"sim{suffix}.mp4"),
                                 "text": PROMPT, "type": "video", "width": 640, "height": 480})
            else:
                rows.append({"episode": episode, "folder": str(folder), "N": read_json(folder / "meta.json")["N"]})
        write_json(ROOT / f"metadata_{name}.json", rows)
    write_json(ROOT / "metadata_smoke.json", read_json(ROOT / "metadata_train.json")[:20])
    print({k: len(split[k]) for k in ["train", "val", "test"]})


if __name__ == "__main__":
    main()
