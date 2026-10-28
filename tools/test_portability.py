"""CPU checks for migration, frozen splits, and original-video resolution."""
import pytest

import common
from common import read_json, reference_video, segment_seed, segments, sha256, write_json
from prepare_uploaded_data import prepare


@pytest.fixture
def dataset(tmp_path):
    root, data = tmp_path / "program", tmp_path / "uploaded episodes"
    root.mkdir()
    split = {"seed": 0, "train": ["episode_000001"], "val": ["episode_000002"], "test": ["episode_000003"]}
    write_json(root / "split.json", split)
    write_json(root / "exclusions.json", [])
    for episode in [split[key][0] for key in ("train", "val", "test")]:
        folder = data / episode
        folder.mkdir(parents=True)
        names = ("real.mp4", "sim.mp4", "real_first81.mp4", "sim_first81.mp4", "first_frame.png", "labels.npz")
        for name in names:
            (folder / name).write_bytes(f"fixture:{episode}:{name}".encode())
        write_json(folder / "meta.json", {"episode": episode, "N": 164,
                   "output_sha256": {name: sha256(folder / name) for name in names}})
    return root, data


def test_metadata_migration_preserves_split(dataset):
    root, data = dataset
    before = sha256(root / "split.json")
    result = prepare(data, root)
    rows = read_json(root / "metadata_train.json")
    assert result["train_rows"] == 10
    assert sum("first81" in row["file_path"] for row in rows) == 3
    assert all(row["file_path"].startswith(str(data)) for row in rows)
    assert sha256(root / "split.json") == before


def test_corrupt_data_does_not_write_metadata(dataset):
    root, data = dataset
    (data / "episode_000003/labels.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(data, root)
    assert not (root / "metadata_train.json").exists()


def test_excluded_episode_rejected(dataset):
    root, data = dataset
    write_json(root / "exclusions.json", [{"episode": "episode_000001"}])
    with pytest.raises(ValueError, match="excluded"):
        prepare(data, root)


def test_original_reference_override_and_relative_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    prepared = source / "episode_000001/prepared"
    prepared.mkdir(parents=True)
    (prepared / "original.mp4").write_bytes(b"reference")
    write_json(prepared / "manifest.json", {"videos": {"image": {"path": "original.mp4"}}})
    monkeypatch.setattr(common, "SOURCE", source)
    monkeypatch.delenv("SIM2REAL_REAL_VIDEOS", raising=False)
    assert reference_video("episode_000001") == prepared / "original.mp4"
    override = tmp_path / "videos"
    override.mkdir()
    (override / "episode_000001.mp4").write_bytes(b"reference")
    monkeypatch.setenv("SIM2REAL_REAL_VIDEOS", str(override))
    assert reference_video("episode_000001") == override / "episode_000001.mp4"
    with pytest.raises(FileNotFoundError):
        reference_video("episode_000002")


@pytest.mark.parametrize("n", [81, 82, 90, 161, 162, 164])
def test_segment_frames_survive_short_tails(n):
    indices = []
    for i, (start, count, padded) in enumerate(segments(n)):
        assert count <= padded <= 81 and padded % 4 == 1
        indices.extend(range(start + bool(i), start + count))
    assert indices == list(range(n))


def test_episode_seed_independent_of_sharding():
    episodes = [f"episode_{i:06}" for i in range(12)]
    serial = {ep: [segment_seed(42, ep, i) for i in range(3)] for ep in episodes}
    shards = {ep: [segment_seed(42, ep, i) for i in range(3)] for shard in range(4) for ep in episodes[shard::4]}
    assert serial == shards


def test_evaluation_uses_configured_gpu_list(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    repo = tmp_path / "upstream"
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\n" +
        "import json, os, sys\n" +
        "print(json.dumps({'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'argv': sys.argv[1:]}))\n")
    python.chmod(0o755)
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "scripts/eval_shards.sh"
    subprocess.run(["bash", str(script), str(output), "val", "pretrained"], check=True, capture_output=True,
                   env={**os.environ, "SIM2REAL_ROOT": str(tmp_path), "SIM2REAL_REPO": str(repo), "SIM2REAL_GPUS": "2,5"})
    for shard, gpu in enumerate(["2", "5"]):
        calls = [json.loads(line) for line in (output / f"shard_{shard}.log").read_text().splitlines()]
        assert len(calls) == 2 and all(call["gpu"] == gpu for call in calls)
        assert all(call["argv"][call["argv"].index("--shard") + 1] == str(shard) for call in calls)
        assert all(call["argv"][call["argv"].index("--num_shards") + 1] == "2" for call in calls)
