"""Tests for the dangerous indexing, scheduling, loading, and ranking contracts."""
import numpy as np
import pytest

from common import segment_seed, segments
from evaluate import summarize_frames
from infer import scheduler_for, strict_load


@pytest.mark.parametrize("n", [81, 82, 90, 161, 162, 164])
def test_segment_coverage(n):
    output = []
    for i, (start, count, padded) in enumerate(segments(n)):
        assert count <= 81 and count <= padded <= 81 and padded % 4 == 1
        indices = list(range(start, start + count))
        output += indices if i == 0 else indices[1:]
    assert output == list(range(n))
    if n == 164:
        assert list(segments(n)) == [(0, 81, 81), (80, 81, 81), (160, 4, 5)]


def test_seeds_independent_of_shards_and_order():
    episodes = [f"episode_{i:06}" for i in range(12)]
    serial = {ep: [segment_seed(42, ep, i) for i in range(3)] for ep in episodes}
    parallel = {ep: [segment_seed(42, ep, i) for i in range(3)] for shard in range(4) for ep in episodes[shard::4]}
    assert serial == parallel
    assert len({s for seeds in serial.values() for s in seeds}) == 36


def test_actual_shift_changes_schedule():
    a, b = scheduler_for({"num_train_timesteps": 1000}, 3), scheduler_for({"num_train_timesteps": 1000}, 5)
    a.set_timesteps(50)
    b.set_timesteps(50)
    assert not np.allclose(a.timesteps.numpy(), b.timesteps.numpy())
    assert not np.allclose(a.sigmas.numpy(), b.sigmas.numpy())


def test_strict_loader_rejects_missing_and_shape(tmp_path):
    import torch
    from safetensors.torch import save_file
    model = torch.nn.Linear(2, 3)
    path = tmp_path / "weights.safetensors"
    save_file({"weight": model.weight.detach()}, str(path))
    with pytest.raises(ValueError, match="missing"):
        strict_load(model, path)
    save_file({"weight": torch.zeros(4, 2), "bias": model.bias.detach()}, str(path))
    with pytest.raises(ValueError, match="shape"):
        strict_load(model, path)
    save_file(model.state_dict(), str(path))
    strict_load(model, path)


def test_detection_failures_are_penalized():
    rows = []
    for ref, gen, error in [(True, True, 10), (True, False, None), (False, False, None)]:
        rows.append({"reference_detected": ref, "generated_detected": gen, "centroid_px": error,
            "penalized_centroid_px": error if error is not None else (800 if ref else None),
            **{key: 1. for key in ["psnr", "ssim", "lpips", "fg_psnr", "fg_lpips"]}})
    stats = summarize_frames(rows)
    assert stats["centroid_median"] == 10
    assert stats["penalized_centroid_median"] == 405
    assert stats["detection_failure_rate"] == 0.5
    assert stats["reference_detection_rate"] == 2 / 3


def test_checkpoint_selection_action_band():
    from experiment import rank_candidates
    candidates = [dict(name="a", action_score=10, fg_lpips=.2, step=500),
                  dict(name="b", action_score=11.9, fg_lpips=.1, step=1000),
                  dict(name="c", action_score=12.1, fg_lpips=.01, step=1500)]
    assert rank_candidates(candidates)[0]["name"] == "b"
