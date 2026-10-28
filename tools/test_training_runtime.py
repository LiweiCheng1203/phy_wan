"""Regression checks for resumed training termination and per-rank RNG files."""
from types import SimpleNamespace

import torch

from training_runtime import (compile_training_program, load_training_generators,
                              save_rank_random_state, save_training_generators)


def test_resumed_loop_stops_at_requested_update(tmp_path):
    source = tmp_path / "trainer.py"
    source.write_text('''
from types import SimpleNamespace
def main():
    args = SimpleNamespace(max_train_steps=20)
    global_step = 10
    for epoch in range(8):
        for batch in range(6):
            global_step += 1
            if global_step >= args.max_train_steps:
                break
    return global_step
''')
    original = source.read_bytes()
    namespace = {}
    exec(compile_training_program(source), namespace)
    assert namespace["main"]() == 20
    assert source.read_bytes() == original


def test_random_state_saved_for_each_rank(tmp_path, monkeypatch):
    from accelerate import Accelerator
    monkeypatch.setattr("accelerate.checkpointing.is_cuda_available", lambda: False)
    Accelerator(cpu=True)
    for rank in range(4):
        save_rank_random_state(SimpleNamespace(process_index=rank, step=12), tmp_path)
    for rank in range(4):
        state = torch.load(tmp_path / f"random_states_{rank}.pkl", weights_only=False)
        assert state["step"] == 12
        assert {"random_state", "numpy_random_seed", "torch_manual_seed"} <= state.keys()


def test_custom_training_generators_continue_after_restore(tmp_path):
    import numpy as np
    scope = dict(rng=np.random.default_rng(42), index_rng=np.random.default_rng(43),
                 torch_rng=torch.Generator().manual_seed(44))
    scope["rng"].random(5)
    torch.rand(5, generator=scope["torch_rng"])
    save_training_generators(tmp_path, 2, scope)
    expected_np = scope["rng"].random(10)
    expected_index = scope["index_rng"].integers(1000, size=10)
    expected_torch = torch.rand(10, generator=scope["torch_rng"])
    load_training_generators(tmp_path, 2, scope)
    assert np.array_equal(scope["rng"].random(10), expected_np)
    assert np.array_equal(scope["index_rng"].integers(1000, size=10), expected_index)
    assert torch.equal(torch.rand(10, generator=scope["torch_rng"]), expected_torch)


def test_missing_training_generators_rejected(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="custom training generator"):
        load_training_generators(tmp_path, 0, {})
