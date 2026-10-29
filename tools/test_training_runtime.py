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


NOISE_TRAINER = '''
from types import SimpleNamespace
import numpy as np
import torch
def main():
    args = SimpleNamespace(seed=1, max_train_steps=2)
    accelerator = SimpleNamespace(device="cpu", process_index=0)
    index_rng = np.random.default_rng(np.random.PCG64(43))
    global_step = 0
    seen = []
    for epoch in range(2):
        for batch in range(3):
            latents = torch.zeros(2, 4, 3, 2, 2)
            ref_latents = torch.ones(2, 4, 1, 2, 2)
            ref_latents_conv_in = torch.zeros_like(latents)
            ref_latents_conv_in[:, :, :1] = ref_latents
            seen.append(ref_latents_conv_in.clone())
            global_step += 1
            if global_step >= args.max_train_steps:
                break
    return seen, cond_noise_rng
'''


def test_condition_noise_patch_applies_once_per_batch(tmp_path):
    source = tmp_path / "trainer.py"
    source.write_text(NOISE_TRAINER)
    original = source.read_bytes()
    calls = []
    namespace = {"_sim2real_condition_noise": lambda latent, generator: calls.append(generator) or latent + 1}
    exec(compile_training_program(source, condition_noise=True), namespace)
    seen, generator = namespace["main"]()
    assert len(seen) == len(calls) == 2
    assert all(g is generator for g in calls) and isinstance(generator, torch.Generator)
    assert all(torch.all(x[:, :, :1] == 2) and torch.all(x[:, :, 1:] == 0) for x in seen)
    assert source.read_bytes() == original
    source.write_text(NOISE_TRAINER.replace("return seen, cond_noise_rng", "return seen, None"))
    namespace = {}
    exec(compile_training_program(source), namespace)
    assert torch.all(namespace["main"]()[0][0][:, :, :1] == 1)


def test_condition_noise_patch_requires_unique_target(tmp_path):
    import pytest
    source = tmp_path / "trainer.py"
    for text in [NOISE_TRAINER.replace("ref_latents_conv_in[:, :, :1] = ref_latents\n", "pass\n"),
                 NOISE_TRAINER.replace("ref_latents_conv_in[:, :, :1] = ref_latents\n",
                                       "ref_latents_conv_in[:, :, :1] = ref_latents\n"
                                       "            ref_latents_conv_in[:, :, :1] = ref_latents\n"),
                 NOISE_TRAINER.replace("    index_rng = np.random.default_rng(np.random.PCG64(43))\n", "")]:
        source.write_text(text)
        with pytest.raises(RuntimeError, match="condition noise patch"):
            compile_training_program(source, condition_noise=True)


def test_pinned_upstream_accepts_condition_noise():
    import pytest
    from common import REPO
    trainer = REPO / "scripts/wan2.1_fun/train_control.py"
    if not trainer.exists():
        pytest.skip("Pinned upstream trainer not available")
    compile_training_program(trainer, condition_noise=True)


def test_condition_noise_statistics():
    from condition_noise import apply_condition_noise
    latent = torch.randn(4000, 2, 1, 3, 3).to(torch.bfloat16)
    same, sigma = apply_condition_noise(latent, torch.Generator().manual_seed(0), 0., .3)
    assert torch.equal(same, latent) and torch.all(sigma == 0)
    a, sigma = apply_condition_noise(latent, torch.Generator().manual_seed(0), .75, .3)
    b, _ = apply_condition_noise(latent, torch.Generator().manual_seed(0), .75, .3)
    assert torch.equal(a, b) and a.dtype == torch.bfloat16 and a.shape == latent.shape
    assert sigma.min() >= 0 and sigma.max() <= .3
    assert abs((sigma == 0).float().mean().item() - .25) < .03
    noised = sigma > .05
    residual = (a.float() - latent.float())[noised] / sigma[noised].view(-1, 1, 1, 1, 1)
    assert abs(residual.std().item() - 1) < .05
    assert torch.equal(a[sigma == 0], latent[sigma == 0])


def test_condition_noise_generator_continues_after_restore(tmp_path):
    scope = dict(cond_noise_rng=torch.Generator().manual_seed(45))
    torch.rand(3, generator=scope["cond_noise_rng"])
    save_training_generators(tmp_path, 1, scope)
    expected = torch.rand(10, generator=scope["cond_noise_rng"])
    load_training_generators(tmp_path, 1, scope)
    assert torch.equal(torch.rand(10, generator=scope["cond_noise_rng"]), expected)
    with __import__("pytest").raises(KeyError):
        load_training_generators(tmp_path, 1, {})
