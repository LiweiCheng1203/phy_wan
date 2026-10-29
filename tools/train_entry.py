"""Execute the unchanged upstream trainer with runtime contract/provenance checks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from accelerate import Accelerator

from common import REPO, ROOT, sha256
from condition_noise import apply_condition_noise
from infer import strict_load
from training_precision import prepare_full_precision_parameters
from training_runtime import (GENERATORS, compile_training_program, load_training_generators,
                              save_rank_random_state, save_training_generators)

sys.path.insert(0, str(REPO))
from videox_fun.models import WanTransformer3DModel
import videox_fun.data.dataset_image_video as dataset_module

audit_dir = Path(os.environ.get("SIM2REAL_AUDIT", str(ROOT / "reports/train_audit")))
audit_dir.mkdir(parents=True, exist_ok=True)
rank = int(os.environ.get("RANK", 0))
log_path = audit_dir / f"rank{rank}.jsonl"
started = time.monotonic()
launch_argv = sys.argv[1:]
# Options of this wrapper only; removed before the upstream parser sees the command line.
private_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
private_parser.add_argument("--cond_noise_prob", type=float, default=0.)
private_parser.add_argument("--cond_noise_sigma_max", type=float, default=0.)
private_args, _ = private_parser.parse_known_args()


def strip_options(argv, names):
    kept, i = [], 0
    while i < len(argv):
        name, inline = argv[i].partition("=")[::2]
        if name in names:
            i += 1 if inline else 2
        else:
            kept.append(argv[i])
            i += 1
    return kept


sys.argv = sys.argv[:1] + strip_options(launch_argv, {"--cond_noise_prob", "--cond_noise_sigma_max"})
condition_noise = private_args.cond_noise_prob > 0
assert 0 <= private_args.cond_noise_prob <= 1 and private_args.cond_noise_sigma_max >= 0
assert condition_noise == (private_args.cond_noise_sigma_max > 0), "Set both condition noise options or neither"
runtime_parser = argparse.ArgumentParser(add_help=False)
runtime_parser.add_argument("--output_dir", required=True)
runtime_parser.add_argument("--max_train_steps", required=True, type=int)
runtime_parser.add_argument("--checkpointing_steps", type=int, default=250)
runtime_parser.add_argument("--video_sample_n_frames", required=True, type=int)
runtime_parser.add_argument("--fix_sample_size", required=True, type=int, nargs=2)
runtime_args, _ = runtime_parser.parse_known_args()
frames = runtime_args.video_sample_n_frames
assert frames % 4 == 1, f"Wan VAE needs 4k+1 frames, got {frames}"
height, width = runtime_args.fix_sample_size


def record(event, **values):
    with open(log_path, "a") as f:
        f.write(json.dumps({"event": event, "rank": rank, "seconds": time.monotonic() - started, **values}, allow_nan=False) + "\n")


original_load = WanTransformer3DModel.from_pretrained.__func__


@classmethod
def checked_load(cls, path, *args, **kwargs):
    # Resume checkpoints contain FP32 training weights; avoid rounding them on load.
    kwargs["torch_dtype"] = torch.float32
    model = original_load(cls, path, *args, **kwargs)
    directory = Path(path) / kwargs.get("subfolder", "")
    hashes = strict_load(model, directory)
    record("strict_transformer_load", path=str(directory), sha256=hashes,
           parameters=sum(p.numel() for p in model.parameters()))
    return model


WanTransformer3DModel.from_pretrained = checked_load
original_batch = dataset_module.get_video_reader_batch
sample_index = None
sample_calls = 0


def checked_batch(reader, indices):
    global sample_index, sample_calls
    sample_calls += 1
    assert len(indices) == frames and (indices[1:] - indices[:-1] == 1).all(), indices
    if sample_calls % 2:
        sample_index = indices.copy()
    else:
        assert (sample_index == indices).all(), "Control/target sample indices differ"
    if sample_calls <= 2:
        record("sample_indices", worker_pid=os.getpid(), indices=indices.tolist(), stream=sample_calls)
    return original_batch(reader, indices)


dataset_module.get_video_reader_batch = checked_batch
original_iter = torch.utils.data.DataLoader.__iter__


def checked_iter(loader):
    for index, batch in enumerate(original_iter(loader)):
        if index == 0 and isinstance(batch, dict) and "ref_pixel_values" in batch:
            shape = list(batch["pixel_values"].shape)
            assert shape[1:] == [frames, 3, height, width], shape
            assert batch["control_pixel_values"].shape == batch["pixel_values"].shape
            assert torch.equal(batch["ref_pixel_values"], batch["pixel_values"][:, :1])
            assert torch.all(batch["clip_idx"] == 0)
            record("batch_contract", shape=shape, control_shape=list(batch["control_pixel_values"].shape),
                   ref_is_window_first_frame=True, clip_indices=batch["clip_idx"].tolist())
        yield batch


torch.utils.data.DataLoader.__iter__ = checked_iter
original_prepare = Accelerator.prepare
monitored = []


def checked_prepare(acc, *args, **kwargs):
    models = [x for x in args if isinstance(x, WanTransformer3DModel)]
    optimizers = [x for x in args if isinstance(x, torch.optim.Optimizer)]
    if models and optimizers:
        model, optimizer = models[0], optimizers[0]
        precision = prepare_full_precision_parameters(model, optimizer)
        record("training_precision", **precision)
        train_ids = {id(p) for p in model.parameters() if p.requires_grad}
        all_ids = {id(p) for p in model.parameters()}
        optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert train_ids == all_ids == set(optimized) and len(optimized) == len(set(optimized))
        record("optimizer_contract", trainable_parameters=sum(p.numel() for p in model.parameters()),
               frozen_modules_excluded=True, full_transformer=True, parameter_dtypes=sorted(set(str(p.dtype) for p in model.parameters())))
        for name, p in model.named_parameters():
            if p.requires_grad and ("bias" in name or "head" in name):
                monitored.append((name, p, p.detach().flatten()[:1024].float().cpu().clone()))
        def check_forward(module, args, kwargs):
            full_ref = kwargs.get("full_ref")
            if full_ref is not None:
                assert torch.count_nonzero(full_ref) == 0, "Full reference branch input must be zero"
        model.register_forward_pre_hook(check_forward, with_kwargs=True)
    result = original_prepare(acc, *args, **kwargs)
    return result


Accelerator.prepare = checked_prepare
noise_stats = {"calls": 0, "sigmas": []}


def checked_condition_noise(latent, generator):
    latent, sigma = apply_condition_noise(latent, generator, private_args.cond_noise_prob,
                                          private_args.cond_noise_sigma_max)
    sigma = sigma.float().cpu().tolist()
    noise_stats["calls"] += 1
    noise_stats["sigmas"] += sigma
    if noise_stats["calls"] <= 20:
        record("condition_noise", call=noise_stats["calls"], sigma=sigma, latent_shape=list(latent.shape))
    return latent


original_clip = Accelerator.clip_grad_norm_
last_grad_norm = None


def checked_clip(acc, params, *args, **kwargs):
    global last_grad_norm
    norm = original_clip(acc, params, *args, **kwargs)
    if norm is not None:
        assert torch.isfinite(norm).all(), f"Nonfinite gradient norm: {norm}"
        last_grad_norm = float(norm)
        record("gradient", norm=last_grad_norm, clipping_threshold=float(args[0] if args else kwargs["max_norm"]))
    return norm


Accelerator.clip_grad_norm_ = checked_clip
original_init_trackers = Accelerator.init_trackers


def checked_init_trackers(acc, project_name, config=None, init_kwargs=None):
    # The private options never reach upstream args, so add them to the tracker config here.
    config = {**(config or {}), "window_frames": frames, "frame_size": [height, width], "cond_noise_prob": private_args.cond_noise_prob,
              "cond_noise_sigma_max": private_args.cond_noise_sigma_max, "launch_argv": " ".join(launch_argv)}
    return original_init_trackers(acc, project_name, config=config, init_kwargs=init_kwargs or {})


Accelerator.init_trackers = checked_init_trackers
original_log = Accelerator.log
last_log = None


def training_metrics(step):
    global last_log
    metrics = {"gpu/peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    now = time.monotonic()
    if last_log is not None and step > last_log[0]:
        metrics["time/sec_per_step"] = (now - last_log[1]) / (step - last_log[0])
    last_log = (step, now)
    if last_grad_norm is not None:
        metrics["grad_norm"] = last_grad_norm
    if noise_stats["sigmas"]:
        sigmas = noise_stats["sigmas"]
        metrics["cond_noise/sigma_mean"] = sum(sigmas) / len(sigmas)
        metrics["cond_noise/clean_frac"] = sum(s == 0 for s in sigmas) / len(sigmas)
        noise_stats["sigmas"] = []
    return metrics


def checked_log(acc, values, step=None, **kwargs):
    if "train_loss" in values and step is not None:
        values = {**values, **training_metrics(step)}
    for name, value in values.items():
        if isinstance(value, (int, float)):
            assert torch.isfinite(torch.tensor(value)), (name, value)
    if acc.is_main_process:
        record("training", step=step, values=values, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if "train_loss" in values and step is not None and (
            step % runtime_args.checkpointing_steps == 0 or step >= runtime_args.max_train_steps):
        path = Path(runtime_args.output_dir) / ".rank_random_states" / f"checkpoint-{step}"
        save_rank_random_state(acc, path)
        save_training_generators(path, acc.process_index, sys._getframe(1).f_locals)
        acc.wait_for_everyone()
        record("rank_random_state_saved", step=step, accumulation_step=acc.step)
    return original_log(acc, values, step=step, **kwargs)


Accelerator.log = checked_log
original_save = Accelerator.save_state


def timed_save(acc, output_dir, *args, **kwargs):
    begin = time.monotonic()
    result = original_save(acc, output_dir, *args, **kwargs)
    staged = Path(output_dir).parent / ".rank_random_states" / Path(output_dir).name
    if acc.is_main_process:
        for state in staged.glob("*random_states_*.pkl"):
            shutil.copy2(state, Path(output_dir) / state.name)
    record("checkpoint_saved", path=str(output_dir), save_seconds=time.monotonic() - begin,
           bytes=sum(p.stat().st_size for p in Path(output_dir).rglob("*") if p.is_file()))
    return result


Accelerator.save_state = timed_save
original_restore = Accelerator.load_state


def checked_restore(acc, input_dir, *args, **kwargs):
    result = original_restore(acc, input_dir, *args, **kwargs)
    scope = sys._getframe(1).f_locals
    load_training_generators(input_dir, acc.process_index, scope)
    record("training_generators_restored", path=str(input_dir),
           generators=[name for name in GENERATORS if scope.get(name) is not None])
    return result


Accelerator.load_state = checked_restore
record("start", argv=launch_argv, torch=torch.__version__, trainer_sha256=sha256(REPO / "scripts/wan2.1_fun/train_control.py"),
       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
trainer_path = REPO / "scripts/wan2.1_fun/train_control.py"
record("runtime_corrections", epoch_stop_guard=True, per_rank_random_states=True, condition_noise=condition_noise)
record("condition_noise_config", enabled=condition_noise, prob=private_args.cond_noise_prob,
       sigma_max=private_args.cond_noise_sigma_max, target="ref_latents_conv_in[:, :, :1]", clip_image="clean",
       window_frames=frames, frame_size=[height, width])
namespace = {"__name__": "__main__", "__file__": str(trainer_path)}
if condition_noise:
    namespace["_sim2real_condition_noise"] = checked_condition_noise
exec(compile_training_program(trainer_path, condition_noise=condition_noise), namespace)
changes = {name: float((p.detach().flatten()[:1024].float().cpu() - initial).abs().max()) for name, p, initial in monitored}
record("completed", parameter_changes=changes, peak_cuda_bytes=torch.cuda.max_memory_allocated())
assert any(v > 0 for v in changes.values()), "No monitored Transformer parameters changed"
