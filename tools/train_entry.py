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
from infer import strict_load
from training_precision import prepare_full_precision_parameters
from training_runtime import (compile_training_program, load_training_generators,
                              save_rank_random_state, save_training_generators)

sys.path.insert(0, str(REPO))
from videox_fun.models import WanTransformer3DModel
import videox_fun.data.dataset_image_video as dataset_module

audit_dir = Path(os.environ.get("SIM2REAL_AUDIT", str(ROOT / "reports/train_audit")))
audit_dir.mkdir(parents=True, exist_ok=True)
rank = int(os.environ.get("RANK", 0))
log_path = audit_dir / f"rank{rank}.jsonl"
started = time.monotonic()
runtime_parser = argparse.ArgumentParser(add_help=False)
runtime_parser.add_argument("--output_dir", required=True)
runtime_parser.add_argument("--max_train_steps", required=True, type=int)
runtime_parser.add_argument("--checkpointing_steps", type=int, default=250)
runtime_args, _ = runtime_parser.parse_known_args()


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
    assert len(indices) == 81 and (indices[1:] - indices[:-1] == 1).all(), indices
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
            assert shape[1:] == [81, 3, 480, 640], shape
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
original_clip = Accelerator.clip_grad_norm_


def checked_clip(acc, params, *args, **kwargs):
    norm = original_clip(acc, params, *args, **kwargs)
    if norm is not None:
        assert torch.isfinite(norm).all(), f"Nonfinite gradient norm: {norm}"
        record("gradient", norm=float(norm), clipping_threshold=float(args[0] if args else kwargs["max_norm"]))
    return norm


Accelerator.clip_grad_norm_ = checked_clip
original_log = Accelerator.log


def checked_log(acc, values, step=None, **kwargs):
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
    load_training_generators(input_dir, acc.process_index, sys._getframe(1).f_locals)
    record("training_generators_restored", path=str(input_dir), generators=["rng", "torch_rng", "index_rng"])
    return result


Accelerator.load_state = checked_restore
record("start", argv=sys.argv[1:], torch=torch.__version__, trainer_sha256=sha256(REPO / "scripts/wan2.1_fun/train_control.py"),
       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
trainer_path = REPO / "scripts/wan2.1_fun/train_control.py"
record("runtime_corrections", epoch_stop_guard=True, per_rank_random_states=True)
exec(compile_training_program(trainer_path), {"__name__": "__main__", "__file__": str(trainer_path)})
changes = {name: float((p.detach().flatten()[:1024].float().cpu() - initial).abs().max()) for name, p, initial in monitored}
record("completed", parameter_changes=changes, peak_cuda_bytes=torch.cuda.max_memory_allocated())
assert any(v > 0 for v in changes.values()), "No monitored Transformer parameters changed"
