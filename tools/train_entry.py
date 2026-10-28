"""Execute the unchanged upstream trainer with runtime contract/provenance checks."""
import functools
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch
from accelerate import Accelerator

from common import REPO, ROOT, sha256
from infer import strict_load

sys.path.insert(0, str(REPO))
from videox_fun.models import WanTransformer3DModel
import videox_fun.data.dataset_image_video as dataset_module

audit_dir = Path(os.environ.get("SIM2REAL_AUDIT", str(ROOT / "reports/train_audit")))
audit_dir.mkdir(parents=True, exist_ok=True)
rank = int(os.environ.get("RANK", 0))
log_path = audit_dir / f"rank{rank}.jsonl"
started = time.monotonic()


def record(event, **values):
    with open(log_path, "a") as f:
        f.write(json.dumps({"event": event, "rank": rank, "seconds": time.monotonic() - started, **values}, allow_nan=False) + "\n")


original_load = WanTransformer3DModel.from_pretrained.__func__


@classmethod
def checked_load(cls, path, *args, **kwargs):
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
    return norm


Accelerator.clip_grad_norm_ = checked_clip
original_log = Accelerator.log


def checked_log(acc, values, step=None, **kwargs):
    for name, value in values.items():
        if isinstance(value, (int, float)):
            assert torch.isfinite(torch.tensor(value)), (name, value)
    if acc.is_main_process:
        record("training", step=step, values=values, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    return original_log(acc, values, step=step, **kwargs)


Accelerator.log = checked_log
original_save = Accelerator.save_state


def timed_save(acc, output_dir, *args, **kwargs):
    begin = time.monotonic()
    result = original_save(acc, output_dir, *args, **kwargs)
    record("checkpoint_saved", path=str(output_dir), save_seconds=time.monotonic() - begin,
           bytes=sum(p.stat().st_size for p in Path(output_dir).rglob("*") if p.is_file()))
    return result


Accelerator.save_state = timed_save
record("start", argv=sys.argv[1:], torch=torch.__version__, trainer_sha256=sha256(REPO / "scripts/wan2.1_fun/train_control.py"),
       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
runpy.run_path(str(REPO / "scripts/wan2.1_fun/train_control.py"), run_name="__main__")
changes = {name: float((p.detach().flatten()[:1024].float().cpu() - initial).abs().max()) for name, p, initial in monitored}
record("completed", parameter_changes=changes, peak_cuda_bytes=torch.cuda.max_memory_allocated())
assert any(v > 0 for v in changes.values()), "No monitored Transformer parameters changed"
