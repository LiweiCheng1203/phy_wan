"""Strict checkpoint loading and deterministic autoregressive episode reconstruction."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from common import (DATA, MODEL, PROMPT, REPO, ROOT, decode, encode, fingerprint,
                    read_json, reference_video, segment_seed, segments, sha256, write_json)


def checkpoint_state(path):
    import torch
    from safetensors.torch import load_file
    path = Path(path)
    if (path / "transformer").is_dir():
        path = path / "transformer"
    if path.is_dir():
        index = path / "diffusion_pytorch_model.safetensors.index.json"
        if index.exists():
            files = [path / f for f in sorted(set(read_json(index)["weight_map"].values()))]
        else:
            files = [path / "diffusion_pytorch_model.safetensors"]
            if not files[0].exists():
                files = [path / "diffusion_pytorch_model.bin"]
    else:
        files = [path]
    state = {}
    for f in files:
        part = load_file(str(f)) if f.suffix == ".safetensors" else torch.load(f, map_location="cpu", weights_only=True)
        part = part.get("state_dict", part)
        assert not state.keys() & part.keys(), "Duplicate checkpoint keys"
        state.update(part)
    return state, files


def strict_load(transformer, path):
    state, files = checkpoint_state(path)
    expected = transformer.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = {k: [list(state[k].shape), list(expected[k].shape)] for k in expected.keys() & state.keys()
                  if state[k].shape != expected[k].shape}
    if missing or unexpected or mismatched:
        raise ValueError({"checkpoint": str(path), "missing": missing, "unexpected": unexpected, "shape": mismatched})
    transformer.load_state_dict(state, strict=True)
    return {str(f): sha256(f) for f in files}


def scheduler_for(config, shift):
    from diffusers import FlowMatchEulerDiscreteScheduler
    options = dict(config)
    options.update(shift=shift, use_dynamic_shifting=False)
    return FlowMatchEulerDiscreteScheduler(**options)


def load_pipeline(ckpt):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO))
    from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
    from videox_fun.pipeline import WanFunControlPipeline

    cfg = OmegaConf.to_container(OmegaConf.load(REPO / "config/wan2.1/wan_civitai.yaml"))
    tr = cfg["transformer_additional_kwargs"]
    transformer = WanTransformer3DModel.from_pretrained(str(MODEL / tr.get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=tr, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    hashes = strict_load(transformer, MODEL if ckpt == "pretrained" else ckpt)
    vk, tk, ik = cfg["vae_kwargs"], cfg["text_encoder_kwargs"], cfg["image_encoder_kwargs"]
    vae = AutoencoderKLWan.from_pretrained(str(MODEL / vk.get("vae_subpath", "vae")), additional_kwargs=vk).to(torch.bfloat16)
    text = WanT5EncoderModel.from_pretrained(str(MODEL / tk.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=tk, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
    clip = CLIPModel.from_pretrained(str(MODEL / ik.get("image_encoder_subpath", "image_encoder"))).to(torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL / tk.get("tokenizer_subpath", "tokenizer")), local_files_only=True)
    pipe = WanFunControlPipeline(transformer=transformer, vae=vae, tokenizer=tokenizer,
        text_encoder=text, clip_image_encoder=clip, scheduler=scheduler_for(cfg["scheduler_kwargs"], 3))
    pipe.to("cuda")
    for model in [transformer, vae, text, clip]:
        model.requires_grad_(False)
        model.eval()
    return pipe, cfg, hashes


def tensor_video(frames):
    import torch
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="pretrained")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_segment_only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    assert 0 <= args.shard < args.num_shards
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    episodes = read_json(ROOT / "split.json")[args.split][args.shard::args.num_shards][:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"ckpt": args.ckpt, "config": {k: v for k, v in vars(args).items()
        if k not in ["shard", "num_shards", "limit", "output"]}, "split": sha256(ROOT / "split.json"),
        "implementation": sha256(__file__), "common": sha256(Path(__file__).with_name("common.py"))})
    remaining = []
    for episode in episodes:
        meta = args.output / episode / "generation.json"
        if meta.exists() and read_json(meta).get("signature") == signature:
            prior = read_json(meta)
            if all((meta.parent / n).exists() and sha256(meta.parent / n) == h for n, h in prior["output_sha256"].items()):
                print("REUSE", episode, flush=True)
                continue
        remaining.append(episode)
    if not remaining:
        return
    pipe, cfg, weight_hashes = load_pipeline(args.ckpt)
    for episode in remaining:
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        folder = DATA / episode
        out = args.output / episode
        out.mkdir(parents=True, exist_ok=True)
        sim = decode(folder / "sim.mp4")
        first = np.array(Image.open(folder / "first_frame.png").convert("RGB"))
        condition = tensor_video(first[None])
        chunks, details = [], []
        for i, (start, count, padded) in enumerate(segments(len(sim))):
            if args.first_segment_only and i:
                break
            control = sim[start:start + count]
            if padded > count:
                control = np.concatenate([control, np.repeat(control[-1:], padded - count, axis=0)])
            pipe.scheduler = scheduler_for(cfg["scheduler_kwargs"], args.shift)
            seed = segment_seed(args.seed, episode, i)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = pipe(PROMPT, negative_prompt="", height=480, width=640, num_frames=padded,
                    num_inference_steps=args.steps, guidance_scale=args.guidance_scale, shift=args.shift,
                    generator=torch.Generator(device="cuda").manual_seed(seed), control_video=tensor_video(control),
                    start_image=condition, clip_image=Image.fromarray(first), ref_image=None).videos
            assert result.ndim == 5 and result.shape[:2] == (1, 3) and result.shape[2] >= padded, result.shape
            result = result.detach().float().cpu().clamp(0, 1)[:, :, :count]
            assert torch.isfinite(result).all()
            condition = result[:, :, -1:].clone()
            # The next condition stays in memory; MP4 is never used as a recurrent input.
            first = condition[0, :, 0].permute(1, 2, 0).mul(255).round().byte().numpy()
            frames = result[0].permute(1, 2, 3, 0).numpy().copy()
            chunks.append(frames if i == 0 else frames[1:])
            details.append({"index": i, "start": start, "R": count, "L": padded, "seed": seed,
                "timesteps": pipe.scheduler.timesteps.detach().cpu().float().tolist(),
                "sigmas": pipe.scheduler.sigmas.detach().cpu().float().tolist()})
        generated = np.concatenate(chunks)
        expected = min(81, len(sim)) if args.first_segment_only else len(sim)
        assert generated.shape == (expected, 480, 640, 3), generated.shape
        # Float32 .npy preserves the pipeline output for metrics, before video quantization.
        with open(out / "frames.partial.npy", "wb") as f:
            np.save(f, generated)
        (out / "frames.partial.npy").replace(out / "frames.npy")
        export = (generated * 255).round().astype(np.uint8)
        encode(out / "generated.mp4", export)
        real = decode(reference_video(episode, read_json(folder / "meta.json")))[:expected]
        encode(out / "triptych.mp4", np.concatenate([real, sim[:expected], export], axis=2))
        write_json(out / "generation.json", {"signature": signature, "episode": episode, "N": expected,
            "ckpt": args.ckpt, "checkpoint_sha256": weight_hashes, "guidance_scale": args.guidance_scale,
            "shift": args.shift, "steps": args.steps, "seed": args.seed, "segments": details,
            "seconds": time.monotonic() - begin, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "output_sha256": {name: sha256(out / name) for name in ["frames.npy", "generated.mp4", "triptych.mp4"]}})
        print("GENERATED", episode, expected, round(time.monotonic() - begin, 1), flush=True)


if __name__ == "__main__":
    main()
