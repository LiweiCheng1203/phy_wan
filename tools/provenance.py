"""Capture immutable environment, pretrained assets, and source provenance."""
from pathlib import Path
import platform
import subprocess

import torch

from common import MODEL, REPO, ROOT, read_json, sha256, write_json

files = {str(p.relative_to(MODEL)): {"sha256": sha256(p), "bytes": p.stat().st_size}
         for p in sorted(MODEL.rglob("*")) if p.is_file()}
record = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
    "git_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
    "git_status": subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True),
    "model_files": files, "config": (REPO / "config/wan2.1/wan_civitai.yaml").read_text(),
    "gpu_info": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True),
    "inventory": read_json(ROOT / "inventory.json"),
    "task_code_sha256": {str(p.relative_to(ROOT)): sha256(p) for sub in ["tools", "scripts", "configs"]
                         for p in sorted((ROOT / sub).rglob("*")) if p.is_file() and "__pycache__" not in str(p)}}
write_json(ROOT / "reports/provenance.json", record)
print("PROVENANCE", record["git_commit"], len(files), flush=True)
