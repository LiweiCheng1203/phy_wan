"""Resumable HTTP range download with mandatory whole-file SHA-256 verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.request

from common import sha256

p = argparse.ArgumentParser()
p.add_argument("url")
p.add_argument("output", type=Path)
p.add_argument("sha256")
p.add_argument("--workers", type=int, default=16)
args = p.parse_args()
if args.output.exists() and sha256(args.output) == args.sha256:
    raise SystemExit(0)
parts = args.output.with_name(args.output.name + ".parts")
parts.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(urllib.request.Request(args.url, headers={"Range": "bytes=0-0"}), timeout=40) as response:
    assert response.status == 206, response.status
    total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
chunk = 8 * 1024 * 1024


def fetch(index):
    start, end = index * chunk, min(total, (index + 1) * chunk) - 1
    path = parts / str(index)
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    error = None
    for retry in range(5):
        try:
            request = urllib.request.Request(args.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == f"bytes {start}-{end}/{total}"
                with open(path.with_suffix(".tmp"), "wb") as f:
                    while data := response.read(1024 * 1024):
                        f.write(data)
            assert path.with_suffix(".tmp").stat().st_size == end - start + 1
            path.with_suffix(".tmp").replace(path)
            print("PART", index, end + 1, total, flush=True)
            return path
        except Exception as exc:
            error = exc
            print("RETRY", index, retry, repr(exc), flush=True)
            time.sleep(2)
    raise RuntimeError(f"Part {index}: {error}")


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    paths = list(pool.map(fetch, range((total + chunk - 1) // chunk)))
temp = args.output.with_name(args.output.name + ".tmp")
with open(temp, "wb") as f:
    for path in paths:
        with open(path, "rb") as source:
            while data := source.read(4 * 1024 * 1024):
                f.write(data)
assert temp.stat().st_size == total and sha256(temp) == args.sha256, "Asset SHA-256 mismatch"
temp.replace(args.output)
print("VERIFIED", args.output, args.sha256, flush=True)
