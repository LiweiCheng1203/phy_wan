#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${SIM2REAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${SIM2REAL_REPO:-$(dirname "$TASK_ROOT")/VideoX-Fun}
commit=13bb3c625e8b1e457d7124d92207b49e00e4a6fe
if [ ! -e "$REPO" ]; then
  git clone --no-checkout https://github.com/aigc-apps/VideoX-Fun.git "$REPO"
  git -C "$REPO" checkout --detach "$commit"
fi
actual=$(git -C "$REPO" rev-parse HEAD)
if [ "$actual" != "$commit" ]; then
  echo "Expected VideoX-Fun $commit; found $actual at $REPO. Set SIM2REAL_REPO to a matching checkout." >&2
  exit 1
fi
echo "VideoX-Fun verified: $actual ($REPO)"
