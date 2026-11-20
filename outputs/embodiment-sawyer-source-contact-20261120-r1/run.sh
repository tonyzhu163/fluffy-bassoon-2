#!/usr/bin/env bash
# One Baige Distributed Training worker. Arguments are the diagnostic pin and fresh attempt.
set -euo pipefail
pin=${1:?pin}
attempt=${2:?attempt}
[[ $(uname -s) == Linux && $(hostname) == job-*-master-0 && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^native[0-9]+$ ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
if git -C "$code" symbolic-ref -q HEAD; then exit 1; fi
git -C "$code" diff --quiet HEAD --
r=/mowenjie/fluffy-bassoon-2
n=embodiment-sawyer-source-contact-20261120-r1
run="$r/outputs/$n/$attempt"
mkdir "$run"
exec > >(tee "$run/run.log") 2>&1
trap 'rc=$?; echo "$rc $(date -u +%FT%TZ)" > "$run/exit.txt"' EXIT
printf '%s %s %s\n' "$pin" "$(hostname)" "$(date -u +%FT%TZ)" > "$run/launch.txt"
setup="$r/setups/libero-meta-learning-20261117-r1"
orig="$r/setups/libero-paligemma-dualarm-20261120-r2/source-fc61bf2a"
export CROSSARM_RELEASE="$r/sources/libero-pro2-e5500c69" CROSSARM_SIM_PYTHON="$setup/env/bin/python"
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export PYTHONPATH="$orig/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
mkdir "$run/tmp" "$run/cache"
export TMPDIR="$run/tmp" XDG_CACHE_HOME="$run/cache" NUMBA_CACHE_DIR="$run/cache/numba"
cd "$run"
{
  hostname
  date -u +%FT%TZ
  df -hT .
  df -i .
  if command -v quota >/dev/null; then quota -s || true; else echo "quota command unavailable"; fi
  cat /sys/fs/cgroup/{cpu.max,memory.max}
  "$CROSSARM_SIM_PYTHON" -Bu - <<'PY'
import glob, os, sys
from pathlib import Path
quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
cards = glob.glob('/dev/nvidia[0-9]*')
print('Interpreter', sys.executable, sys.version, 'cards', cards, 'quota cores', int(quota)/int(period))
assert not cards and 0 < int(quota) / int(period) <= 8
assert os.environ['CROSSARM_SIM_PYTHON'] == sys.executable
PY
} | tee allocation.txt
(cd "$here"; sha256sum run.sh replay.py geometry.py summarize.py) > source.sha256
mesa="$r/setups/embodiment-command-frame-20261120-r1/mesa-runtime"
(cd "$mesa"; echo '78686e3a44f2daf8a67b5150672ba828afbd4b33f6d4df87d6b4e11a1c10f4af  SHA256SUMS' | sha256sum -c -; sha256sum --quiet -c SHA256SUMS)
export LD_LIBRARY_PATH="$mesa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBGL_DRIVERS_PATH="$mesa/dri" __EGL_VENDOR_LIBRARY_FILENAMES="$mesa/50_mesa.json"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1
export MOTOR_CODE_REVISION=fc61bf2afb48d4639664db93dae64ae9cb04e2df
rc=0
"$CROSSARM_SIM_PYTHON" -Bu "$here/replay.py" --checkout "$orig" \
  --panel "$r/outputs/libero-paligemma-dualarm-20261120-r2/large-panel-v3" --out "$run/replay" --workers 6 --traces || rc=$?
"$CROSSARM_SIM_PYTHON" -Bu "$here/summarize.py" "$run/replay"
exit "$rc"
