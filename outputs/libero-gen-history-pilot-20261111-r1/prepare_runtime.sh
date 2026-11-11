#!/usr/bin/env bash
# CPU-only installation into this pilot's private environment; no model execution.
set -euo pipefail
TASK_PROJECT=${1:?project root}
TASK_CODE=${2:?pinned project checkout}
TASK_SETUP="$TASK_PROJECT/setups/libero-gen-history-pilot-20261111-r1"
TASK_RUN="$TASK_PROJECT/outputs/libero-gen-history-pilot-20261111-r1"
TASK_FILES="$TASK_CODE/outputs/libero-gen-history-pilot-20261111-r1"
TASK_BPP="$TASK_PROJECT/cache/bpp-58125e6-20261111-r1"
TASK_ENV="$TASK_SETUP/env"
mkdir -p "$TASK_SETUP" "$TASK_RUN"
export UV_CACHE_DIR="$TASK_SETUP/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_SETUP/python"
export HF_HOME="$TASK_SETUP/huggingface-cache"
export HF_HUB_DISABLE_XET=1
python - "$TASK_SETUP" <<'PY'
import json, pathlib, shutil, sys
p=pathlib.Path(sys.argv[1]); d=shutil.disk_usage(p)
assert d.free > 30 * 2**30, 'Need 30 GiB free for isolated preparation'
(p/'storage-before.json').write_text(json.dumps({'free_bytes':d.free,'quota_command':shutil.which('quota')}))
(p/'write-probe').write_text('ok'); assert (p/'write-probe').read_text()=='ok'
PY
if [[ ! -x "$TASK_SETUP/uv-0.13.7/bin/uv" ]]; then
  python -m pip install --target "$TASK_SETUP/uv-0.13.7" 'uv==0.13.7'
fi
TASK_UV="$TASK_SETUP/uv-0.13.7/bin/uv"
"$TASK_UV" --version > "$TASK_SETUP/uv-version.txt"
test "$(git -C "$TASK_BPP" rev-parse HEAD)" = 58125e682acceefef3d60c071212dc483a4db1c9
timeout 900 git -C "$TASK_BPP" submodule update --init --depth 1 deps/LIBERO
test "$(git -C "$TASK_BPP/deps/LIBERO" rev-parse HEAD)" = 8f1084e3132a39270c3a13ebe37270a43ece2a01
if [[ ! -x "$TASK_ENV/bin/python" ]]; then
  "$TASK_UV" venv --python 3.10.20 "$TASK_ENV"
fi
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" 'pip==24.0' 'setuptools==65.5.0' 'wheel==0.38.4'
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" -r "$TASK_FILES/runtime-requirements.txt"
# gym 0.21's old metadata needs the legacy pip/wheel toolchain used by BPP.
"$TASK_ENV/bin/python" -m pip install --no-build-isolation --no-deps 'gym==0.21.0'
"$TASK_ENV/bin/python" -m pip install --no-deps 'git+https://github.com/austinapatel/robomimic.git@f4357ed548b656800636ee459c7a2f31f95b5a08'
"$TASK_ENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$TASK_BPP/deps/LIBERO" -e "$TASK_BPP"
"$TASK_ENV/bin/python" -m pip freeze > "$TASK_SETUP/installed.txt"
"$TASK_ENV/bin/python" "$TASK_FILES/fetch_assets.py" --manifest "$TASK_FILES/ASSET-MANIFEST.json" --destination "$TASK_SETUP/assets" --receipt "$TASK_RUN/ASSET-DOWNLOAD.json"
printf 'CPU_RUNTIME_AND_ASSETS_PREPARED\n'
