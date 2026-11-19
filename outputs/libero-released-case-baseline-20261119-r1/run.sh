#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
pro="$root/sources/libero-pro2-e5500c69"
sim="$root/setups/libero-meta-learning-20261117-r1"
setup="$root/setups/$(basename "$here")"
py="$sim/env/bin/python"
pin=$1
attempt=$2
[[ "$attempt" =~ ^[a-z0-9]+$ ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain) ]]
[[ $(git -C "$pro" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
[[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
[[ $CGPU_COUNT == 1 && $NVIDIA_VISIBLE_DEVICES == "$NVIDIA_VISIBLE_GPUS_UUID" && $NVIDIA_VISIBLE_DEVICES != *,* ]]
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES" JAX_PLATFORMS=cuda
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
export OPENPI_WORKER_PYTHONPATH="$pro:$code/src"
export LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
out="$root/outputs/$(basename "$here")/$attempt"
mkdir -p "$(dirname "$out")"
mkdir "$out"
exec 9>"$root/outputs/libero-bowl-decisions-20261118-r1/device.lock"
flock -w 3600 9
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
df -h "$out" > "$out/storage.txt"
printf probe > "$out/write-probe"
rm "$out/write-probe"
start=$(date +%s)
server_pid=
cleanup() {
    local rc=$?
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" || true
    fi
    echo "$rc $(date +%s) $start" > "$out/exit.txt"
}
trap cleanup EXIT
echo "$pin $(hostname) $$" > "$out/launch.txt"
PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
    --expected-count 1 --purpose training --direct-hostname "$(hostname)" --output "$out/gpu.json" > "$out/probe.log" 2>&1
export PYTHONPATH="$setup/server-packages:$pro:$sim/simulator-source:$code/src"
export LIBERO_REMOTE_HOST=127.0.0.1
LIBERO_REMOTE_PORT=$("$py" -B -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
export LIBERO_REMOTE_PORT
base_url="http://127.0.0.1:$LIBERO_REMOTE_PORT"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export LIBERO_REMOTE_SCHEDULER_DB_PATH="$out/scheduler.sqlite3"
export LIBERO_REMOTE_TASK_MANIFEST="$pro/assets/task_manifest.json"
export LIBERO_REMOTE_WORKER_TIMEOUT_SECONDS=180 LIBERO_REMOTE_LOG_LEVEL=warning
export LIBERO_REMOTE_AUTHENTICATION_REQUIRED=true
LIBERO_REMOTE_API_KEY=$("$py" -B -c 'import secrets; print(secrets.token_urlsafe(32))')
export LIBERO_REMOTE_API_KEY
LIBERO_REMOTE_API_KEY_HASHES=$("$py" -B -c 'import hashlib,json,os; print(json.dumps({hashlib.sha256(os.environ["LIBERO_REMOTE_API_KEY"].encode()).hexdigest():"dev-eval"}))')
export LIBERO_REMOTE_API_KEY_HASHES
cd "$out"
"$py" -Bu -m libero_remote.server.main > "$out/server.log" 2>&1 &
server_pid=$!
for ((check=0; check<60; ++check)); do
    if curl --silent --fail --max-time 5 "$base_url/v1/health" > "$out/health.json"; then
        break
    fi
    kill -0 "$server_pid"
    sleep 2
done
test -s "$out/health.json"
timeout --kill-after=20s 110m "$py" -Bu "$here/evaluate.py" --base-url "$base_url" \
    --release "$pro" --motor-python "$legacy/envs/openpi-215abfb/bin/python" --output "$out/metrics" > "$out/run.log" 2>&1
