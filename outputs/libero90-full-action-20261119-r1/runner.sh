#!/usr/bin/env bash
# From a clean pinned checkout on Baige:
# runner.sh FULL_GIT_PIN STAGE EPISODE_ROOT [--shard-index I --shards N] [--meta FILE --imitation FILE]
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: bash runner.sh PIN {smoke|fitting|development|live-development|live-test} EPISODE_ROOT [options]" >&2
    exit 2
fi
pin=$1
stage=$2
output=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
shared=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
host=$(hostname)
[[ "$pin" =~ ^[0-9a-f]{40}$ && "$(git -C "$code" rev-parse HEAD)" == "$pin" ]] || {
    echo "Expected the full deployed Git revision" >&2; exit 1
}
[[ -z "$(git -C "$code" status --porcelain)" ]] || { echo "Source checkout is dirty" >&2; exit 1; }
sister=outputs/libero90-meta-retry-20261119-r1
own="outputs/$(basename "$here")"
required=("$own/runner.sh" "$own/rollouts.py" "$sister/collect.py" "$sister/standard_sim.py" "$sister/cases.json")
probe_mode=(--baige-container --purpose evaluation)
case "$stage" in
    smoke)
        [[ "$host" == aibox-* && $# == 0 ]] || { echo "Smoke takes exactly one fitting case on a Baige development machine" >&2; exit 2; }
        exec 9>>"$shared/outputs/libero-bowl-decisions-20261118-r1/device.lock"
        flock -w 600 9
        probe_mode=(--direct-hostname "$host" --purpose training)
        ;;
    fitting|development|live-development|live-test)
        [[ "$host" =~ ^job-[a-z0-9]+-(master|worker)-[0-9]+$ ]] || {
            echo "Use a Baige Distributed Training allocation for panel stages" >&2; exit 2
        }
        if [[ "$stage" == live-* ]]; then required+=("$own/model.py"); fi
        ;;
    *) echo "Unknown stage $stage" >&2; exit 2 ;;
esac
# These run directories are normally gitignored. Explicitly require tracked sources.
git -C "$code" ls-files --error-unmatch -- "${required[@]}" >/dev/null
if [[ "$stage" == smoke ]]; then
    [[ ${CGPU_COUNT:?} == 1 && ${NVIDIA_VISIBLE_DEVICES:?} == "${NVIDIA_VISIBLE_GPUS_UUID:?}" && "$NVIDIA_VISIBLE_DEVICES" != *,* ]]
    export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# Same OpenPI, stock LIBERO, EGL, and numerical runtime as the sister collector.
export OPENPI_ROOT="$legacy/third_party/openpi-215abfb" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
model_python="$legacy/envs/openpi-215abfb/bin/python"
probe_python="$shared/setups/robocasa365-native-20261101-r1/env/bin/python"
export READER_PYTHON="$probe_python"
export LIBERO_SIM_PYTHON=/mowenjie/envs/wam-libero-plus-py310/bin/python
export LIBERO_STD_ROOT="$shared/deps/libero-dsrl-87edbd1-20261029-r1/repo"
export LIBERO_CONFIG_PATH="$shared/cache/libero-dsrl-87edbd1-20261029-r1"
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$code/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export MAGICK_HOME=/mowenjie/libs/libero-plus-imagemagick
export MAGICK_CODER_MODULE_PATH="$MAGICK_HOME/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders"
export LD_LIBRARY_PATH="$MAGICK_HOME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "$stage" != smoke ]]; then
    # GLVND dispatch from the qualified runtime plus the allocated driver.
    glvnd="$shared/setups/libero-gen-history-pilot-20261111-r1/os-libs01/prefix/usr/lib/x86_64-linux-gnu"
    test -f "$glvnd/libEGL.so.1" && test -f "$glvnd/libGL.so.1"
    export LD_LIBRARY_PATH="$glvnd:$LD_LIBRARY_PATH"
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
output=$(realpath -m "$output")
[[ "$output/" == "$shared/outputs/"* && "$output/" != "$code/"* ]] || {
    echo "Episode root must be on the shared project output mount outside the source checkout" >&2; exit 2
}
mkdir -p "$output/.attempts"
attempt=$(mktemp -d "$output/.attempts/$stage-XXXXXXXX")
started=$(date +%s)
trap 'rc=$?; echo "$rc $(date +%s) $started" > "$attempt/exit.txt"' EXIT
printf '%s\n' "$host $pin $stage" > "$attempt/launch.txt"
df -h "$output" > "$attempt/storage.txt"
df -i "$output" >> "$attempt/storage.txt"
"$probe_python" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 "${probe_mode[@]}" \
    --output "$attempt/gpu.json" > "$attempt/probe.log" 2>&1
"$probe_python" -B -c 'import json,sys
gpu = json.load(open(sys.argv[1]))["devices"][0]["cuda_name"]
if not any(name in gpu for name in ("4090", "6000D", "RPBZZZ6")): sys.exit(f"Unqualified accelerator {gpu}")' "$attempt/gpu.json"
# The stage and provenance come solely from the verified positional arguments.
for flag in "$@"; do
    case "$flag" in --pin*|--output*|--stage*|--) echo "Reserved rollout argument $flag" >&2; exit 2;; esac
done
command=("$model_python" -Bu "$here/rollouts.py" "$stage" "$@" --pin "$pin" --output "$output")
if [[ "$stage" == smoke ]]; then command=(timeout --kill-after=15s 15m "${command[@]}"); fi
"${command[@]}" 2>&1 | tee "$attempt/rollout.log"
