#!/usr/bin/env bash
# Submit in one exclusive Baige Distributed Training container.
set -Eeuo pipefail
pin=${1:?expected Git revision}
attempt=${2:?fresh attempt identifier}
[[ $# == 2 && $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^native[0-9]+$ ]]
here=$(cd "$(dirname "$0")" && pwd)
code=$(git -C "$here" rev-parse --show-toplevel)
host=$(hostname)
[[ $(uname -s) == Linux && $host == job-*-master-0 ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" ]]
if git -C "$code" symbolic-ref -q HEAD; then exit 2; fi
git -C "$code" diff --quiet HEAD --
[[ -z $(git -C "$here" ls-files --others --exclude-standard -- '*.py' '*.sh') ]]
for f in run.sh native.py model.py teacher.py; do git -C "$here" ls-files --error-unmatch "$f" >/dev/null; done

project=/mowenjie/fluffy-bassoon-2
r2=libero-paligemma-dualarm-20261120-r2
campaign="$project/outputs/$r2"
setup="$project/setups/libero-meta-learning-20261117-r1"
out="$project/outputs/$(basename "$here")/$attempt"
mkdir "$out"
exec > >(tee "$out/run.log") 2>&1
started=$(date +%s)
sampler=
finish() {
    local rc=$?
    if [[ -n $sampler ]]; then kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; fi
    printf '%s %s %s\n' "$rc" "$(date +%s)" "$started" > "$out/exit.txt"
}
trap finish EXIT
trap 'printf "Failed at line %s (status %s)\n" "$LINENO" "$?" >&2' ERR
trap 'exit 143' TERM
printf '%s %s %s %s\n' "$pin" "$host" "$$" "$started" > "$out/launch.txt"
(cd "$here" && sha256sum *.py *.sh) > "$out/source.sha256"
{
    date -Ins
    findmnt -T "$out" -o TARGET,FSTYPE,OPTIONS
    df -hP "$out"
    df -iP "$out"
    if command -v quota >/dev/null; then quota -s || true; else echo 'quota unavailable'; fi
    cat /sys/fs/cgroup/{cpu.max,memory.max}
} > "$out/allocation.txt"

export R2_CHECKOUT="$project/setups/$r2/source-fc61bf2a"
export MOTOR_CODE_REVISION=fc61bf2afb48d4639664db93dae64ae9cb04e2df
export CROSSARM_RELEASE="$project/sources/libero-pro2-e5500c69"
export LIBERO_PRO_ROOT="$setup/simulator-source" LIBERO_CONFIG_PATH="$setup/libero-config"
export CROSSARM_SIM_PYTHON="$setup/env/bin/python"
export VLM_PYTHON="$project/setups/robocasa365-native-20261101-r1/env/bin/python"
export DUAL_OUTPUT="$out" DUAL_ARCHIVE="$project/outputs/libero-paligemma-crossarm-20261120-r1/assets01/goal5-base.hdf5"
export DUAL_WEIGHTS=/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-mix-224/snapshots/d1d8734c9c3ad0ccfeea4afc270faa356c2ba515
export PYTHONPATH="$here:$R2_CHECKOUT/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$R2_CHECKOUT/outputs/$r2:$R2_CHECKOUT/outputs/libero-paligemma-crossarm-20261120-r1"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export FLUFFY_GPU_EXECUTION=baige MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
[[ $(git -C "$R2_CHECKOUT" rev-parse HEAD) == "$MOTOR_CODE_REVISION" ]]
git -C "$R2_CHECKOUT" diff --quiet HEAD --
[[ $(git -C "$CROSSARM_RELEASE" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
test -r "$DUAL_WEIGHTS/model.safetensors.index.json"
test -r "$DUAL_ARCHIVE"
fit="$campaign/sft02/fit"
panel="$campaign/large-panel-v3"
(
  cd "$campaign"
  sha256sum -c - <<'SHA'
77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f  sft02/fit/candidates/step-006400.pt
307e81088eb5b67506e8b30b3348f564c1c142f7e293e6db34f3149776929e80  sft02/fit/manifest.json
663077ec7aad3621173d725d2cf1550e3859a7ca46a0de50727e25aec8c6bfc0  large-panel-v3/dev/manifest.json
7e2122efbd976638680c8215c3830eeb974179c5944f5185b38771988a3d464a  large-panel-v3/dev/scores/77d18c386b2cc176252cfb68449e4415748382f554046ca50367ed4592676d9f/stride-5/report.json
SHA
) | tee "$out/input-verification.txt"

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid != *$'\n'* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $gpu != *$'\n'* ]]
case "$gpu" in
    *6000D*|*RPBZZZ6*) export NCCL_P2P_DISABLE=1 ;;
    *4090*) ;;
    *) echo "Unqualified accelerator: $gpu" >&2; exit 2 ;;
esac
libs="$project/setups/libero90-full-action-20261119-r1/desktop-libs02"
glvnd="$libs/prefix/usr/lib/x86_64-linux-gnu"
for lib in libEGL.so.1 libOpenGL.so.0 libX11.so.6; do test -r "$glvnd/$lib"; done
test -r "$libs/nvidia-vendor.json"
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
driver_paths=""
for path in "${paths[@]}"; do
    if [[ -n $path && ! -f $path/libtorch_python.so ]]; then driver_paths+=":$path"; fi
done
export LD_LIBRARY_PATH="$glvnd:$libs/prefix/lib/x86_64-linux-gnu$driver_paths"
export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
cd "$out"
stage() {
    local name=$1
    shift
    printf '%s\t%s\tstart\n' "$(date -Ins)" "$name" | tee -a stages.tsv
    "$@" 2>&1 | tee "$name.log"
    printf '%s\t%s\tcomplete\n' "$(date -Ins)" "$name" | tee -a stages.tsv
}
stage probe "$VLM_PYTHON" -Bu "$R2_CHECKOUT/scripts/[train]gpu_probe.py" --expected-count 1 --purpose training \
    --baige-container --baige-exclusive-training --output "$out/gpu.json"
# Read-only telemetry identifies the model process and its concurrent native children.
(
    while true; do
        date -Ins
        nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader
        ps -eo pid,ppid,comm,args | awk -v p="$here/native" 'index($0,p) && $3 ~ /^python/{print $1,$2,$3,$6,$7}'
        sleep 20
    done
) > "$out/telemetry.log" 2>&1 &
sampler=$!
stage selftest "$VLM_PYTHON" -Bu "$here/model.py" selftest
common=(--out "$out" --panel "$panel" --fit "$fit")
render=(--weights "$DUAL_WEIGHTS" --workers 8)
stage prepare "$CROSSARM_SIM_PYTHON" -Bu "$here/native.py" prepare "${common[@]}"
stage smoke "$VLM_PYTHON" -Bu "$here/native.py" smoke "${common[@]}" "${render[@]}"
stage collect "$VLM_PYTHON" -Bu "$here/native.py" collect "${common[@]}" "${render[@]}"
stage fit "$VLM_PYTHON" -Bu "$here/native.py" fit --out "$out" --model "$out/fitted"
sha256sum "$out"/fitted/{model.npz,fit.json} > "$out/fitted.sha256"
stage evaluate "$VLM_PYTHON" -Bu "$here/native.py" evaluate "${common[@]}" "${render[@]}" --model "$out/fitted"
sha256sum -c "$out/fitted.sha256"
test -s "$out/evaluation.json"
