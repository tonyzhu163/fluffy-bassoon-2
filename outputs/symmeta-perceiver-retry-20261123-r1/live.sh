#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
campaign=xemb-icml-campaign-20261121-r1
setup="$root/setups/$campaign"
sim="$root/setups/libero-meta-learning-20261117-r1"
python="$legacy/envs/openpi-215abfb/bin/python"
panels=(gpu09test gpu09iiwa gpu06kinova3 gpu08pi0fix)

environment() {
    export CROSSARM_RELEASE="$root/sources/libero-pro2-e5500c69" LIBERO_PRO_ROOT="$sim/simulator-source"
    export LIBERO_CONFIG_PATH="$sim/libero-config" OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    local libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02"
    local drivers="" path
    local -a paths
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n $path && ! -f "$path/libtorch_python.so" ]]; then drivers+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$drivers"
}

invoke() {
    local panel=$1 script=$2 folder
    shift 2
    case "$panel" in
        gpu09test) folder=source-ff2ddf0a ;;
        gpu09iiwa|gpu08pi0fix) folder=source-wave2d ;;
        gpu06kinova3) folder=source-wave2b ;;
        *) return 2 ;;
    esac
    local src="$setup/$folder"
    PAIRED_DEPS="$src" PYTHONPATH="$setup/cv2-headless-overlay:$src/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$src/outputs/libero-crossarm-20261120-r1:$src/outputs/libero-xemb-ur5e-sysid-20261120-r1:$src/outputs/libero-xemb-ur5e-scale-20261120-r1:$src/outputs/$campaign" \
        "$python" -Bu "$here/$script" "$@"
}

check_tree() {
    local head dirty
    head=$(git -C "$1" rev-parse HEAD)
    dirty=$(git --no-optional-locks -C "$1" status --porcelain)
    [[ $head == "$2" && -z $dirty ]] || { echo "Wrong or dirty checkout $1 at $head" >&2; return 1; }
}

prepare_inputs() {
    local panel
    for panel in "${panels[@]}"; do
        mkdir "$out/$panel"
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu invoke "$panel" live.py prepare "$panel" "$out" \
            --predictions-sha "$prediction_sha" > "$out/$panel/prepare.log" 2>&1
    done
    "$python" -B "$here/paired_records.py" seal "$out" --predictions-sha "$prediction_sha"
}

run_lanes() {
    local lane task panel shard rc=0 pid stage_rc token egl
    local -a pids=() selectors=()
    export FLUFFY_GPU_EXECUTION=baige JAX_PLATFORMS=cuda NCCL_P2P_DISABLE=1
    export __EGL_VENDOR_LIBRARY_FILENAMES="$root/setups/libero90-full-action-20261119-r1/desktop-libs02/nvidia-vendor.json"
    unset LIBGL_ALWAYS_SOFTWARE
    # Some containers supply only NVIDIA_VISIBLE_DEVICES. Preserve explicit CUDA masks.
    # The probe checks every physical scheduler UUID against the CUDA PCI inventory.
    if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then
        export CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((count-1)))
    fi
    PYTHONPATH="$code/src" "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
        "$code/scripts/[train]gpu_probe.py" --expected-count "$count" --purpose evaluation --baige-container \
        --output "$out/gpu.json" > "$out/probe.log" 2>&1
    PYTHONPATH="$code/src" "$python" -Bu "$here/paired_devices.py" "$out" "$count" "$pin"
    mapfile -t selectors < <("$python" -B - "$out" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
receipt = p/"receipts"/((p/"execution.sha256").read_text().strip()+".json")
for d in json.loads(receipt.read_bytes())["devices"]:
    print(d["visibility_token"], d["egl_device"])
PY
)
    [[ ${#selectors[@]} == "$count" ]]
    for ((lane=0; lane<count; lane++)); do
        (
            read -r token egl <<< "${selectors[lane]}"
            export CUDA_VISIBLE_DEVICES="$token" MUJOCO_EGL_DEVICE_ID="$egl"
            local failures=0
            for ((task=lane; task<32; task+=count)); do
                panel=${panels[task/8]}
                shard=$((task%8))
                stage_rc=0
                invoke "$panel" live.py shard "$panel" "$out" --shard "$shard" --lane "$lane" \
                    --reuse "${reuse[@]}" > "$out/$panel/shard-$shard.log" 2>&1 || stage_rc=$?
                echo "$stage_rc" > "$out/$panel/shard-$shard.rc"
                if ((stage_rc)); then failures=1; fi
            done
            exit "$failures"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    (( rc == 0 )) || return "$rc"
    "$python" -B "$here/paired_records.py" reduce "$out"
}

payload() {
    local mode=$1 pin=$2 out=$3 predictions=$4 prediction_sha=$5 count=$6
    shift 6
    local -a reuse=("$@")
    check_tree "$code" "$pin"
    environment
    check_tree "$CROSSARM_RELEASE" e5500c690c1498eca1b7050350654964190cdf70
    check_tree "$legacy/third_party/openpi-215abfb" 215abfb217dbac7d5f1273282331b9b1866c0479
    cp -- "$predictions" "$out/predictions.json"
    printf '%s  %s\n' "$prediction_sha" "$out/predictions.json" | sha256sum -c -
    prepare_inputs
    if [[ $mode == run ]]; then run_lanes; fi
}

main() {
    [[ $# -ge 7 ]] || { echo 'Usage: live.sh prepare|run PIN NEW_ATTEMPT PREDICTIONS SHA256 GPU_COUNT WALL_SECONDS [PRIOR_ATTEMPT ...]' >&2; return 2; }
    local mode=$1 pin=$2 attempt=$3 predictions=$4 prediction_sha=$5 count=$6 wall=$7
    shift 7
    [[ $mode == prepare || $mode == run ]]
    [[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && $predictions == /* ]]
    [[ $prediction_sha =~ ^[0-9a-f]{64}$ && $wall =~ ^[1-9][0-9]*$ && $(uname -s) == Linux ]]
    if [[ $mode == run ]]; then
        [[ $(hostname) == job-*-master-0 && ( $count == 4 || $count == 8 ) ]]
    else
        [[ $count == 0 ]]
    fi
    local run="$root/outputs/$(basename "$here")" old
    local -a sources=()
    for old in "$@"; do
        [[ $old =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && $attempt != "$old" && -f "$run/$old/contract.json" ]]
        sources+=("$run/$old")
    done
    out="$run/$attempt"
    mkdir -p "$run"
    mkdir "$out"
    start=$(date +%s)
    export out start
    trap 'rc=$?; trap - EXIT; if [[ -f "$out/contract.json" ]]; then "$python" -B "$here/paired_records.py" reduce "$out" --partial > "$out/partial.log" 2>&1 || rc=1; fi; echo "$rc $start $(date +%s)" > "$out/exit.txt"; exit "$rc"' EXIT
    printf '%s\n' "host=$(hostname)" "commit=$pin" "mode=$mode" "cards=$count" "wall_seconds=$wall" \
        "predictions=$predictions" "predictions_sha256=$prediction_sha" "reuse=${sources[*]}" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"
    timeout --kill-after=5s "${wall}s" bash -euo pipefail -c \
        'source "$1"; shift; payload "$@"' bash "$here/live.sh" \
        "$mode" "$pin" "$out" "$predictions" "$prediction_sha" "$count" "${sources[@]}" > "$out/run.log" 2>&1
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
