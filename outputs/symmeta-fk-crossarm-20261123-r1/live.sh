#!/usr/bin/env bash
set -euo pipefail
start=$(date +%s)
[[ $# == 6 ]] || { echo 'Usage: live.sh PIN NEW_ATTEMPT GPU_COUNT WALL_SECONDS MANIFEST MANIFEST_SHA256' >&2; exit 2; }
pin=$1 attempt=$2 count=$3 wall=$4 manifest=$5 manifest_sha=$6
[[ $pin =~ ^[0-9a-f]{40}$ && $attempt =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
[[ $count =~ ^[1-8]$ && $wall =~ ^[1-9][0-9]{0,4}$ && $manifest == /* && $manifest_sha =~ ^[0-9a-f]{64}$ ]]
(( wall > 10 && count * wall <= 28800 ))
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
root=/mowenjie/fluffy-bassoon-2
legacy=/mowenjie/fluffy-bassoon
campaign=xemb-icml-campaign-20261121-r1
[[ $(hostname) == job-*-master-0 && "$here" == "$code/outputs/symmeta-fk-crossarm-20261123-r1" ]]
out="$root/outputs/symmeta-fk-crossarm-20261123-r1/$attempt"
mkdir "$out"
finish() {
    local rc=$?
    printf '%s %s %s\n' "$rc" "$start" "$(date +%s)" > "$out/exit.txt"
    python3 - "$out" "$rc" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
panels = {}
for panel in ("gpu09iiwa", "gpu06kinova3", "gpu08pi0fix"):
    p = out / panel
    withheld = []
    if (p / "panel.json").is_file():
        withheld = [r["index"] for r in json.loads((p / "panel.json").read_text())["rows"]
                    if r["withheld_reason"] is not None]
    missing = [i for i in range(96) if i not in withheld and not (p / f"shard-{i//12}/{i:02d}.json").is_file()]
    panels[panel] = dict(prepared=(p/"panel.json").is_file(), merged=(p/"summary.json").is_file(),
                         missing_indices=missing, withheld_indices=withheld, completed_pairs=96-len(missing)-len(withheld),
                         failed_stages={f.stem: int(f.read_text()) for f in p.glob("*.rc") if int(f.read_text())})
(out / "status.json").write_text(json.dumps(dict(exit_code=int(sys.argv[2]), panels=panels), indent=2))
PY
}
trap finish EXIT
export here code root legacy campaign out count wall pin manifest manifest_sha
payload() {
    local sim="$root/setups/libero-meta-learning-20261117-r1" release="$root/sources/libero-pro2-e5500c69"
    local libs="$root/setups/libero90-full-action-20261119-r1/desktop-libs02" setup="$root/setups/$campaign"
    local python="$legacy/envs/openpi-215abfb/bin/python"
    local paths path driver_paths="" device task panel stage_rc rc=0 pid
    local -a panels=(gpu09iiwa gpu06kinova3 gpu08pi0fix) pids=()
    [[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git --no-optional-locks -C "$code" status --porcelain) ]]
    [[ $(git -C "$release" rev-parse HEAD) == e5500c690c1498eca1b7050350654964190cdf70 ]]
    [[ $(git -C "$legacy/third_party/openpi-215abfb" rev-parse HEAD) == 215abfb217dbac7d5f1273282331b9b1866c0479 ]]
    [[ -z $(git --no-optional-locks -C "$legacy/third_party/openpi-215abfb" status --porcelain) ]]
    cp -- "$manifest" "$out/manifest.json"
    printf '%s  %s\n' "$manifest_sha" "$out/manifest.json" | sha256sum -c -
    printf '%s\n' "commit=$pin" "tree=$code" "host=$(hostname)" "cards=$count" "wall_seconds=$wall" \
        "manifest=$manifest" "sha256=$manifest_sha" > "$out/config.txt"
    df -h "$out" > "$out/storage.txt"
    export CROSSARM_RELEASE="$release" LIBERO_PRO_ROOT="$sim/simulator-source" LIBERO_CONFIG_PATH="$sim/libero-config"
    export OPENPI_DATA_HOME="$legacy/cache/openpi-215abfb" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    export JAX_PLATFORMS=cuda FLUFFY_GPU_EXECUTION=baige XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    unset LIBGL_ALWAYS_SOFTWARE
    IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
    for path in "${paths[@]}"; do
        if [[ -n "$path" && ! -f "$path/libtorch_python.so" ]]; then driver_paths+=":$path"; fi
    done
    export LD_LIBRARY_PATH="$libs/prefix/usr/lib/x86_64-linux-gnu$driver_paths"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$libs/nvidia-vendor.json"
    invoke() {
        local mode=$1 name=$2 folder src
        shift 2
        case "$name" in
            gpu06kinova3) folder=source-wave2b ;;
            gpu09iiwa|gpu08pi0fix) folder=source-wave2d ;;
            *) return 2 ;;
        esac
        src="$setup/$folder"
        PYTHONPATH="$setup/cv2-headless-overlay:$src/src:$release:$sim/simulator-source:$src/outputs/libero-crossarm-20261120-r1:$src/outputs/libero-xemb-ur5e-sysid-20261120-r1:$src/outputs/libero-xemb-ur5e-scale-20261120-r1:$src/outputs/$campaign" \
            "$python" -Bu "$here/live.py" "$mode" "$name" "$out/$name" "$@"
    }
    for panel in "${panels[@]}"; do
        mkdir "$out/$panel"
        stage_rc=0
        JAX_PLATFORMS=cpu invoke prepare "$panel" --manifest "$out/manifest.json" --sha256 "$manifest_sha" \
            > "$out/$panel/prepare.log" 2>&1 || stage_rc=$?
        echo "$stage_rc" > "$out/$panel/prepare.rc"
        if (( stage_rc )); then rc=$stage_rc; fi
    done
    (( rc == 0 )) || return "$rc"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((count-1)))
    "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu "$code/scripts/[train]gpu_probe.py" \
        --expected-count "$count" --purpose training --baige-container --baige-exclusive-training \
        --output "$out/gpu.json" > "$out/probe.log" 2>&1
    python3 - "$out/gpu.json" <<'PY'
import json, sys
for device in json.load(open(sys.argv[1]))["devices"]:
    name = device["cuda_name"].upper()
    if "RTX PRO 6000" not in name and name != "NVIDIA RPBZZZ6":
        raise SystemExit(f"Expected qualified RTX PRO 6000-series GPU, found {name}")
PY
    for ((device=0; device<count; device++)); do
        (
            export CUDA_VISIBLE_DEVICES="$device" MUJOCO_EGL_DEVICE_ID="$device"
            for ((task=device; task<24; task+=count)); do
                panel=${panels[task/8]}
                shard=$((task%8))
                stage_rc=0
                invoke shard "$panel" --shard "$shard" > "$out/$panel/shard-$shard.log" 2>&1 || stage_rc=$?
                echo "$stage_rc" > "$out/$panel/shard-$shard.rc"
            done
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    for panel in "${panels[@]}"; do
        stage_rc=0
        JAX_PLATFORMS=cpu invoke merge "$panel" > "$out/$panel/merge.log" 2>&1 || stage_rc=$?
        echo "$stage_rc" > "$out/$panel/merge.rc"
        if (( stage_rc )); then rc=$stage_rc; fi
    done
    return "$rc"
}
export -f payload
remaining=$((start + wall - 8 - $(date +%s)))
(( remaining > 0 ))
timeout --kill-after=5s "${remaining}s" bash -euo pipefail -c payload > "$out/run.log" 2>&1
