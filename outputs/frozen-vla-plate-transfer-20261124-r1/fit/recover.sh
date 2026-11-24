#!/usr/bin/env bash
# Use scheduler-local scratch for the original trainer and reserve every shared output inode up front.
set -euo pipefail
if [[ ${1:-} == run ]]; then
    shift
    exec timeout --kill-after=15s 5280s bash "$0" job "$@"
fi
mode=${1:?reserve or run}
wrapper=${2:?wrapper revision}
attempt=${3:?new attempt}
[[ $wrapper =~ ^[0-9a-f]{40}$ && $attempt =~ ^gpu[0-9]+$ ]]
pin=7f749ae21b8d74ad58536c00ae4aafd5ffca45c5
seal=3e133b69d850318cb20d61549e2f96ce6a7da8548cf05d25af7401ec28047c78
checkpoint=df65496e1c012e7da293a5795bb447cdac9c0baa0c0bce96fb25501a87b09b53
legacy=/mowenjie/fluffy-bassoon
base=/mowenjie/fluffy-bassoon-2
role=outputs/frozen-vla-plate-transfer-20261124-r1/fit
parent=$base/$role
code=$parent/code-$pin
data=$parent/../data/dataset-v1
prior=$parent/gpu02
out=$parent/$attempt
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $(git -C "$here" rev-parse HEAD) == "$wrapper" && -z $(git -C "$here" status --porcelain --untracked-files=no) ]]
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
[[ $(sha256sum "$data/seal.json" | cut -d' ' -f1) == "$seal" ]]
[[ $(sha256sum "$prior/optimizer.msgpack" | cut -d' ' -f1) == "$checkpoint" ]]
files=(config.json native-before.json heldout-inputs.json heldout-windows.json source-before.json
       optimizer.msgpack progress.jsonl source-after.json native-after.json)
for arm in panda mixed; do
    for f in windows.json train-inputs.json head.npz fit.json; do files+=("$arm/$f"); done
done
# Publish the family receipt last. There is no partial fit.json at the advertised path.
files+=(fit.json)
if [[ $mode == reserve ]]; then
    mkdir "$out"
    mkdir "$out/panda" "$out/mixed"
    for f in "${files[@]}"; do : > "$out/$f.reserved"; done
    for f in console.log exit.txt.reserved gpu.json.reserved checkpoint-0.msgpack checkpoint-1.msgpack \
             progress-live.jsonl storage.txt reservation.txt launch.txt; do : > "$out/$f"; done
    # Exercise the same preallocated slots that the periodic writer will reuse.
    for slot in 0 1; do
        cat "$prior/optimizer.msgpack" > "$out/checkpoint-$slot.msgpack"
        [[ $(sha256sum "$out/checkpoint-$slot.msgpack" | cut -d' ' -f1) == "$checkpoint" ]]
    done
    df -h "$out" > "$out/storage.txt"
    df -i "$out" >> "$out/storage.txt"
    quota -s >> "$out/storage.txt" 2>/dev/null || true
    printf '%s\n' "$wrapper" "$pin" "$seal" "$checkpoint" > "$out/reservation.txt"
    printf 'Reserved %s shared files; backup readback succeeded.\n' "$(find "$out" -type f | wc -l)"
    cat "$out/storage.txt"
    exit 0
fi
[[ $mode == job && $(hostname) == job-*-master-0 ]]
[[ $(head -1 "$out/reservation.txt") == "$wrapper" ]]
mv -T "$out/reservation.txt" "$out/reservation-claimed.txt"
exec >> >(tee -a "$out/console.log") 2>&1
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt.reserved"; mv -T "$out/exit.txt.reserved" "$out/exit.txt"' EXIT
printf '%s\n' "$(hostname)" "$pin" "$attempt" "$seal" "$start" "$wrapper" "$checkpoint" > "$out/launch.txt"
scratch=$(mktemp -d /tmp/plate-fit-XXXXXXXX)
# Verify genuine machine-local working storage before reading model weights.
[[ $(stat -c %d "$scratch") != "$(stat -c %d "$out")" ]]
df -h "$scratch" >> "$out/storage.txt"
df -i "$scratch" >> "$out/storage.txt"
export TMPDIR=$scratch/tmp XDG_CACHE_HOME=$scratch/runtime-cache
mkdir "$TMPDIR" "$XDG_CACHE_HOME"
export OPENPI_DATA_HOME=$legacy/cache/openpi-215abfb
export PYTHONPATH=$base/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay:$code/src
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 FLUFFY_GPU_EXECUTION=baige
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
IFS=: read -ra paths <<< "${LD_LIBRARY_PATH:-}"
clean=()
for p in "${paths[@]}"; do
    if [[ -n $p && ! -f $p/libtorch_python.so ]]; then clean+=("$p"); fi
done
export LD_LIBRARY_PATH=$(IFS=:; echo "${clean[*]}")
name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $name != *$'\n'* && $name == *4090* ]]
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* && $uuid == "${NVIDIA_VISIBLE_DEVICES:?}" ]]
    export CUDA_VISIBLE_DEVICES=0
fi
cpus=$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
probe=$base/setups/robocasa365-native-20261101-r1/env/bin/python
taskset -c "$cpus" "$probe" -Bu "$code/scripts/[train]gpu_probe.py" --expected-count 1 \
    --baige-container --baige-exclusive-training --purpose training --output "$scratch/gpu.json"
cat "$scratch/gpu.json" > "$out/gpu.json.reserved"
mv -T "$out/gpu.json.reserved" "$out/gpu.json"
python=$legacy/envs/openpi-215abfb/bin/python
taskset -c "$cpus" "$python" -Bu "$code/$role/fit_plate.py" --data "$data" --seal "$seal" \
    --out "$scratch" --pin "$pin" --prior "$prior" &
worker=$!
slot=1
last=""
while kill -0 "$worker" 2>/dev/null; do
    if [[ -s $scratch/optimizer.msgpack && -s $scratch/progress.jsonl ]]; then
        # Local trainer atomically replaces optimizer.msgpack. Capture a single opened inode.
        signature=$(stat -c '%i %s %Y' "$scratch/optimizer.msgpack")
        if [[ $signature != "$last" ]]; then
            slot=$((1-slot))
            cat "$scratch/optimizer.msgpack" > "$out/checkpoint-$slot.msgpack"
            cat "$scratch/progress.jsonl" > "$out/progress-live.jsonl"
            last=$signature
        fi
    fi
    sleep 4
done
wait "$worker"
for f in "${files[@]}"; do
    [[ -s $scratch/$f && -f $out/$f.reserved ]]
    cat "$scratch/$f" > "$out/$f.reserved"
    cmp -s "$scratch/$f" "$out/$f.reserved"
    mv -T "$out/$f.reserved" "$out/$f"
done
echo "Sealed $out/fit.json $(sha256sum "$out/fit.json" | cut -d' ' -f1)"
