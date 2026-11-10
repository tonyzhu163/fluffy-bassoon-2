#!/usr/bin/env bash
set -Eeuo pipefail
root=/mowenjie/fluffy-bassoon-2
run=outputs/robocasa-bowl-dpt-labels-20261110-r1
code=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
src=$code/$run
base=$root/outputs/robocasa-sft-reproduction-20261109-r1
study=$root/$run
py=$root/setups/robocasa365-native-20261101-r1/env/bin/python
pin=$1 split=$2 id=$3 gate=$4
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
"$py" -B -c 'import json,sys; assert json.load(open(sys.argv[1]))["qualifies"] is True' "$gate"
case "$split" in
    train) number=24; first=987200 ;;
    validation) number=8; first=987400 ;;
    confirmation) number=16; first=987600 ;;
    *) exit 2 ;;
esac
[[ $id =~ ^[a-z0-9-]+$ ]]
out=$study/$id
export LD_LIBRARY_PATH="$study/dev-graphics-01/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$base/sources/openpi/src:$base/sources/openpi/packages/openpi-client/src:$base/sources/robocasa:$code/src:$code/outputs/robocasa-sft-reproduction-20261109-r1:$code/outputs/robocasa-meta-feedback-20261109-r1:$code/outputs/robocasa-compact-bowls-20261109-r1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=73 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
"$py" -Bu "$src/admit.py" "$first" "$number" "$out-geometry"
mapfile -t seeds < <("$py" -B -c 'import json,sys; print(*json.load(open(sys.argv[1]))["admitted"],sep="\n")' "$out-geometry/admitted.json")
test "${#seeds[@]}" = "$number"
bash "$src/run.sh" "$pin" "$id" "${seeds[@]}"
"$py" -B "$src/readout.py" "$out" "$out-geometry/admitted.json" > "$out/coverage.json"
"$py" -B "$src/export_targets.py" "$out" "$out-geometry/admitted.json" "$root/inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json" "$out/targets"
"$py" -B "$src/history_check.py" "$out" "$out-geometry/admitted.json" > "$out/history-diagnostic.json"
