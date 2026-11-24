#!/usr/bin/env bash
set -eu -o pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel)
export STATS_PIN=${1:?full commit}
mode=${2:?qualify or analyze}
out=${3:?new attempt directory}
shift 3
[[ $(uname -s) == Linux && "$STATS_PIN" =~ ^[0-9a-f]{40}$ && $(git -C "$repo" rev-parse HEAD) == "$STATS_PIN" ]]
[[ -z $(git -C "$repo" status --porcelain --untracked-files=no) ]]
mkdir "$out"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
cpu=${STATS_CPU:-31}
[[ "$cpu" =~ ^[0-9]+$ ]]
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
exec 2> >(tee "$out/stderr.log" >&2)
echo "host=$(hostname) pin=$STATS_PIN cpu=$cpu mode=$mode python=$py" > "$out/launch.txt"
case "$mode" in
    qualify) taskset -c "$cpu" "$py" -Bu "$here/qualify.py" "$@" > "$out/qualification.json" ;;
    analyze) taskset -c "$cpu" "$py" -Bu "$here/analyze.py" "$@" --out "$out/analysis.json" ;;
    *) exit 2 ;;
esac
