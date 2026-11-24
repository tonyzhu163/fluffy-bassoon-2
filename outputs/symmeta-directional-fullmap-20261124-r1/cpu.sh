#!/usr/bin/env bash
# PIN STAGE NEW_ATTEMPT [upstream arguments]. Charge every stage against the same 900 s envelope.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
code=$(git -C "$here" rev-parse --show-toplevel)
pin=${1:?pin}; stage=${2:?qualify/select/replay}; attempt=${3:?new cpu attempt}
shift 3
[[ $(git -C "$code" rev-parse HEAD) == "$pin" && -z $(git -C "$code" status --porcelain --untracked-files=no) ]]
[[ $attempt =~ ^cpu[0-9]+$ ]]
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/$(basename "$here")
up=$root/outputs/symmeta-active-map-reader-20261124-r1
mkdir -p "$run"
exec 9>"$run/.cpu-budget.lock"
flock -n 9
used=0
for f in "$run"/cpu*/exit.txt; do
    if [[ -e $f ]]; then read -r rc start finish < "$f"; used=$((used+finish-start)); fi
done
left=$((900-used))
(( left > 5 ))
case "$stage" in
  qualify) script=qualify.py; args=("$up/cpu01");;
  select) [[ ${1:-} =~ ^gpu[0-9]+$ && -f "$run/$1/forward-receipt.json" && $(cut -d' ' -f1 "$run/$1/exit.txt") == 0 ]]
          [[ ! -e "$run/selection-sealed" ]]; script=select_gate.py; args=("$run/$1");;
  replay) [[ ${1:-} =~ ^cpu[0-9]+$ && ${2:-} =~ ^gpu[0-9]+$ && -f "$run/selection-sealed" ]]
          [[ ${3:?independent source hash} =~ ^[a-f0-9]{64}$ ]]
          [[ $(cat "$run/$1/source-selection.sha256") == "$3" && $(cat "$run/selection-sealed") == "$1 $3" ]]
          [[ ! -e "$run/replay-complete" ]]; script=replay.py; args=("$run/$1" "$up/$2" "$up/cpu01" "$3");;
  *) exit 2;;
esac
mkdir "$run/$attempt"
out=$run/$attempt
df -h "$out" > "$out/storage.txt"; df -i "$out" >> "$out/storage.txt"
start=$(date +%s)
echo "$pin $(hostname) $stage $start budget_left=$left core=48" > "$out/launch.txt"
trap 'rc=$?; echo "$rc $start $(date +%s)" > "$out/exit.txt"' EXIT
taskset -c 48 true
awk '$1=="cpu48"' /proc/stat > "$out/idle.txt"
sleep 2
awk '$1=="cpu48"' /proc/stat >> "$out/idle.txt"
awk 'NR==1{i=$5; t=$2+$3+$4+$5+$6+$7+$8+$9; next} {n=$2+$3+$4+$5+$6+$7+$8+$9-t; if(n<=0 || ($5-i)/n<0.9) exit 1} END{if(NR!=2) exit 1}' "$out/idle.txt"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$code/src" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=$root/setups/libero-meta-learning-20261117-r1/env/bin/python
TIMEFORMAT='wall=%R user=%U sys=%S'
{ time timeout -k 1 "$((left - $(date +%s) + start - 1))" taskset -c 48 "$py" -Bu "$here/$script" "$out" "${args[@]}" > "$out/$stage.log" 2>&1; } 2> "$out/time.txt"
case "$stage" in
 select) echo "$attempt $(cat "$out/source-selection.sha256")" > "$run/selection-sealed";;
 replay) echo "$attempt" > "$run/replay-complete";;
esac
