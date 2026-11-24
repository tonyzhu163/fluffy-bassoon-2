#!/usr/bin/env bash
# Each stage gets a new attempt. Selection and target access require different invocations.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
pin=${1:?pin}; stage=${2:?stage}; attempt=${3:?attempt}; shift 3
root=/mowenjie/fluffy-bassoon-2
run=$root/outputs/$(basename "$here")
[[ $(git -C "$here" rev-parse HEAD) == "$pin" && -z $(git -C "$here" status --porcelain --untracked-files=no) ]]
[[ $attempt =~ ^cpu[0-9]+$ ]]
mkdir -p "$run"
exec 9>"$run/.owner-budget.lock"
flock -n 9
case "$stage" in
  source|archive) budget=900; meter=offline;;
  physical) budget=600; meter=physical;;
  *) exit 2;;
esac
used=0
for f in "$run"/cpu*/exit.txt; do
  [[ -e $f ]] || continue
  read -r rc start finish charge < "$f"
  if [[ $charge == "$meter" ]]; then used=$((used+finish-start)); fi
done
left=$((budget-used))
((left>5))
mkdir "$run/$attempt"
out=$run/$attempt
start=$(date +%s)
trap 'rc=$?; echo "$rc $start $(date +%s) $meter" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $stage $start core=49 remaining=$left" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
df -i "$out" >> "$out/storage.txt"
taskset -c 49 true
awk '$1=="cpu49"' /proc/stat > "$out/idle.txt"
sleep 2
awk '$1=="cpu49"' /proc/stat >> "$out/idle.txt"
awk 'NR==1{i=$5;t=$2+$3+$4+$5+$6+$7+$8+$9;next}{n=$2+$3+$4+$5+$6+$7+$8+$9-t;if(n<=0 || ($5-i)/n<.9)exit 1}END{if(NR!=2)exit 1}' "$out/idle.txt"
export UNCERTAIN_PIN="$pin" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$here/../../src" CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
limit=$((left + start - $(date +%s) - 1))
if [[ $stage == physical ]]; then
  export REACH_CPUS=49 REACH_CACHE="$out/cache" TMPDIR="$out/tmp" MPLCONFIGDIR="$out/cache/mpl"
  mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
  timeout -k 1 "$limit" bash "$here/../frozen-vla-reachability-20261123-r1/cpu.sh" "$here/physical.py" "$out" "$@" > "$out/run.log" 2>&1
else
  timeout -k 1 "$limit" taskset -c 49 "$root/setups/libero-meta-learning-20261117-r1/env/bin/python" -Bu "$here/$stage.py" "$out" "$@" > "$out/run.log" 2>&1
fi
