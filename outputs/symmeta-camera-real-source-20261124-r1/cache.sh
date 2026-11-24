#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/env.sh"
pin=${1:?deployed cache pin}
render=${2:?completed render attempt}
attempt=${3:?fresh cache attempt}
[[ $pin =~ ^[0-9a-f]{40}$ && $(git -C "$here" rev-parse HEAD) == "$pin" && $attempt =~ ^[0-9a-z_-]+$ && $render =~ ^[0-9a-z_-]+$ ]]
[[ $(uname -s) == Linux && -z $(git --no-optional-locks -C "$here" status --porcelain --untracked-files=no) ]]
[[ $(awk '{print $1}' "$run/$render/exit.txt") == 0 ]]
exec 9>"$run/prep.lock"
flock -n 9
out=$run/$attempt
mkdir "$out"
used=$(awk '{s+=$1} END {print int(s)}' "$run"/*/wall.seconds)
remaining=$((2100-used))
((remaining>5))
start=$(date +%s)
trap 'rc=$?; end=$(date +%s); echo $((end-start)) > "$out/wall.seconds"; echo "$rc $start $end" > "$out/exit.txt"' EXIT
echo "$pin $(hostname) $start $remaining" > "$out/launch.txt"
df -h "$out" > "$out/storage.txt"
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void JAX_PLATFORMS=cpu
python3 - <<'PY' > "$out/idle.json"
import json,time
def sample():
    with open('/proc/stat') as f:
        return {s[0]:list(map(int,s[1:9])) for line in f if (s:=line.split())[0] in ('cpu42','cpu43','cpu44','cpu45')}
a=sample();time.sleep(2);b=sample()
d={k:[j-i for i,j in zip(a[k],b[k])] for k in a}
print(json.dumps(d));assert len(d)==4 and all(x[3]>.9*sum(x) for x in d.values()),d
PY
runpart(){
    local core=$1
    shift
    timeout -k 2s "$((start+remaining-$(date +%s)-2))s" taskset -c "$core" "$py" -Bu "$here/real_bank.py" "$1" "$run/$render" "$out" "${@:2}"
}
runpart 42 allocate > "$out/init.log" 2>&1
pids=()
for slot in 0 1 2 3; do
    runpart "$((42+slot))" build --slot "$slot" > "$out/cache-$slot.log" 2>&1 &
    pids+=($!)
done
rc=0
for child in "${pids[@]}"; do wait "$child" || rc=1; done
(( rc == 0 ))
runpart 42 seal --digital "$primary/prep01/bank" > "$out/seal.log" 2>&1
