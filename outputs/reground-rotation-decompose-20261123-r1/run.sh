#!/usr/bin/env bash
# Run the decomposition arms on one Baige Distributed Training node. Launch logic as stream.sh.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
code=$(cd "$here/../.." && pwd -P)
root=${REGROUND_RUNTIME_ROOT:-/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1}
export REGROUND_RUNTIME_ROOT=$root
if [[ ${1:-} == _lanes ]]; then
    shift
    attempt=$1 cards=$2
    shift 2
    bash "$here/arms.sh" probe "$attempt" "$cards"
    pids=()
    for ((rank=0; rank<cards; rank++)); do
        bash "$here/arms.sh" lane --attempt "$attempt" --rank "$rank" "$@" \
            > "$attempt/arms-gpu$rank.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    bash "$here/arms.sh" summary --attempt "$attempt" "$@"
    exit
fi
[[ $# -ge 4 && $(hostname) == job-*-master-0 && -n ${KUBERNETES_SERVICE_HOST:-} ]]
pin=$1 id=$2 cards=$3 seconds=$4
shift 4
[[ $pin =~ ^[a-f0-9]{40}$ && $id =~ ^[[:alnum:]_-]+$ && $cards =~ ^(2|4|5)$ ]]
[[ $seconds =~ ^[0-9]+$ && $seconds -gt 30 && $(git -C "$code" rev-parse HEAD) == "$pin" ]]
export SYMMETA_EXPECTED_CODE=$pin REFERENCE_STARTED=$(date +%s) REFERENCE_PHASE=symmeta REFERENCE_CODE_PIN=$pin
attempt=$root/decompose-native/attempts/$id
mkdir -p "$(dirname "$attempt")"
mkdir "$attempt"
"$root/env/bin/python" -B "$here/../reground-rotation-reference-20261123-r1/reference.py" reserve --attempt "$attempt" --gpus "$cards" \
    --wall "$seconds" --cap "${DECOMPOSE_CAP_GPUH:-14}"
finish() {
    rc=$? end=$(date +%s)
    printf '{"exit_code":%d,"end_utc":%d,"allocated_gpu_seconds":%d}\n' \
        "$rc" "$end" "$(( (end - REFERENCE_STARTED) * cards ))" > "$attempt/exit.tmp"
    mv "$attempt/exit.tmp" "$attempt/exit.json"
}
trap finish EXIT
remaining=$((REFERENCE_STARTED + seconds - 15 - $(date +%s)))
(( remaining > 0 ))
timeout --kill-after=10s "${remaining}s" bash "$here/run.sh" _lanes "$attempt" "$cards" "$@" \
    > "$attempt/launch.log" 2>&1
