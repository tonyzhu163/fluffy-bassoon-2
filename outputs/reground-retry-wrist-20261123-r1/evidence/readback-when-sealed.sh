#!/usr/bin/env bash
# Baige CPU. Wait for retry01 and run its documented full readback on the producing checkout.
set -euo pipefail
run_root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
store="$run_root/scored"
attempt="$store/attempts/retry01"
shopt -s nullglob
while :; do
    batches=("$store"/batches/batch[0-9][0-9][0-9]/producer.json)
    if [[ -s "$attempt/exit.txt" ]]; then
        if (( ${#batches[@]} < 70 )); then
            echo "Terminal launcher receipt with ${#batches[@]}/70 batches" >&2
            exit 3
        fi
        break
    fi
    if (( ${#batches[@]} == 70 )); then
        # Each directory is committed before its rank exits. Give accounting time to publish.
        for ((i=0; i<6; i++)); do
            [[ -s "$attempt/exit.txt" ]] && break
            sleep 5
        done
        break
    fi
    sleep 15
done
pin=$(python3 - "$store" <<'PY'
from pathlib import Path
import json,sys
store = Path(sys.argv[1])
pin, host, phase, cards, start, wall = (store/"attempts/retry01/launch.txt").read_text().split()
assert pin in {"0bfa5b087a2b5d5ba9d6ecebafd9db5288947186","ac28987f221e336cef2ea64dc16dfffc9f0e74c9"}
assert phase == "full" and cards == "4"
expected = {f"batch{i:03d}" for i in range(70)}
assert {p.name for p in (store/"batches").glob("batch*")} == expected
for name in sorted(expected):
    path = store/"batches"/name
    contract = json.loads((path/"contract.json").read_text())
    producer = json.loads((path/"producer.json").read_text())
    assert contract["code"] == producer["code"] == pin, name
    assert producer["attempt"] == "retry01" and producer["host"] == host, name
print(pin)
PY
)
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
bash "$run_root/src-$pin/readback.sh" "$pin" full
python3 - "$store" "$pin" <<'PY'
from pathlib import Path
import json,sys
dest = Path(sys.argv[1])/"readback-full"
summary = json.loads((dest/"summary.json").read_text())
cases = json.loads((dest/"cases.json").read_text())
assert summary["code"] == sys.argv[2]
assert summary["new_pairs"] == len(cases) == 280
assert len({(c["task"],c["init_id"],c["rep"]) for c in cases}) == 280
assert summary["new_seed_labels"] == [0] and summary["original_rows_per_arm"] == 1400
assert summary["distinct_support_layouts"] == 120
assert summary["paired"]["success"]["all"]["n_unique_starts"] == 280
for goal in ("goal_1","goal_2"):
    assert summary["paired"]["success"][goal]["n_unique_starts"] == 35
print(f"Verified full readback and 280 unique rows at {dest}")
PY
