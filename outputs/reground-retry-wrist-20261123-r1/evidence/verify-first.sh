#!/usr/bin/env bash
# Run on Baige CPU after the first atomic publication. Emit a JSON receipt to stdout.
set -euo pipefail
run_root=/mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1
attempt_id=${1:-retry01}
[[ $# -le 1 && $attempt_id =~ ^retry[0-9]+$ ]]
native_attempt="$run_root/scored/attempts/$attempt_id"
read -r RETRY_PIN launch_host phase allocated start wall < "$native_attempt/launch.txt"
case "$RETRY_PIN" in
    0bfa5b087a2b5d5ba9d6ecebafd9db5288947186|ac28987f221e336cef2ea64dc16dfffc9f0e74c9) ;;
    *) echo "Unexpected producing pin $RETRY_PIN" >&2; exit 1 ;;
esac
export RETRY_PIN
here="$run_root/src-$RETRY_PIN"
attempt="$run_root/evidence/first-batch-cpu-$attempt_id"
mkdir -p "$attempt"
source "$here/env.sh"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
"$py" -Bu - "$native_attempt" "$launch_host" "$allocated" <<'PY'
import contextlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

attempt, host, allocated = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with contextlib.redirect_stdout(sys.stderr):
    import reference as ref
    ref.configure()
    import retry
    import summarize
    from reground.eval.jobs import load_traces
    import numpy as np

    admission = ref.read(attempt / "gpu.json")
    devices = admission["devices"]
    assert allocated == len(devices) == 4
    assert admission["status"] == "passed" and admission["exclusive_training_admission"]
    assert admission["allocation_type"] == "baige_container" and admission["hostname"] == host
    assert all(d["cuda_name"] == "NVIDIA GeForce RTX 4090" for d in devices)
    assert all(len({d[k] for d in devices}) == 4 for k in ("cuda_uuid","pci_bus_id","allocation_ordinal"))
    store = attempt.parent.parent
    published = [(ref.read(p), p.parent) for p in (store/"batches").glob("batch[0-9][0-9][0-9]/producer.json")]
    producer, path = min(published, key=lambda item: item[0]["ended"])
    assert producer["attempt"] == attempt.name and producer["host"] == host
    assert producer["code"] == os.environ["RETRY_PIN"]
    rank = producer["rank"]
    receipt = ref.read(attempt / f"rank{rank}.json")
    index = int(path.name[5:])
    assert index in receipt["batches"] and receipt["device"] == devices[rank] and receipt["egl"] == producer["egl"]
    keys = retry.schedule()[index]
    retry.validate(path, keys)
    contract = ref.read(path / "contract.json")
    reports = {arm:ref.read(path / (arm + ".json")) for arm in retry.ARMS}
    traces = [load_traces(path / arm) for arm in retry.ARMS]
    changes = []
    for key in keys:
        left, right = [t[key]["A"][:10] for t in traces]
        n = min(len(left),len(right))
        assert n > 0
        np.testing.assert_array_equal(left[:n,[0,1,2,5,6]], right[:n,[0,1,2,5,6]])
        changes.append(float(np.abs(left[:n,3:5]-right[:n,3:5]).max()))
    support = ref.read(path/"support.json")
    receipt = dict(job=host.removesuffix("-master-0"), attempt=attempt.name, code=contract["code"], batch=path.name,
        checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        validator="retry.validate with native horizon and released policy defaults", producer=producer,
        upstream=contract["upstream"], policy=contract["policy"], reader_weights=contract["reader"]["weights_sha256"],
        admission=dict(host=host,status=admission["status"],exclusive=True,devices=devices),
        query_keys=keys, support_keys=[s["support"] for s in support],
        opening_sha256=reports[retry.ARMS[0]]["opening_sha256"],
        max_opening_wrist_xy_change_per_lane=changes,
        joined_opening_translation_angular_z_gripper_equal=True,
        arms={arm:[{k:r[k] for k in ("task","init_id","steps","success","unsafe")} for r in data["rows"]]
              for arm,data in reports.items()},
        native_costs=summarize.journals(path.glob("*.jsonl")))
print(json.dumps(receipt, indent=2))
PY
