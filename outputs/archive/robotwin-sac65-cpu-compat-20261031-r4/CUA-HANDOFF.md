# r4 owner handoff: fixed local runtime and same complete proof capture

Prepare an unused run/output/input basename
`robotwin-sac65-cpu-compat-20261031-r4` after exact-pin review. Preserve r1–r3.
Fresh form:0GPU /8CPU /64GiB /shm2 /**19minutes**, autofault OFF. Stage the same
three exact ownership helpers in r4 `owned-primitives`.

The five-argument scheduler command is:

```text
bash <fresh-deployed-code>/outputs/robotwin-sac65-cpu-compat-20261031-r4/job.sh \
  <fresh-deployed-code> <reviewed-full-pin> \
  /mowenjie/fluffy-bassoon-2/outputs/robotwin-sac65-cpu-compat-20261031-r4 \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r4/actual-bootstrap.json \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r4/owned-primitives
```

The parent keeps this original deployed path for durable input provenance. Its
owned30-second stage creates/validates an exact job-local Git checkout, without
changing the original or resetting the shared deadline. Model/tests/consumers
then execute from that clean local checkout; the owner retains both path records.

Install the unchanged functions from `capture_complete_proof.js` before arming.
Use the explicitly metadata-verified modern Mac interpreter for the bridge:

```text
/Users/mowenjie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -I -B -S <local-code>/outputs/robotwin-sac65-cpu-compat-20261031-r4/dispatch_bootstrap.py \
  --inbox <fresh-local-inbox> --code-commit <reviewed-full-pin> \
  --ssh-target <configured-target> --ssh-port <configured-port>
```

The ready receipt must show its live PID, Python3.12.14 executable/version,
exact pin/run/nonce, unexpired lifetime and task name
`mwj_1031_robotwin_sac65_cpu_r4_<first-seven-pin-characters>`. Do not submit if
the bridge has not armed under the approved runtime. Python<3.10 on macOS is
rejected before inbox creation; no automatic fallback to CLT3.9.

Keep Submit, fresh visible actual-task navigation and
`await publishCompleteCpuProof(...)` in the **same** CUA invocation. Set the
original pre-Submit55s deadline and hard tool timeout<=60s. Wait through missing
host/partial table/cells/text; complete wrong resources or extra roles reject.
Native PNG/JPEG bytes and AX are saved before the nonce marker. The bridge must
automatically publish the authentic, hash-bound ticket last. Record actual CUA
and dispatcher receipts; manual publication must not be called automatic success.

If proof or any phase fails, retain evidence and reconcile actual scheduler cost.
Do not mutate a live pin, repeat a pilot, reset a timer, resubmit automatically
or borrow any of the211sealed seconds.
