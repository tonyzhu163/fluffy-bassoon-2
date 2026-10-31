# Closed teacher00/01/02 replay diagnosis

Latest main863175fba7982239294a3539ad94a8d8420ade91 was reviewed and incorporated
before freezing this component. Reused demo-reader/capture-helper bytes are
identical to the actual8fb producer. Ended source/data remain immutable.

One necessary read-only CPU slice, at most300worker-seconds inside the existing
1800phase CPU ceiling (746used/1054remaining at admission preparation). The parent
alone schedules, observes and settles cost. Include the12focused synthetic tests,
setup, full hash/CRC checks, diagnosis, small evidence and cleanup in one original
absolute clock. No new grant, model, simulator, VLA query, robot action or SGD.
No GPU retry, demonstration exclusion, two-demo fit or executor change follows
automatically. All original8TRAIN/4VAL cases and failed02 remain fixed.

The immutable producer is8fb991c8d5a3b9f86a81e2a4f2cf55b185cec81f,
job-471ns3j0v2n4: motor/zero/parity/gradient gates passed; teacher00/01 succeed104
actions each and02 times out400. This is a replay/data compatibility failure,
not a trained reactive-class result. Existing1008newactions/266GPU-seconds stay
charged, with full old recovery and raw corpus protected. No cache or fit exists.

## Explicit inputs and outputs

- Result: /mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/capacity-r1/experiment/result.json,
  SHAe9035c734ff5f7aa688d93e5a476794e452f1b19c2dedc813e5af262f0acd05b.
- Accepted demos: /mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/admission/capacity-r1/demos.json,
  SHAa80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4.
- Explicit source-evidence input: SOURCE.json in this directory,4764bytes,
  SHAb20b5be52f1ac40506f3aa97f13c45b71a5a53faf5837150df7d0e89c509e8c5.
  It is copied verbatim from the parent's saved source review, includes all3
  closed reset.json SHA references and exact pinned compression/downsample text.
  Supply PATH/SHA explicitly and preflight its presence; no hidden fixture path.
- Whole HDF5 files and all corpus manifests/chunks/NPZ/frame blobs come only
  from those bound records. No pickle, weights, recovery tensor or live ledger
  is loaded. Root/path/ordered-frame/command identity and file/typed-array SHA,
  ZIP CRC, actual mask and duration checks fail closed on inconsistent artifacts.
- Fresh output: a new inspection subdirectory under the parent's new CPU output.
  Twelve small paired native-RGB PNG panels (reset/reference-mid/actual-mid/final,
  all3views), JSON and one608-valid-command CSV. ≤16MiB output, ≤256source files,
  each≤100MiB/aggregate≤1GiB read bound, no giant transfers or source mutation.

Every actual valid issued command is compared against the original physical14D
HDF5 target indexed by cumulative ACTUAL commands, with declared final-target
hold. Unexecuted issued padding is reported separately. Every chunk's next
public drive target is compared with its last valid request after native gripper
clipping; report left/right arm/gripper errors and terminal-partial-command
qualification. This is not proof of measured joint/object poses or dense physics
delivery. Native wall duration is not simulation/TOPP time.

Report separate adjacent-reference jump maxima and nominal-gripper transition
versus stationary-arm ranges using source allclose atol1e-5/rtol0. Reconstruct
only deterministic compress_path retained indices from actual initial drive
targets plus issued requests; do not call TOPP or draw jitter. The pinned native
executor compresses/re-times arm paths and distributes gripper timing over the
original chunk slots. This can alter arm-hold/close synchronization relative to
author dense execution/save_freq15/boundary pictures; it is a mechanism to check,
not an established cause of02. Stationary-path Gaussian jitter has std2e-4, not
a hard tolerance; private motor_rng covers VLA Gaussian noise only, not native
global-NumPy planner randomness. Small end-drive residuals may include that
jitter. No new RNG framework or source/runtime fix is part of this diagnosis.

Source camera matrices/shapes, actual seeds/language/drive state and paired
images are reported. The actual closed resets lack merged task_config/get_info,
native camera matrices, measured joints and actor poses; mark unavailable.
Equal seeds/initial drive targets or similar pixels cannot prove full scene
identity. Expert commands are independent of visual policy features, so image
differences diagnose scene/capture compatibility rather than vision-driven
action selection. The report always retains scene_identity_proven=false and
automatic_retry_authorized=false. audit_complete means inspected, not teacher
competence, fitting success, or capacity qualification.

## Parent execution contract

Reuse the existing qualified CPU allocation/proof/clock/owned-cleanup mechanism;
this component adds no wrapper, dispatcher, scheduler, PID controls or observer.
The parent obtains genuine0GPU/2CPU/16GiB attestation before model/test imports
and supplies one ticket binding source, host, exact output, original start and
deadline, maximum worker-seconds≤300 and prior phase cost≥746/ceiling1800.
The same deadline starts before setup/tests and is passed through unchanged.

Additional ticket fields expected by this inspector:

```json
{
  "phase_id": "frozen-pi05-endtoend-20261031-r1",
  "gpu_count": 0,
  "hostname": "ACTUAL_CPU_HOST",
  "source_commit": "EXACT_INSPECTOR_PIN",
  "maximum_cpu_worker_seconds": 300,
  "prior_phase_cpu_charge": 746,
  "maximum_phase_cpu_worker_seconds": 1800,
  "inspection_output": "/ABS/FRESH_CPU_OUTPUT/inspection",
  "output": "/ABS/FRESH_CPU_OUTPUT",
  "zero_gpu_attestation_sha256": "ACTUAL_PROOF_SHA",
  "started_unix": 0,
  "deadline_unix": 300
}
```

The last two values are placeholders for actual epoch times. Deadline≤start+300,
and must still include all setup/tests/inspection; no reset after a failure.
The attestation requires passed=true, matching hostname, actual CPU/RAM values,
device_count_before_mask=0 and cuda_initialized_before_mask=false.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/test_robotwin_replay_diagnosis.py --junitxml=replay.xml
python scripts/diagnostics/robotwin_replay_diagnosis.py \
  --result /ABS/capacity-r1/experiment/result.json --result-sha256 e9035c734ff5f7aa688d93e5a476794e452f1b19c2dedc813e5af262f0acd05b \
  --demos /ABS/admission/capacity-r1/demos.json --demos-sha256 a80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4 \
  --source-evidence /ABS/CODE/outputs/robotwin-replay-diagnosis-20261031-r1/SOURCE.json \
  --source-evidence-sha256 b20b5be52f1ac40506f3aa97f13c45b71a5a53faf5837150df7d0e89c509e8c5 \
  --allocation /ABS/CPU_TICKET.json --allocation-sha256 SHA \
  --zero-gpu-proof /ABS/zero-gpu-attestation.json --zero-gpu-proof-sha256 SHA \
  --source-commit EXACT_INSPECTOR_PIN --deadline-unix ORIGINAL_SHARED_DEADLINE \
  --output /ABS/FRESH_CPU_OUTPUT/inspection
```

12prepared test cases use tiny synthetic captures through the existing recorder,
matching the three retained outcome/count shapes with608synthetic slots; these
are not actual archived payloads or physical executions. No test,
array computation or image decode has run on macOS. Actual data conclusions wait
for the parent's one allocated Linux slice and terminal cost/cleanup evidence.
