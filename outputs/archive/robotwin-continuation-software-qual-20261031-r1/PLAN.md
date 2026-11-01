# Joint CPU software qualification — pinned for owner review, not submitted

One fresh Baige **0-GPU, 4-CPU, 16-GiB, 600-second** allocation can run these two
independent software phases. No allocation, deployment, model/runtime
qualification, source action, source-process control or source reservation is
created by this preparation. The production owner has supplied the corrected
pin; root independently reviews, deploys and launches this auxiliary artifact.

## Bound inputs and remaining root actions

1. Production is bound to corrected owner commit
   `0c16c2b794f8de5b78603206917e72195083c265` at
   `/mowenjie/fluffy-bassoon-2/worktrees/robotwin-trained-continuation-code-20261031-r2`.
   The six selected files cover the continuous runner, production-mode
   preflight flag, unchanged one-shot defaults and three actual owned-fake-process
   regressions for durable spawn registration before deferred termination.
   The older reference
   `4e9ddf89a02ae2d90b37a305c55cc91781fa8c11` remains discovery evidence only.
2. LIBERO is bound to sole-owner commit
   `26b55eae2a5ce6545cdc97b3a875cc6e3a7b46fa` at
   `/mowenjie/fluffy-bassoon-2/worktrees/libero-cached-prefix-b1-code-20261031-r1`.
3. Reservation status is `ready_pending_owner_review_not_submitted`.
   Root independently reviews this artifact, reviews the latest `main` before
   launch, records the reviewed base and deploys the exact artifacts/candidates
   through Git into clean detached worktrees. The explicit launcher invocation
   is root's post-review action; no status edit or automated submission follows
   from this commit. Active source worktrees are not reused or modified. No
   test/model computation runs on development SSH.
4. Supply the resulting exact root launcher commit as `--launcher-pin` and the
   actual CPU job hostname captured at startup as `--expected-hostname`. Root
   compares the recorded hostname with the allocated scheduler job ID before
   accepting the receipt. The launcher pin is
   an explicit post-commit argument, avoiding a self-referential commit hash
   inside its own reservation. Keep the output identity fresh; no automatic
   retry is configured.

Both source pins and the launcher must be exact and clean before execution and
again after their relevant tests. Unresolved pins or a different reservation
status are rejected before the Torch allocation receipt or test phases. Root
owns the final review and launch; R4 source work is independent of this gate.

## Exact test selections

Phase1 runs in one fresh Python process from the corrected production root:

```text
tests/test_robotwin_trained_pair_continuation.py
tests/test_robotwin_trained_pair_plan.py
tests/test_robotwin_trained_pair_preflight.py
tests/test_robotwin_trained_pair_runtime.py
tests/test_robotwin_trained_pair_audit.py
tests/test_robotwin_trained_pair_continuation_spawn.py
```

Static AST expansion at final0c16c2b finds67 production cases: continuation14,
plan17, preflight16, runtime14, audit3 and spawn3. The same four paired files now contain50
cases because the owner added three preflight regressions. Older reference
4e9ddf8 contained8 continuation plus47 paired cases. No new owner regressions
are excluded to preserve the older count. The gate records actual JUnit counts
rather than fixing acceptance at67. Every selected file must contribute
nonzero passing cases; `FINAL-SOURCE-DISCOVERY.json` records the final counts.

Fixture dependencies `tests/test_robotwin_trial_resume.py` and
`tests/test_robotwin_trained_parallel.py` must exist at the production pin.
Their named fixtures are imported by the selected tests; those files are
**not independently collected or retested**. The selected suites use tiny
synthetic CPU learners, temporary journals and owned temporary fake processes.
They do not receive production process identities, source ledger paths or
robot/model inputs.

The spawn regression file launches real temporary fake child/grandchild
processes inside the CPU allocation. It exercises deferred SIGTERM/SIGINT and
durable registration/callback failure cleanup only for identities it created;
it neither controls source jobs nor runs a model, simulator or robot.

Phase2 runs in a separate fresh Python process from the sole LIBERO owner root:

```text
tests/test_vla_cached_scalar_motor.py
```

Static AST expansion at26b55ea finds43 prepared CPU fake cases. The injected
backend/native oracle does not load model weights; its test fixture explicitly
rejects JAX, JAXLIB, OpenPI and OpenPI-client imports. This is not an actual
LIBERO model, robot, numerical-parity or throughput qualification.

The unchanged c62 and c166 suites are not CLI selections. No broad test
directory or model/robot/GPU tests are collected. `discovery.json` records the
static paths, source-content identities, imports and expanded reference counts.
Discovery reads source through Git and does not execute the test modules.

## Runtime and dependencies

Reuse the existing environment without installation or cache migration:

```text
/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
```

The CPU software paths require Python's standard library, pytest, NumPy,
PyTorch and gymnasium, plus Linux `/proc`, process sessions and the qualified
fake-process primitives. Existing source fixture/helper modules and these
production scripts must be present at the final production pin:

```text
scripts/diagnostics/robotwin_trained_pair.py
scripts/diagnostics/robotwin_trained_pair_preflight.py
scripts/diagnostics/robotwin_trained_pair_runtime.py
scripts/diagnostics/robotwin_trained_pair_audit.py
scripts/diagnostics/robotwin_trained_pair_continuation.py
```

Phase2 itself needs only pytest, NumPy and its candidate module plus standard
library for the injected fakes. Actual LIBERO/OpenPI/JAX/robot assets or model
weights are not required and must not be added to this gate. No dependency
setup, download or source-control process is part of the allocation.

## Admission, execution and acceptance

`job.sh` refuses macOS, a hostname different from the explicit `job-*` binding,
an unresolved launcher pin or an existing output. `qualify.py` validates the
reviewed reservation, exact clean Git identities and selected test files.

A separate Python process then writes `actual-cpu-allocation.json`: actual
hostname, existing Python/Torch versions, incoming CUDA/NVIDIA visibility,
CUDA device count/availability/initialization, CPU affinity and cgroup CPU/RAM
metadata. It requires count0, unavailable CUDA and no initialized CUDA context.
The incoming visibility is retained for this admission; only after it passes
does the coordinator set `CUDA_VISIBLE_DEVICES=''` for the test phases.

Each phase has its own `PYTHONPATH` containing only that phase's `src`, `tests`
and `scripts/diagnostics`, its own Python process, raw `pytest.log`, intent,
JUnit XML and parsed receipt. Plugin autoload and pytest addopts are disabled,
bytecode/user-site writes are disabled, and pytest's cache provider is disabled.
Temporary fake artifacts stay under the fresh phase output. Worktrees must
remain clean. An ordinary Phase1 test failure does not suppress the independent
Phase2 attempt while the external wall envelope remains available.

Qualification passes only when both Python test commands exit0, every selected
file has nonzero JUnit cases, every case passes and failures/errors/skips/xfails
are all0. Strict xfail handling and explicit JUnit inspection prevent skips or
expected failures from being reported as a passing gate. Actual counts are
saved. No claim of native model/robot qualification follows from a CPU pass.

The external GNU timeout covers admission and both phases: **560seconds plus
at most10seconds termination grace**, leaving30seconds under the600-second
scheduler ceiling for launcher/final receipts. Partial logs or absent/partial
JUnit after timeout remain failed/incomplete evidence. `exit-code.txt` and
`launch-outcome.json` record the external result; there is no retry or rollover.
No previously unused GPU/action qualification capacity is reused.

## Root-owned launch template after final review/deployment

```bash
bash /mowenjie/fluffy-bassoon-2/worktrees/robotwin-continuation-software-qual-20261031-r1-launch/outputs/robotwin-continuation-software-qual-20261031-r1/job.sh \
  --launcher-pin "$ROOT_REVIEWED_AUXILIARY_COMMIT" \
  --expected-hostname "$(hostname)" \
  --output /mowenjie/fluffy-bassoon-2/outputs/robotwin-continuation-software-qual-20261031-r1
```

This is a root launch template, not an executable submission request. Both
source pins are supplied by their owners. The only local verification is
standard-library JSON/AST inspection and `bash -n`; no test collection, tests,
model construction, forward, allocation, deployment or process signals run on
macOS.
