# Future wrapper guard patch — not used by the active panel

This isolated detached worktree starts at active wrapper
`2fd2096906e52aabeaeb984f333fe1aaa6c4e196`. The execution owner explicitly
keeps the healthy current job and its code immutable. This patch is preparation
for a later invocation only: no live file edit, allocation, migration, replay
or runtime process change occurs here.

The current run's outer auxiliary script hardpins the intended plan SHA and
the scheduler independently caps it at90minutes. Those facts are not a proof
that Baige immediately kills every process when a Python parent receives
SIGKILL. The future patch addresses the two wrapper-level follow-ups:

1. Reject any `--plan-sha` other than the compiled immutable `PLAN_SHA`, and
   check that constant again at the file-read boundary. Alternate100-update
   endpoint plans cannot be substituted with caller-computed checksums.
2. Pass expected parent PID and `/proc` starttime to each K8 child. Before
   model loading, Git checks, CUDA, renderer or simulator setup, check its
   actual PPID/start identity, arm Linux `PR_SET_PDEATHSIG=SIGKILL`, verify
   `PR_GET_PDEATHSIG`, then recheck PPID/start identity. A dead/reused parent,
   unavailable identity, blocked prctl or incorrect readback fails closed.

The kernel guard is installed in the already exec'd child rather than a
parent's `preexec_fn`. The parent launches physical workers from its serial
main thread, which is the thread whose death Linux tracks. A parent death
before arming is caught by the post-check; after arming the signal stops the
independent-session physical worker. This does not prove entire-allocation
or all-helper teardown. Existing normal owned PGID/SID cleanup remains, and
the scheduler ceiling and actual cumulative GPU accounting are still required.

Abrupt child death deliberately does not reconstruct physics or finalize a
fabricated outcome. Native action reservations/raw intents remain for
conservative accounting and owner diagnosis. No automatic retry is added.
Model, backend, adapter, cases, conditions, action cap and inference behavior
are unchanged. No source continuation implementation is included.

## Required allocated CPU validation before any future use

```bash
python -m pytest -q \
  tests/test_robotwin_parent_guard.py \
  tests/test_robotwin_history_live.py \
  tests/test_robotwin_history_evaluate.py
```

The future wrapper also includes these files in its mandatory startup gate.
Tests cover caller-checksum substitution, parent loss before arming, loss or
PID reuse during arming, wrong signal readback, parent-identity transport and
an actual Linux parent-SIGKILL scenario with a separate-session worker. That
test creates and kills ONLY its own CPU test processes, verifies identities
before fallback cleanup, and imports no simulator or motor. Its five-second
death deadline tests the kernel behavior instead of assuming container teardown.

Only static syntax, diff and source-identity checks were performed during this
preparation. No allocated tests or model computation ran here. The execution
owner chooses the later CPU qualification/allocation and any explicit rollout;
active `2fd2096` must not be modified or silently replaced by this future pin.
