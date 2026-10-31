# Executable source orchestration

The separately prepared source entry point is
`python -m fluffy_bassoon.meta_rl.robotwin_steering_source`; the matching thin
script is `scripts/diagnostics/robotwin_steering_source.py`. It runs only on
the sole owner's actual allocated Linux GPU. It neither schedules nor grants
resources. Existing collector `aad29a87` remains unchanged and has its separate
23/23 allocated CPU pass (job-wopjg7o5vte4, owner-reported79 worker-seconds).
The new orchestration tests have not been executed by this chat.

## Inputs and command

The owner supplies SHA-bound JSON files for `--scope`, `--allocation`,
`--interface`, `--joint-conventions`, `--selection`, and `--profile`, with a
matching `--<name>-sha256` for each. `--profile` is the actual `SourceProfile`
constructor fields, not the earlier unadmitted preparation template.

Other required arguments are `--output`, `--model-path`, `--norm-stats`,
`--rlinf-root`, `--robotwin-root`, `--gpu-probe`, `--source-commit`, `--run-id`.
The exact clean deployed commit and actual host/CUDA probe must match.
Motor identity is the safetensors **file** SHA; the frozen tensor fingerprint
from A1 is checked separately after loading and after collection.

The selection receipt has schema `frozen_motus_source_selection_v1`, status
`selected_from_completed_A1`, selection_basis
`natural_A1_predeclared_criteria`, condition_outcomes_used=false, and binds
task/arm/mapping/interface/motor/episode_limit/task_horizon hashes. It includes
`a1_result`, `qualification_receipt`, and `selection_rule` path/SHA references
plus `evidence_episode_ids`. The fixed rule SHA is
`87c8af53bcea7ee09ea55f11380b4a310d94e6ff2fc01e6f7b648b39299086b5`.
The program recomputes the ORIGINAL restricted summary: greatest strict
confirmed-rescue case count, fixed noise/KV/action ties, at least one2/2 winner
versus0/2 matched-native case. All contributing actual confirmation IDs and
paired initial identities must be cited. No eligible arm stops source admission;
it does not prove steering impossible.

The allocation ticket has schema `frozen_motus_source_allocation_v1` and binds
scope_sha256/profile_sha256/source_commit/hostname/job_id/gpu_count=1/output,
the existing phase_output_root and exact action_ledger header/root. It includes:

- reserved_gpu_seconds, actual prior_phase_gpu_seconds and
  other_reserved_phase_gpu_seconds, whose sum stays within28,800;
- allocation_started_unix, original driver_deadline_unix,
  maximum_episode_seconds, maximum_episode_update_seconds and
  cleanup_reserve_seconds>=60; no clock resets or idle wait for approval;
- max_full_recovery_bytes measured/bounded for the actual profile,
  recovery_output_reservation_bytes>=60 times that bound, and
  other_future_output_reservation_bytes under the same aggregate64GiB ceiling;
- CPU_qualification_receipt with actual status=passed and this exact source
  commit; the focused tests consume the already-funded aggregate1,800 CPU
  worker-seconds, not a new allowance;
- student_profiles mapping `12` and `24` to actual Transformer profile
  path/SHA references, and control_qualification referencing the student's
  binding of ORIGINAL A1 plan/interface/qualification/completed episode.

The entry point verifies the existing ledger genesis before opening the same
canonical `CampaignBudget`. It verifies native task horizons and source joint
limits, actual free space and an exclusive write/flush/fsync/unlink probe before
the first reset. It installs the two-sign condition once before canonical
corpus registration. The actual source mapping still has the original A1 hash.

## Behavior, publications and recovery

Six actual independent teachers use the existing `collect_round_robin` and
`run_complete_episode`. Validation restores full source state into a separate
actor/private stream. Updates0 uses bounded uniform even if observed chunks
already crossed32; learned validation samples the actual frozen actor. No
validation transition enters source replay or updates. Fixed case-specific
private innovations pair milestones without consuming source RNG.

At balanced12 and24, after ALL six TRAIN/VAL lineages have complete episodes,
completed update boundaries and fixed validation receipts, orchestration calls
the student-owned `steering_history_prefix.build_prefix`. TRAIN uses the actual
collector publications; VAL receives protected snapshots at the same boundary.
The deterministic builder reads original episode manifests/reset records and
full recoveries. No operator guesses case IDs, accepted seeds or observation
hashes. `completed=true` describes a complete data prefix; AD learning/DPT teacher
qualification remains pending unless separate actual evidence is supplied.
Validated input snapshots and published full wrapper hashes may differ; their
actual acting weights/updates/sampling mode require the student's explicit
identity binding, never a fabricated full-file alias.

`source-state.json` records actual six-lineage counts/optimizer cycles/steps,
all action charges, immutable prefix references and closed-boundary status.
It also binds an actual global Python/NumPy/Torch/CUDA/control recovery.
`--resume <source-state.json> --resume-sha256 <SHA>` accepts only same-code,
same-profile closed boundary stops, restores all captured global streams after
loading the motor, and restores each teacher's complete private state into new
recovery output directories. Partial physical or partial optimizer states stay
preserved for execution-owner reconciliation. No reset replaces partial work.

The optional `after_source` callable in the orchestration API is reserved for
the existing control owner's separately pinned24-case A2 helper; no duplicate
cross-condition implementation or second CLI is added here. Initial source
collection does not depend on an unfinished mean-teacher diagnostic.

## Focused allocated CPU gate

```
python -m pytest -q tests/test_robotwin_steering_source.py --junitxml=<fresh-owned-output>/source-entrypoint.junit.xml
```

This checks natural-A1 rescue eligibility/ties/refusals; actual uniform versus
learned frozen validation; existing actor archive use; global RNG restoration;
and a synthetic whole initial24 wiring exercise with the REAL bounded recovery
sink, all six TRAIN/VAL publications at12/24 and distinct validated/published
wrapper refs. The fixture's one-step early successes generate zero updates and
must not be relabeled as learning. It is not native simulation or teacher
competence. The student's own builder/raw-array integration has its own exact
source component and focused tests, inside the same remaining CPU allowance.
