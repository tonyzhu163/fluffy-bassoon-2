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

`source_corpus_root` is a separate append-only canonical phase directory under
the shared output root, outside immutable per-job outputs. Reopening it must
retain the exact contract and runtime (including actual recorded renderer
binding); no runtime or old record is rewritten to accommodate a new host/GPU.
Only new episode/chunk files and the existing corpus inspection/counter views
are appended/updated. This keeps a resumed12→24 lineage in one original corpus
instead of inventing a copied or concatenated replacement bank. Source resume
checks this root against the actual captured boundary before model/reset.

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
Both validation cases receive whole-panel action/time admission before either
reset, so a healthy operational boundary cannot silently repeat a paid case.
Milestones12/24 set the common runner's `expected_reset` to baseline0's actual
accepted seed/observation/features hashes and `paired_instruction` to its exact
`reset_info.instruction` list. The common backend binds that actual instruction
before encoding and recording for the entire episode; this entry point does
not rewrite text/features or weaken any initial-input equality check. The
diagnosed common paired-instruction fix must be included and qualified in the
deployed composite before these later source-validation milestones can run.

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
The saved manifest and its hash-verified global checkpoint both bind the exact
canonical ledger root/header, journal SHA tip and event count. Equal action
totals from another journal are insufficient. Before opening the ledger, resume
checks the saved tip/count; under its existing lock, it permits only the one
fresh run-registration event and unchanged physical counters. Any intervening
append or root/tip/count mismatch blocks before model/reset execution.

The optional `after_source` callable in the orchestration API is reserved for
the existing control owner's separately pinned24-case A2 helper; no duplicate
cross-condition implementation or second CLI is added here. Initial source
collection does not depend on an unfinished mean-teacher diagnostic.
The owner ticket's `run_condition_cross=true` enables the small binding adapter:
two trained TRAIN1 actors, one fixed blind alias selected by greatest final
source-validation successes/2 with minusTRAIN1-first ties, original validation
receipts, immutable full controller references plus pure acting hashes and
sampled_gaussian mode. It calls the control owner's `run_condition_cross` with
the common episode wrapper and all actual source/validation reset identities.
Updates0 is not silently replaced with an untrained neural actor for A2.
The helper owns the exact24-grid, paired RNG/reset checks and measured.25
within-condition mismatch and.125 aggregate blind criteria. No mean-mode or
additional episode is introduced. A clean closed A2 boundary is adopted only
from its actual helper checkpoint plus boundary JSON, complete committed IDs,
exact saved ledger charges and source-global checkpoint. It forwards that exact
state to the helper's grid-prefix validator. Failed or partial episodes cannot
resume. The helper's explicit pre-pending episode admission is not repeated
after saving its pending cursor, avoiding a second deadline rejection race.

The adapter consumes the student's ORIGINAL balanced24 `blind_selection`
receipt and exact prefix final-controller references; it does not derive a
second selector. Missing/failed A2 evidence keeps both AD/DPT scientific
admission pending. After the actual helper completes and passes, source calls
the same builder with the original six publications, original A2 rich reference
as `condition_cross=...`, and a NEW `balanced-024-with-cross.json`. Previously
published12/24 pending data and their actual qualification references are never
overwritten. Only the24 prefix has this one funded grid; no extra12-case grid
or early scientific fit is implied. The student verifier independently reads
the original plan/raw24 evidence and recomputes contrasts and policy/mode
bindings before allowing an objective's fit.

An explicitly owner-admitted later allocation may use `--resume` for a complete
24-source prefix whose A2 stage was deferred, or for the actual closed A2 cursor.
The same exact code/profile/corpus/journal/teacher recoveries are required;
completed source episodes are skipped and original pending prefixes reused.
No completed/failed full grid is automatically repeated.

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
