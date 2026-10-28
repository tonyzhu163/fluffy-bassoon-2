# Native DPT warm-start recovery

Completed with exit zero on 2026-10-28 at 18:46:56 Asia/Shanghai. See
[RESULTS.md](RESULTS.md) for the audited candidate, frozen live diagnostics,
one-step PPO transition and limitations. The launch contract below is preserved.

This run executes only the already approved native stages of the
[original pilot](../ctm-dpt-warmstart-20261028-r1/PLAN.md). Seven compact
candidates from r1 are retained and are not retrained. The recovery corrects
the decoded action horizon mismatch documented in the
[r1 audit](../ctm-dpt-warmstart-20261028-r1/RESULTS.md).

- Baige job `job-l3coajmm9noh`, name `mwj_1028_ctm_dpt_native2_g1_79da195`.
- Submitted 2026-10-28 18:12:38 Asia/Shanghai; running since 18:12:55.
- One RPBZZZ6 GPU, 16 CPU cores, 64 GiB RAM, 8 GiB shared memory, 50-minute cap.
- Reviewed main/source: `79da19500f4e898d078539b4c562c150e610c16a`.
- Worktree: `/mowenjie/fluffy-bassoon-2/worktrees/ctm-dpt-warmstart-20261028-r2`.
- Output: `/mowenjie/fluffy-bassoon-2/outputs/ctm-dpt-warmstart-20261028-r2`.

The launch command is the pinned worktree's
`outputs/ctm-dpt-warmstart-20261028-r1/launch.sh`, with arguments
`79da19500f4e898d078539b4c562c150e610c16a ctm-dpt-warmstart-20261028-r2 native`.
The launcher validates its clean source, scheduler container and allocated GPU.
It repeats relevant tests before model work: **67 passed, 9 skipped, 16.79s**
in this allocation. The skips are Slurm-only tests; the actual compact CLI
pretraining/resume/PPO transition is included among passing tests.

Stages remain two complete native history batches, a paired frozen-motor versus
inverse-rotation label-policy competence gate, then at most 100 flow-matching
updates/300 seconds if that gate passes. Frozen candidate normal/wiped/blind
evaluation at iteration zero precedes one actual PPO update. Native artifacts
have their own compatibility family and cannot load compact checkpoints.

At the first artifact check (18:18 Shanghai), history batch one was saved with
two four-attempt trials, 640 simulator steps, 128 policy chunks and 128 extra
label-motor calls. Collection-only optimizer steps remain zero. Its manifest
records the source/motor/interface and SHA256
`af401a1a648704efdcf40037cff78a29196625da4c5dfa06dcea4596b076aa51`.
Training-history success (.625) is collector behavior, not learned CTM behavior
or the pending paired label-competence result.

The 18:23 artifact check found both complete history batches (1,280 training
simulator steps total) and no failure receipt. Batch two was saved at 18:20:44,
SHA256 `f1a9c2ca926e30589ac38bb2a4bcf03f872b4ace4607dd2b5349e76bc0e663c0`.
The collector then performs its four paired trials per policy before writing
the iteration-two metrics and gate result. The timestamped evidence snapshot
contains the two-batch manifest, qualification, allocation/renderer binding,
configuration and first metric; it does not imply the pending gate has passed.

The realized trial plans expose a substantial pilot limitation: batch one,
which will train the CTM, has rotations `[45, 0]`; reserved validation batch two
has `[0, 0]`. Neither contains a -45-degree trial. A lower validation loss here
therefore cannot demonstrate learned correction across the declared training
support. Keep this tiny execution screen separate from any claim of a useful
robot adaptation initializer; the later live behavior check remains necessary.

Live status remains in the single canonical `outputs/coordination.md` referenced
by this repository. The existing 20-minute heartbeat monitors this run and
will report a meaningful change, failure or completion.
