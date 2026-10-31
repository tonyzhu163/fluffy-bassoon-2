# Fixed500 two-goal robot comparison

Status: prepared but not launched/reserved. Superseded by the common-runtime
comparison of both source banks and controls in one frozen motor process;
see the current [campaign plan](../../outputs/libero40-source-1m-20261029-r1/PLAN.md).

Use the completed Spatial00 + LIBERO10:00 source assembly and its four
fixed500 AD/DPT full/static candidates. Evaluate each objective's full and
trained static controllers plus an episode-reset intervention using exactly
the full controller's weights. The target cases are the original predeclared
LIBERO90 task0 and task1, nominal state0, scene seed123. The case dictionaries
are reused from the one-goal comparison, so role/query seeds stay identical.

Each of the12 method/case groups executes an independent zero-support query
and a real one-support episode followed by a separate fresh-reset query.
This is36 actual episodes, with a14,760-action worst case at400 control steps
plus10 settling actions per episode. The joint ceiling is16,000 actions,
including all settling, partial attempts and unresolved RPCs. A group is not
started without its three-episode reservation. No SGD, native baseline or SAC
actor is run. The source trainer's singular Spatial00 SAC compatibility field
is ignored; this comparison does not construct or imply a joint SAC model.

The parent funds the16k slice from future unlaunched source PPO allocations
before launch, retaining the1M benchmark ceiling. After allocation GPU/tests
pass, the launcher creates the exact-run/source escrow in the shared PPO
ledger; the evaluator itself only reads and verifies it. A preflight failure
therefore collects no actions and creates no action reservation. Full40
collection approval remains separate.

Run only after the two-goal scale manifest is completed and its file hash has
been recorded. In a clean commit-pinned worktree inside one actual allocated
GPU with a25-minute limit:

```bash
bash scripts/diagnostics/vla_source40_two_goal_evaluate.sh \
  EXACT_REVIEWED_SOURCE FRESH_ROBOT_RUN \
  libero40-two-goal-scales-20261030-r1 COMPLETED_SCALES_FILE_SHA256
```

The prepared launcher runs the allocation GPU probe and focused Linux tests,
then allows1320 seconds in the evaluator and a1380-second process timeout.
Qualification uses the first actual query reset, with no additional physical
reset. Append `--resume` only to the same source, run, source manifest and
16k escrow. Completed groups are verified and skipped. Partial groups resume
the existing episode-boundary controller recovery; previously executed and
uncertain physical actions remain charged.

The run binds the scale manifest, fixed500 matrix, assembly file/content
hashes, full source-exposure hash, source costs and reservation. Complete
episode traces retain both public observations, proposed/executed noise,
decoded actions, raw/shaped rewards, success and boundaries. Before each
physical action RPC, raw inputs/features/noise/decoded actions are archived;
reset and actual outcome receipts are archived separately. Every complete
episode retains controller context before/after, generator and global RNG,
motor RNG and its trial/episode cursor. The existing evaluation recovery
retains authoritative next-episode context, sampler and journal costs.

The evaluator verifies nominal-state identity, actual initial RGB/wrist/state
array equality and paired query RNG within each group, then verifies those
initial arrays across all six methods on that case. Full and episode-reset
methods share the same policy object and file hash. The final result reports
all groups, exact charged actions and their trace/context checksums. Compare
these robot outcomes with the original one-goal fixed500 results while
reporting source data and interaction costs separately. Two development goals
and two target cases do not establish broad transfer or reliable superiority.

Equal case/RNG identifiers alone do not establish numerical identity of the
motor across separate allocations. The completed target-online DSRL run has
matching initial arrays/noise but first-action deltas up to0.00415 versus the
earlier native controls. A zero-action motor replay diagnostic is measuring
that confound. Do not call cross-run comparisons exactly paired without the
corresponding numerical evidence; within-run same-weight memory lesions
remain separately auditable.
