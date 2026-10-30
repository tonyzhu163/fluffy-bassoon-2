# Native RoboTwin AD/DPT source-history exporter

`meta_rl.robotwin_history_corpus` reads `robotwin_public_decisions_v1` from
the existing three-worker source campaign. It does not change the collector,
load the frozen motor/simulator, or implement a supervised robot student.
Use an immutable source checkpoint and explicitly pinned teacher checkpoint.
Teacher competence, including deterministic mean-action quality, remains a
separate source-only measurement.

## Supported inputs

- Native2062 public features, full H32x32 physical noise bounded at1.5,
  and four native feedback/clock scalars. Source tasks are exactly the six
  declared RoboTwin tasks. Task-specific horizons are retained from each
  contract; older400-step data keep their separate contract.
- Source PPO, separately trained reset PPO, and SAC contexts. PPO action
  samples and saved3090-dimensional public inputs/reset masks are checked
  against real actions/rewards and recorded episode/primitive clocks.
- Source coordinator recoveries (`states[method]`) at the saved source
  boundary, or per-method source/stage recoveries (`learner`, `schedule`).
  Explicit source method selection is required. Target-adapted SAC or target
  query recoveries are rejected. Teacher/source feature/motor/horizon identity
  must match. Source PPO uses `gtrxl_noise_ppo` schema1; SAC uses its saved
  actor contract/config. Neither restoring optimizer state nor SGD is needed.
- Training selects only `phase=source` and trials covered by the committed source cursor.
  Every selected trial needs four ended episodes with distinct indices0–3.
  Duplicate/restarted ambiguous trials, unresolved/native-only chunks and
  external budget cuts fail. `--allow-legacy-cut` explicitly quarantines only
  an incomplete LAST trial in the older action-cut contract; never relabel it
  as K4 or discard a complete trial based on reward. Costs stay in its ledger.
- Explicit `--partition validation --validation-iteration N` instead selects
  the complete six-task source-validation panel at the matching immutable
  iteration-N source snapshot. Its acting weights must match that snapshot;
  no validation SAC updates are permitted. Export BC/DPT only for this fixed
  policy panel. Training and validation keep separate cases and costs.

The scientific runtime comparison ignores only `actual_render_pci`, which the
driver omits from checkpoint contracts. The complete raw runtime file remains
hash-pinned. Acting archive file SHA, payload parameter SHA and a recomputed
tensor digest must all agree. Native selected action durations must reconcile
exactly with frozen source/task counters; an explicitly quarantined legacy cut
remains included in that reconciliation and in the original physical ledger.

The source corpus can keep growing after the selected source checkpoint, but
selected files and both checkpoints must retain their hashes. Prefer the
owner's immutable iteration10/25/50/100 stage snapshots. A moving latest
checkpoint is not a fixed snapshot.

## Projected continuation roots

A continuation chain can be projected into a fresh flat view without changing
original records. A projection receipt binds the exact ordered source roots,
ancestor manifests/contracts/migrations, closed checkpoint and full-file
content hashes. Source and destination files retain their bytes; an unequal
relative-path collision cannot be resolved by choosing a newer record.

The projection-aware reader permits aliases only for recorded acting-policy
checkpoints, whose producer stores canonical absolute paths. Each alias must
name exactly an admitted original root's policies path and a contained
projected policies path with the same full SHA. Original lexical paths and
their ancestors cannot be symlinks, and unrelated identical-byte archives do
not become admitted origins. Actor kind, tensor/version/context identities
remain checked. General RGB/native/JSON path containment is unchanged.

The reader rechecks all metadata links and original archive bytes; provenance
is retained in the exporter input hashes. Stage/coordinator serialization may
differ only when actual source/task counters, trial/update counts, source
contract and actor tensors establish the same closed source state. A boolean
attestation does not replace those checks. This compatibility extension needs
its own exact-pin CPU qualification; the old single-root qualification does
not automatically qualify a composed bank. Labels/history/action accounting
and the supervised trainer remain the existing implementations.

## Label and history semantics

AD targets equal actual physical steering noise. The exported histories group
same-task source trials chronologically across actual optimizer/policy changes;
both update clocks and acting-parameter identities must vary. A fixed-policy
trial or a renamed clock alone is not AD. SAC warmup remains bounded uniform,
with no invented actor checkpoint; its first actor action may also have update0.

DPT-style labels use a source-trained SAC actor or recurrent PPO teacher.
The convention is deterministic physical `1.5*tanh(mean)`. Recurrent state is
rebuilt under the selected teacher weights from ACTUAL previous physical noise
and reward, independently at each original K4 source trial. Its own full versus
episode-reset rules determine teacher resets. Stale behavior caches and proposed
teacher actions never become teacher history or student feedback.

The student source history spans grouped same-task source trials. Exported
`history_inputs` retain actual previous noise/reward and episode boundaries
across those source-trial boundaries, with the original episode/primitive clock
values. `source_trial_starts` separately records original teacher reset points.
This is a declared chronological-query distillation variant, not independent
query sampling in the original DPT paper. IDs, version hashes and update clocks
are audit fields and are not appended to model inputs.

## Output and consumer boundary

Each objective writes NPZ plus `index.json` under a fresh output identity:

- Current and next `observations [T,2062]`, actual `executed_noise [T,1024]`, and separate
  `target_noise [T,1024]`; `history_inputs [T,3090]` for the full-history input.
- Actual learning/native rewards, separate termination/truncation flags,
  budget-cut flags, episode/source-trial starts, original episode/primitive
  clocks and horizons, valid executed durations, policy versions/update clocks.
- Per-task history `lengths`, task/trial identities in the manifest, exact
  input/checkpoint hashes, excluded-record counts and physical cost references.
  Target labels are the only NPZ field that differs between paired objectives.

Labels never overwrite executed noise, outcomes or boundaries. Raw RGB and
native feedback arrays remain at their checksum-pinned source references.
No new simulator actions or VLA queries are consumed. Source costs are shared
once across paired objectives, with quarantined actions still charged by the
original physical ledger.

This native-token schema is **not** a valid `vla_history_train` input: that
LIBERO consumer uses a different feedback layout and bound1 deployment adapter.
The native-bound full/static trainer and portable inference wrapper are
documented in [robotwin-history-policy.md](robotwin-history-policy.md).
Their qualification and actual K8 robot evaluation are separate from array
conversion. A separately trained supervised episode-reset student remains
unimplemented. Converting arrays does not qualify those cells.

## Command

Run computation/tests only on an allocated Linux resource arranged with the
RoboTwin execution owner. Development SSH is for setup/transfer/read-only checks.

```sh
python -m fluffy_bassoon.meta_rl.robotwin_history_corpus \
  --corpus-root <source-experiment/corpus> \
  --source-recovery <immutable-source-recovery.pt> --source-sha256 <SHA256> \
  --source-method sac \
  --teacher-recovery <immutable-source-trained-recovery.pt> \
  --teacher-sha256 <SHA256> --teacher-method sac \
  --objectives ad dpt --output <fresh-output> --device cpu --threads 1 \
  --source-commit <clean-exporter-commit> --expected-hostname <allocated-host>
```

`ppo` and `ppo_reset` are valid explicit source/teacher methods. DPT/BC-only
export uses `--objectives dpt behavior_cloning` when the source has no changing
policy history. Reusing a competent SAC teacher is the initial practical route;
the deterministic convention still needs its own source qualification.
