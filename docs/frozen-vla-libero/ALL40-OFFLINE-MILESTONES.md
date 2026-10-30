# All40 corpus to offline controller milestones

Prepared future execution; no allocation or component-budget reassignment is
made by this code. Root must declare its remaining approved LIBERO32GPU-hour
allocation before running. A bounded first run proposes one GPU/50 minutes,
four continuous500-update trajectories, zero new simulator actions. The
controller/driver guards remain600/2700 seconds; no method or hyperparameter
change is introduced.

The existing production assembler already accepts **all40 training goals and
the campaign's fixed8 source-local validation goals**. Validation is on
disjoint nominal states40..49 within those source goals; it is not40-goal
validation or target90/Plus/Pro data. It selects one latest complete paired
AD/DPT prefix per history, validates complete chronological cases and actual
state/hash separation, exact motor/features/noise contracts, actual teacher
clock changes, actor hashes and full selected collection costs.

The new thin entrypoint resolves inputs from the exact canonical SAC ledger,
not directory timestamps or a glob. All48 reservations must exist. Each
original run/source/cap/schedule must match its completed qualified index,
contract and successful terminal exit. Old reserved Spatial00/LIBERO10 runs
remain selected by their ledger identities. The old Spatial validation
qualification is not substituted for the new canonical validation slot.
Incomplete/raw-only histories remain preserved and explicitly prevent a full
assembly; no goal is dropped, no fixed-teacher prefix is called AD, and no new
collection is started by the offline pipeline.

After all source slots finish, record the **file-byte** hashes of the canonical
`campaign.json` and SAC `ledger.json`. In a fresh exact-source worktree on an
allocated GPU:

```bash
bash scripts/diagnostics/vla_source40_all40_training.sh \
  EXACT_REVIEWED_SOURCE FRESH_ALL40_OFFLINE_RUN \
  CAMPAIGN_FILE_SHA256 COMPLETED_SAC_LEDGER_FILE_SHA256
```

The launcher performs the actual allocation probe and focused Linux checks,
then calls the existing strict assembler. `assembly/all40-inputs.json` records
all48 exact source index hashes, standard assembly file/content hashes,
selected costs and separately spent prior qualification actions. This is not
the entire benchmark spend: PPO, target measurements and other campaign costs
retain their own receipts. No input file, active source worktree or ledger is
modified.

The existing milestone trainer now has an explicit `--all40-source` route.
It requires the hash-pinned production assembly and strict all40/fixed8
admission, while preserving `--qualification-subset` for old pilots. Both
routes use the same optimizer/trainer/export code. AD/DPT full/static each
follow one25→100→500 update trajectory with immutable exports and complete
optimizer/sampler/global RNG recovery, rather than25+100+500 retraining.
Parameters remain seed0, batch8, LR3e-4, width128, two layers/four heads,
memory300, window128, validation every25 updates/four batches. Goal sampling
and loss weighting are unchanged.

Production `scales.json` uses `openpi_pi05_source40_scales_v1`, with
`qualification_subset=false`; its per-scale matrices retain the existing
matched-training format accepted by `load_candidates`. The singular
`frozen_source_sac` field remains a first-source actor reference for
compatibility, not a jointly trained all40 SAC baseline. All final controllers
and source actors remain candidates, with robot performance to be measured.

Append `--resume` only for the same source/run/manifest/configuration and
cumulative time bounds. A time-limited run keeps full recoveries and exact
completed milestones. No later checkpoint is relabeled as an earlier scale.
