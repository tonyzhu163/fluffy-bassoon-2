**Status: COMPLETE, all six workers exit0; final results audited21:35Shanghai.**
[Final outcomes and limits](RESULTS.md).

# Existing-model target first wave

Direct human authorization was verified in coordinator chat
`01a18be4-a6ab-79a2-af3d-13f76308e253`: test the completed models on90/Plus/Pro.
This screen uses the unchanged official OpenPI40 frozen motor, source-only
fixed500 AD-full/AD-static/DPT-full/DPT-static exports and native behavior.
All24 training milestone policy/recovery payload hashes were checked before
launch. No target checkpoint selection, target pretraining or deployment SGD.

Source: `d095a2a4e66a0b79d03c9e823a183c19452d0c10`; reviewed main
`acca706d5e75d1da099ff7ce6fddd727dace0fba`. Plan file SHA256
`e29686c69d7da8410d1145fcfa76b357e9f0018e4662cffe581f0adcb587f767`;
content hash `0921a19aab96a6e00afafa06fd83ac70d8a996b491295146fcdd31cd2e75e63b`.
[Exact cases/groups](plan.json), [separate target ledger](target-ledger.json).

- PhaseA: all90 LIBERO90 task IDs, state0/seed123, plus4 Plus conditions and4
  seeded Pro conditions; native and all4 trained models, zero support.
- PhaseB: LIBERO90 IDs0,10,...80 and all8 shifted cases; AD/DPT full and
  same-weight episode-reset each get one actual support then a separate query.
- Episode-reset K0 references the identical full-model empty-history result;
  it never substitutes the separately trained static model or invents a run.
- Plus is variants667/290 × states0/1. Pro is swap/appearance × seeds123/124
  at state0: seeded repeats across2 variants, not4 nominal states/new tasks.
- LIBERO90 horizon400; Plus/Pro280-step development cutoffs. This is full90
  goal coverage with one initialization each, not exhaustive Plus/Pro results.

There are98 cases,558 groups and626 physical episodes. Nominal243,920 actions
INCLUDES all settling:196,000 PhaseA +47,920 PhaseB. Qualification uses the
first planned reset, adding no reset/episode. Six per-worker nominal+1,000
caps sum249,920 within250k. Six≤3,600-second cumulative GPU holds total6h,
including setup/failures/retries; initial slices3540s, scheduler60min. Funding
comes from reconciled terminal SAC time inside the existing32h approval,
after counting offline3,200s and speed1,800s once. Total component holds
115,086≤115,200seconds. Source contracts/caps/ledgers are unchanged.

All6 allocations passed actual GPU/CUDA checks and77 focused Linux tests.
All626 episodes completed by17:32Shanghai. The later final audit found native
31/90 versus26–27/90 for the students; K1 retained6/9 versus reset5/9 merely
matches native6/9 on the nine-case panel. No performance improvement established. All methods in each paired case
share one actual motor. Raw observations, noises/actions, feedback, contexts,
RNG/cursors, RPC costs and actual frame equality flags are retained. Frame
mismatches preclude a strict causal-memory claim rather than being hidden.
A wallstop preserves complete groups/costs and must report partial coverage.

| Worker | Baige job | Cumulative GPU limit |
| --- | --- | ---: |
|00|job-7usfjchizpoa|3600s|
|01|job-oeefcfxwn25p|3600s|
|02|job-1louiqntu06i|3600s|
|03|job-c7ai91k5ypoa|3600s|
|04|job-l0yi9us6d16o|3600s|
|05|job-ydm04e08evzd|3600s|

Baige roots are `/mowenjie/fluffy-bassoon-2/outputs/` and `worktrees/`, each
with run IDs `libero40-target-firstwave-20261030-wII-r1`. Full source dataset
and checkpoints remain in their original immutable roots. Ordinary DSRL and
PPO comparisons remain separate required experiments; this first wave does
not claim those results or a reliable memory/topline benefit.
