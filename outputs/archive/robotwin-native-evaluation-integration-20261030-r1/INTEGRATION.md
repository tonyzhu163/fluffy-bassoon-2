# Native evaluation clean integration — CPU qualification pending

This detached worktree starts at qualified native runtime
`db0184089fa88a5f96c153f9b80029bb91b77def`, not the older trainer branch.
Current `origin/main` was refreshed successfully and is
`acca706d5e75d1da099ff7ce6fddd727dace0fba`; it is an ancestor of this runtime.
No main changes were omitted or overwritten during integration.

## Changes and dependency resolution

DB lacked `vla_history.py` and `vla_history_train.py`. The native controller
inherits the shared Gaussian-history implementation and uses its checksum
helper; the native trainer uses the shared RNG save/restore helpers. The
EXACT qualified versions of these two modules, their existing history tests
and method documentation were brought from model qualification pin
`1aeea4f9fcc307bb147def92374bb81a3e2cb50e`. No unrelated collector or LIBERO
experiment implementation was copied.

The native exporter, trainer/controller and evaluation commits were then
cherry-picked in their original order. There were no merge conflicts. No
scientific or model changes were introduced to resolve the integration.
`PROVENANCE.json` records the exact source blobs and protected runtime blobs.

The native backend is byte-identical to DB and retains SHA
`eacc64b4daf8f78aeb5e023a2c4d2a2a040eacfd2e662335c118334c2ab50ff3`.
DB's newer native PPO collection callbacks and task-specific clocks were
retained. GTrXL, actor/critic implementation, SAC, warm-start validation,
campaign ledger, raw corpus, assets and final-camera capture retain their
native runtime versions. The PPO query-token helper and PPO configuration
match the qualification pin structurally; its newer collection code was not
replaced with the older trainer checkout's file.

The immutable nine-condition plan and four actual fixed100 endpoints are
unchanged. Plan file SHA remains
`b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c`.
The fitted models still correctly identify their TRAINING pin as1aeea4f;
the integrated deployment commit is a separate execution identity.

## Focused allocated CPU gate

Proposed owner-controlled CPU envelope:0 GPUs,4 CPUs,16 GiB,300 seconds.
No allocation or model/test computation was launched by this integration.
`focused-cpu.sh` checks Linux, the actual job hostname, exact clean commit,
then runs13 focused files covering native history/export/evaluation, shared
history, PPO/SAC, native environment, accounting, raw corpus, driver, assets
and final observations. It then loads all FOUR actual100-update portable
endpoints plus SAC5 through `admit_plan` on CPU. No motor/simulator is created.

Inside the allocated job, from the clean integrated worktree:

```bash
export FB2_EXPECTED_SOURCE_COMMIT=<exact-integrated-commit>
export FB2_EXPECTED_HOSTNAME=<actual-job-hostname>
bash outputs/robotwin-native-evaluation-integration-20261030-r1/focused-cpu.sh
```

The execution owner saves complete test/admission stdout, exact allocation
resources, exit status, timing and source/plan/script hashes under a fresh
output identity. A failure is diagnosed before any retry. Do not label this
integration CPU-qualified until the actual allocated run succeeds.

There are no known implementation/merge blockers after static review.
Allocated CPU qualification and the live allocation's lightweight GPU/CUDA,
capture/runtime/parity and real-episode gate remain required. The owner retains
the exact new evaluation ledger,57,600-action cap and all launch control.
No source continuation or importer code was modified.
