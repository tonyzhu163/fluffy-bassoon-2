# PPO production CPU r1: fixture failure, preserved

Baige `job-xeijw76abacg`, run `libero40-ppo-production-cpu-20261031-r1`,
source `6a24023e0f3ec6a44f952f9bffbf1df28c227c54`, finished exit1.
Actual0GPU/4CPU/16GiB attestation passed. Pytest reported6 passed/1 failed,
with0 skips; the dedicated CPU-worker clock charged82 seconds including its
60-second startup allowance, and0 GPU-seconds. Original logs, JUnit, clock,
failure record and checkpoint integration evidence remain untouched.

The failed reset-delegation unit fixture passed `object()` as its configuration.
The real `prepare` API builds a batch training contract using
`dataclasses.asdict(config)`, so this invalid fixture raised
`TypeError: asdict() should be called on dataclass instances` before delegation.
The correction supplies the same tiny `RL2Config` dataclass shape used by the
other fixture and checks that this config reaches the generic warmstart adapter.
**No production core, algorithm, optimizer, action/time budget or method contract
changes.** Allocated Linux rerun under a new r2 identity is required; no tests or
models were run on the Mac/development host to verify the correction.

The actual shared-PFS FULL migration integration test **passed in r1**. It
verified producer376415e, recoveryc689, every non-contract state leaf, cursor14,
Adam478, typed motor RNG, pending state, copied9,272-action prefix, zero restore
SGD and unchanged original bytes. Thus the overall CPU suite failed because of
its reset fixture, not a failed real FULL migration. This result does not provide
PPO performance evidence or authorize production scaling.

This CPU test failure does not gate the separate source-SAC collection proposal.
`receipt.json` binds the compact diagnosis to the original canonical evidence
hashes; all original run artifacts retain their r1 identity.
