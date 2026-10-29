# Common-runtime source-data robot comparison

Fresh processes produced different compiled BF16 motor actions for identical
saved inputs, although repeats, initialization stages and typed/legacy keys
were bitwise stable within a process. This experiment therefore evaluates
both immutable source banks in **one FrozenOpenPI object and executable**.
It does not claim that changing JAX key types fixes that numerical nuisance.

Source1 is the original one-goal Spatial00 corpus; Source2 adds LIBERO10:00
and its independent local validation. Each bank contributes its four fixed500
AD/DPT full/static candidates plus same-weight episode-reset interventions.
The twelve learned variants each receive0/1 actual supports on the same
predeclared LIBERO90 task0/1, state0, scene seed123. Source condition is separate
from TargetCase and never changes the query seed. Banks are interleaved per
method within a case. All modules are loaded before the first motor inference.
Every comparator resets the motor with legacy `jax.random.PRNGKey(seed)`.

The controls are one native K0 query and one **frozen final target-online SAC**
K0 query per case. The latter restores the complete final recovery from
`libero40-target-online-dsrl-20261030-r2`; no source actor or joint SAC model is
substituted. Both final candidates had15 training episodes and2,400 updates.
Their original adaptation costs are reported separately:5,976 actions for
task0 and6,150 for task1. Old measurement/qualification costs remain in their
original receipts. The new queries perform zero updates and add no replay.

The full design is28 groups and76 real episodes:72 student episodes plus4
control queries. At410 actions per episode, its nominal ceiling is31,160.
The joint32,000-action cap also charges settling, retries and unknown RPCs.
After GPU/tests pass, the launcher reserves the parent-funded exact-run/source
32k target-evaluation slice; the driver only reads it and does not change the
1M campaign. A preflight failure creates no action reservation. The unlaunched16k
two-goal-only plan is superseded, while its prepared artifacts remain intact.

After funding the32k slice in the campaign plan, use a fresh clean pinned worktree in one
allocated GPU for at most40 minutes:

```bash
bash scripts/diagnostics/vla_common_runtime_evaluate.sh \
  EXACT_REVIEWED_SOURCE FRESH_COMMON_RUNTIME_RUN
```

The launcher runs focused Linux tests and the actual allocation GPU probe,
then the driver has a2100-second rollout bound and a2220-second process
timeout. Its immutable scale pins are:

- Source1: `a52a2317f5b353d8e6afa0bc81131896426987197b49256d80b8faef354139dc`.
- Source2: `f1577bcbb5351743871ed2b3b0bcff627ea952f1a1aa0b724e147b7085e5d263`.

Full target recovery, contract and index hashes are also pinned. Existing
trace/context helpers retain actual observations/actions/noises/rewards and
boundaries, pre-RPC inputs, sampler/global/controller/motor RNG and complete
episode/group cursors. Initial nominal states and actual RGB/wrist/state
arrays must match across banks and controls. The qualified sequential motor
parameter hash is recorded before/after. A changed parameter state makes the
result non-primary and fails the run; equality with the earlier replay-probe
reference is also reported.

An invocation ID is attached to each executed group attempt. Exact-source
`--resume` preserves all recoveries and physical charges. If physical episodes
span fresh motor processes, the final result is explicitly non-primary for
strict shared-runtime inference. Completed groups are verified and skipped;
they are never silently relabeled as if a new process had executed them.
The source-scale comparison must be read from the new common-runtime results,
with original source/adaptation costs separate, rather than combining old
cross-process success rates into a causal source-data claim.
