# Native repeated-reset control also fails strict dynamics identity

Baige job`job-lcwxw1uz7kdy` executed three64-scene modes: native batch-end trim,
a repeat of the same native mode, then experimental per-scene-close trim plus
batch-end trim. No VLA/learner was loaded. The job failed its unchanged1e-6
motor gate after recording all three modes; no candidate qualification is
claimed and no tolerance was relaxed.

| Mode relative to first native reset | Post50-control state maxabs | Reward/termination/truncation/mask/action identity |
| --- | ---: | --- |
| Native versus native |0.001026532642 |all exact |
| Per-scene versus native |0.000981113990 |all exact |

Initial head/wrist RGB and14D public joint states were bit-identical in both
comparisons. Python/NumPy/Torch/CUDA hold RNG and actual input action hashes
matched. Exposed articulated initial qpos, qvel and root velocities also
matched in the native repeat. Public probes retain object/root poses,
velocities, contacts and task/wrapper bookkeeping. They are not a complete
native physics snapshot and do not capture all C++ random state or solver
internals. Therefore the observed native repeated-reset variation cannot be
assigned to the experimental trim, and controlling these Python-level RNGs
does not establish the cause of the earlier failed comparison.

The repeated native mode peaked at106.186GiB process RSS. Per-scene close
trim peaked at56.173GiB and ended at55.786GiB after reset. This is useful RAM
attribution, but the candidate is not admitted by substituting a tolerance
based on the control. Production instead uses RLinf's existing batch-end trim
with the experimental callback disabled and an actual1TiB allocation. The
closed native memory02 phase projects608.732GiB total with conservative model/
runtime/rollout/checkpoint allowances, below the unchanged972.8GiB Ray guard.
Actual training phase peaks are monitored separately.

All9,600physical controls and2,560setup/reset allowances remain charged to the
originalrank2 ledger, totaling12,160for this diagnostic and72,960sharedrank2.
Scheduler duration1,971GPU-seconds; creation to release1,987; worker1,937. Created
2026-11-06 00:59:22Shanghai, released01:32:29. Source
`a9ee838c67eba8e22282739ee468ed15bee8670d`. No more diagnostics were launched.

[Raw result](RESULT.json), [native per-environment differences](NATIVE-REPEAT-COMPARISON.json),
[all raw physics hashes](ALL-PHYSICS-FILES.json), [exit](EXIT.json),
[GPU probe](gpu-probe.json), [reference](paired-reference.json).
The six`.json.gz` probes are byte-exact gzip copies of the raw JSON. Actual
submitted actions and outcomes also remain in the immutable remote diagnostic
output as`round1/2/3-hold0.pt`.
