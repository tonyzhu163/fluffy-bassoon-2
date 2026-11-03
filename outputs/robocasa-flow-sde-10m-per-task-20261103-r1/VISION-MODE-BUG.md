# Frozen image encoder mode and Flow-SDE PPO likelihoods

The initial OpenDrawer Q1 source `686d2619`, Baige `job-2z7heniyxam3`,
completed its before panel (40/64) and collected 128 TRAIN episodes, but the
new first-micro-batch guard stopped both ranks before backward or SGD.
Maximum chunk log-ratio errors were 0.1243286 and 0.2873535 at unchanged
weights, above the planned 0.001 bound. All samples and the initial full
checkpoint remain retained. The process charged4352GPU-s; scheduler creation
16:28:48 through terminal16:38:38 bounds this attempt by4720GPU-s. No other
task had been launched.

## Native replay evidence

The one-GPU replay uses the exact saved actor batch, realized denoising chains
and original model weights. It makes no simulator calls or optimizer steps.

| Forward condition | Original 16-sample cohort: max chunk log-ratio | Native shuffled 16-sample cohort |
| --- | ---: | ---: |
| Eval mode, gradients off | 0 | 0 |
| Train mode, gradients enabled | 1.635376 | 0.124329 |
| Train mode, frozen image encoder in eval, gradients enabled | 0 | 0 |

R1 (`job-h6vnge2o397b`, source `fa16fbb3`) established the first two rows.
R2 (`job-crl6vl8lx7a3`, source `090cd7d5`) reproduced them and qualified the
third row. [Full R1](PARITY-REPLAY-R1.json), [full R2](PARITY-REPLAY-R2.json).
This replay qualifies the sampled likelihood comparisons. Subsequent native
eight-GPU qualification passed for all five tasks, as recorded below.

The frozen image encoder contains PyTorch `nn.MultiheadAttention`. Its eval
flag affects forward-path selection even under `no_grad`; freezing parameter
gradients alone does not preserve the rollout computation. Native actor
`run_training()` calls `model.train()` recursively. The fix, `0988b5f9`,
returns only the frozen image encoder to eval before the first micro-batch of
each update. Expert/value training, native shuffle, minibatches, loss,
optimizer, weights, sampler and fixed rollout/eval layouts remain as declared.
No deterministic-kernel setting is enabled. All39required Linux tests pass.

## Previous-run scope

- Confirmed: the completed eighteen-update OpenDrawer full action-expert
  teacher uses source `9ed1d19d42a346c6330c4d7f55303a48d21158ad` and native
  RLinf `8a1233c3fae6f5905adfda7684c455842b0ecfb8`. Its `FlowActor.run_training`
  directly delegates to the same native `EmbodiedFSDPActor.run_training`,
  without restoring image eval mode. Its PPO training is therefore affected.
- The old before/after robot success counts remain observations of the actual
  saved policies. The run already failed its improvement gate; it cannot
  establish how the corrected intended PPO recipe would perform. Its KL/clip
  metrics mix training-mode mismatch with parameter updates.
- Four-method source preparation explicitly reads the old run's TRAIN
  histories and checkpoint18 (`flow-sde-four-methods-20261103-r1/PREPARE.py`,
  `ASSEMBLY.json`). Those fits remain plumbing/reference evidence with an
  affected, non-admitted source, without a claim of improving-source learning.
  Their running jobs and contracts have not been changed here.
- The velocity/KV steering route is separate: `FrozenRoboCasaInterface`
  initializes `native.model.eval().requires_grad_(False)`, and the worker
  rejects a motor in training mode. This specific actor/vision mode defect
  does not apply to that route. Other known pairing/KL issues remain separate.
- No broader claim about every historical learner is established by this audit.

## Corrected qualification

OpenDrawer Q1 retry `job-8vyx98mauo07` pins `0988b5f9` and restores the original
step-zero actor/value/Adam/sampler/RNG checkpoint. The immutable
[before reference](BEFORE-RETRY0.json) reuses all64original before cases;
no second before collection is needed. The failed TRAIN bank is excluded
from learning and remains charged in the same task accounting root. Retry
output and journal identities are new. All five corrected Q1 jobs have now
succeeded and released their allocations. Each completed24optimizer steps,
passed the pre-backward likelihood check on both actor ranks with zero error,
kept the frozen VLM unchanged, and saved a complete eighteen-worker boundary.
The allocated-Linux audit verified all128TRAIN episodes and1536actor samples
per task, unique environment episode joins, exact camera/chain duplication,
and all checkpoint file hashes. Per-task `Q1-AUDIT.json` receipts and the
verified before/resume references are retained here.

All five production jobs resume the qualified step-one boundaries under the
same `0988b5f9` runtime. Their identities and live progress are in
[EXECUTION.json](EXECUTION.json); active worktrees are immutable. Fixed batch
layout remains16TRAIN envs/GPU,8EVAL envs/GPU,one collection round and two
actor ranks. Bitwise deterministic kernels are not required or enabled.

The first full boundary actually occupies about15GiB, exceeding the plan's
0.7GB estimate. Production will tier superseded complete checkpoints to the
existing BOS archive mount with tar comparison, full SHA256 readback and a
restore receipt, while retaining the newest full boundary locally. This is an
execution/storage correction; checkpoint contents and cadence stay unchanged.
