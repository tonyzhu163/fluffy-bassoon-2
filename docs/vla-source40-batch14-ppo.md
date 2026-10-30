# Complete-trial source PPO batches

This source-only implementation has passed allocated Linux tests and actual
fresh and warm-started native-horizon batch14 updates. The immutable R3
one-trial runs remain unchanged; warm start is an explicit, lossless new
phase importing their policy, Adam, sampler and global/motor RNG state.
Full1M scaling has not been launched. The current scientific priority is
robust PPO and ordinary DSRL baselines.

Verified qualification: fresh14 K4/56 episodes/795 decisions/one global update
and two Adam steps; warm14 K4/56 episodes/894 decisions with exact replay and
two additional Adam steps, preserving the inherited476. Warm costs9,272
all-in source actions and910 conservative GPU-seconds. See the
[warm qualification](../outputs/libero40-batch14-warm-qualification-audit-20261031-r1/RESULT.md)
and [fresh qualification](../outputs/libero40-batch14-qualification-audit-20261031-r1/README.md).
These are runtime and recovery evidence, not robot-transfer or memory gains.

| Setting | Batch14 contract |
| --- | --- |
| Motor | Official OpenPI `pi05_libero`, frozen; receipt `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170` |
| Source | All40 Spatial/Object/Goal/LIBERO10 tasks; train state IDs0–39; no90/Plus/Pro data |
| Schedule | Goal index0–9, interleaved Spatial/Object/Goal/10; independent of outcomes |
| Cases | Cycle train-state IDs0–39 separately for each goal; checkpointed scene-seed draws;35 distinct nominal IDs per goal if all100 batches finish |
| Horizons |220 /280 /300 /520 executed actions;10 settling actions per episode |
| Interface |4104 public VLA-prefix features,32D bounded steering noise, H10; actual unchanged samples |
| Trial | Four complete episodes from one actual fixed state/seed; full memory or episode-reset control |
| Global batch |14 complete variable-length trials; one frozen behavior policy throughout collection |
| PPO | Two epochs, Adam LR1e-4, clip0.2, gamma/lambda1, entropy0.01, value coefficient0.5, gradient clip1 |
| Weighting | Sum of losses over all real decisions / total real decisions; no padding; global valid-token advantage normalization |
| Optimizer | One step per epoch on the entire merged batch: two steps, not28 |
| Target |100 complete batches /1400 K4 trials; hard physical ceiling wins |

The horizon profile follows the pinned official [OpenPI LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py#L54-L63).

The implementation reuses `collect_rl2_trial`, `RL2ActorCritic`, its GTrXL core,
the existing GAE/PPO objective, raw episode sink, physical RPC journal, typed
motor-RNG serialization, and complete-trial task cursor. It concatenates real
trials along time into a single lane. Every new trial resets all memory/cache
and position-length leaves and begins with zero previous noise/reward/done.
GAE stops at each K4 trial end; physical episode ends inside a trial retain
credit and, in full mode, memory. Existing R3 files are unchanged.

Before SGD, the trainer checks exact stored-versus-replayed likelihoods and
values for every individual trial and exact individual-versus-merged replay,
including ragged lengths. A changed policy/optimizer during collection is an
error. Focused tests additionally compare the valid-token loss and gradients
with a sum of independently replayed trial losses (float32 gradient tolerances
`rtol=2e-5`, `atol=2e-7`) and prove that trial reset blocks cross-trial gradients.
These tests passed on allocated Linux; they were not run on macOS.

The atomic recovery contains policy, optimizer, global Python/NumPy/Torch/CUDA
RNG, scene sampler, typed motor RNG, source cursor/exposure, the full pending
rollout tensors and receipts, batch plan, one behavior checkpoint hash, and
optimizer/exposure counters. It saves after every complete K4 trial. A partial
14-trial batch survives a wall stop and does not update. The ready-batch snapshot
precedes both optimizer steps; interruption replays the same batch/optimizer/RNG
without recollection. Actual interrupted episode prefixes and unresolved RPC
intents remain charged and retained. A failed current trial is recollected
from the last complete-trial recovery, not treated as a complete rollout.

At every new batch the entire declared worst-case K4 allowance must fit. The
first batch is at most18,400 rollout actions plus10 qualification settling
actions. Any14 consecutive source trials cost at most19,680 before retries.
All1400 trials at native horizons would cost1,904,000 actions before qualification
or retries. Therefore a1M ceiling cannot guarantee100 updates. Early success
may reduce costs; it is measured, not assumed. Retry/requalification actions can
also exhaust the allowance for a preserved partial batch. In that case it stops
with the partial recovery and does not cut episodes, shrink the batch, or exceed
the physical cap. The caller must also charge this run against the campaign's
remaining approved ledger; the per-run cap does not create a new budget.

Summaries expose actual updates/optimizer steps, primitive/settling/physical
costs, successes/failures, S/F trial patterns, fail-then-success counts, task
coverage, nominal state IDs, seeds, and actual nominal/settled state hashes.
A seed alone is not claimed as a new physical state. Unexpected collection,
likelihood, optimizer, or finite-value errors stop with a diagnostic; diagnose
before retrying or expanding scale.

The original40-task SAC corpus had2014 successes/94 failures in2108 training
episodes;454/527 cases are SSSS and only10 learned-behavior trials contain a
failure followed by a later learned-behavior success. This motivates checking
headroom and useful history signal before automatically repeating its recipe.
The batch implementation does not manufacture failures or establish that more
data would repair negative results.

The historical qualification proposal was one14-trial batch on one
allocated GPU, at most20,000 actions and40 minutes, including focused tests,
reset/feature/noise checks and overhead. It should measure optimizer time and
peak allocated/reserved memory as well as pass exact replay/finite gradients.
The actual bounded qualifications above completed. Their original failed or
partial attempts and full recoveries remain retained. No full-scale run follows
automatically, and the new phase needs an explicit cumulative resource budget.

After an approved caller has established its cumulative time/action ledger,
GPU probe, clean pinned worktree, qualified OpenPI/cu128 environment and exact
source schedule, the module invocation is:

```bash
"$MODEL_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_batch14 \
  --source-commit "$PINNED_COMMIT" --expected-hostname "$ALLOCATED_HOSTNAME" \
  --schedule "$SOURCE40_SCHEDULE" --output "$NEW_IMMUTABLE_OUTPUT" \
  --checkpoint /mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1 \
  --openpi-root /mowenjie/fluffy-bassoon/third_party/openpi-215abfb \
  --simulator-python /mowenjie/envs/wam-libero-plus-py310/bin/python \
  --simulator-pythonpath "$PINNED_SIMULATOR_PYTHONPATH" \
  --max-total-actions 20000 --stop-unix "$LEDGER_INVOCATION_DEADLINE" \
  --updates 1 --memory-mode full
```

The40-task schedule/BDDL and original LIBERO commit are checked at load/reset.
An episode-reset comparison uses the same source order and hyperparameters in
a separate immutable funded identity. `--resume` retains the original complete
scientific contract while the caller supplies a new ledger-bounded deadline.
The entry point allocates no GPUs and reserves no budget.

Allocated focused tests, using the qualified repository test environment:

```bash
"$TEST_PYTHON" -m pytest -q tests/test_vla_rl2_batch.py \
  tests/test_vla_source40_batch14.py tests/test_vla_rl2.py \
  tests/test_vla_source40_multitask_ppo.py tests/test_vla_source40_ppo.py
```
