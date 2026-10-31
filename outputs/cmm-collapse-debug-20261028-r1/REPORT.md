# CMM calibration collapse: diagnostic findings

Checked 2026-10-29 00:27 Asia/Shanghai. The evidence points to an optimization
instability followed by partial recovery. No specific implementation defect or
causal training fix has been established. Keep this separate from a conclusion
about the published CMM architecture.

## Live training and provenance

- Secondary school cluster **thu2**, job **48876**, node02, one RTX3090.
- Immutable source: `b6919d6b95a02704dee97cd03813a4c2ac1749de`.
- Run: `fb2-cmm-cal-k20-s0-b6919d6-20261028-r1`, CMM K20, seed 0.
- Best development query accuracy: **16.52%**, epoch 450, MSE **0.08093**.
- Epoch 650: **3.98%**, MSE **0.95327**. Epoch 700: **3.76%**, MSE **0.95863**.
- Latest observed evaluation: epoch **775**, **12.065%**, MSE **0.13415**.
- Still running, no final test. Decay jobs **48983/48984** remain queued for
  priority. No live training, queued job, optimizer, or source was changed.

## What the saved history establishes

| Epoch interval | Median training loss | Median pre-clip gradient norm | Updates clipped |
| --- | ---: | ---: | ---: |
| 400–450 | .04682 | .921 | .03% |
| 450–500 | .04671 | .991 | 1.17% |
| 500–550 | .05228 | 1.654 | 22.92% |
| 550–600 | .08256 | 14.488 | 72.64% |
| 600–650 | .21944 | 29.768 | 66.27% |
| 650–700 | .47298 | 3.008 | 47.78% |

The largest recorded pre-clip norm was **128,012,384** at update 108654.
Instability grew before the collapse; this was not one isolated bad update.
The logger records the norm returned by `clip_grad_norm_`, which is the norm
before clipping. It does not record the size of the Adam parameter update.

The copied-checkpoint replay confirmed a norm of 6.713 was clipped to 5.000.
A restored Adam step was finite; its largest relative parameter update was
0.291% on this one four-lane probe. This rules out missing clipping in that
replay, not every possible optimizer failure earlier in training.

## Allocated CPU diagnostics

Both diagnostics used Slurm CPU allocations, no GPU. Job **49112** completed
exit 0 in 15 seconds and **49114** completed exit 0 in 33 seconds. The first
also passed **20 focused tests**. Diagnostic commits: `bddefc83` and `0b01a943`.
The CMM core, wrapper and synapse source are unchanged from the training pin.

Preserved snapshot: update **115395**, epoch **735**; SHA256
`6006c065d24de41f14eef636ee2f54d0d13c801e9d0e017e224084cd2b7080ae`.
Its full policy, optimizer, counters and RNG payload remain on thu2 at
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/cmm-collapse-debug-20261028-r1/snapshot.pt`.

### Task and support/query behavior

On a 32-episode sanity sample, support MSE was **.01224**, query MSE **.79183**,
and query accuracy **3.906%**. Least squares on the same supports and queries
achieved MSE **.02483**, accuracy **32.031%**. The data retain a usable regression
signal; the collapsed model largely succeeds at copying observed support values
while failing to infer the query mapping. This small sanity check is not a
paper benchmark reproduction or a full test score.

### Gradient paths

Detaching selected state components only for backward left all forward
predictions bitwise identical, as asserted by the diagnostic.

| Backward intervention | Four-lane norm | Median norm across 16 eight-lane batches |
| --- | ---: | ---: |
| Full recurrence | 6.713 | 2.009 |
| Detach LTM between steps | 6.506 | 1.997 |
| Detach activated state | 2.094 | — |
| Detach raw STM | 2.661 | — |
| Detach activated state and STM | — | 1.579 |
| Detach synchronization accumulators | 6.419 | — |

The strongest amplification on the first batch came through the short-term
activation/STM recurrence and the synapse input projection. The broader probe
showed a smaller effect and no extreme spike: maximum full norm **4.638**.
LTM-only detachment changed little in both checks. This argues against blaming
the explicit LTM gradient path alone, but does not replay the historical
catastrophic updates or prove which path initiated their instability.
These detachments are diagnostic interventions, not proposed training fixes.

### Forward memory interventions

256 new episodes, 5,120 queries, frozen epoch-735 snapshot:

| Intervention | Query accuracy | Query MSE |
| --- | ---: | ---: |
| Normal | 4.336% | .82601 |
| Erase LTM at the support/query boundary | 4.238% | .86393 |
| Erase all state at that boundary | 3.945% | .97984 |
| Zero support values | 4.316% | .97937 |

Support information still improves MSE, although the strict accuracy metric is
near chance. LTM erasure has only a small effect at this collapsed checkpoint.
This says nothing decisive about LTM use at the healthy epoch-450 checkpoint.

## Contract audit and remaining uncertainty

The reviewed data construction, full-episode MSE, query-only threshold metric,
AdamW learning rate 1e-4, zero weight decay, batch 64, clip 5, memory layout and
1,000-epoch budget match the stated settings in the
[paper](https://arxiv.org/html/2610.07907v1#Sx4.SS5). State is reset per episode;
the functional recurrent state is neither mutated nor implicitly detached.
The existing equation-oracle tests passed. The authors' linked repository
still returned 404, so upstream numerical parity remains unavailable.

Unstated choices remain: synapse details, initialization, noise interpretation,
normalization epsilons and attention heads. The port has **488,195** parameters
against the paper's **496,129**. Test success does not settle this fidelity gap.
The decay-rate median is **.03625** at the preserved snapshot, corresponding to
a synchronization half-life of about 19 steps; whether this contributed to the
plateau or collapse is unproven. The queued initialization comparisons test
early learning speed and do not, by themselves, isolate late instability.

## Smallest next discriminating experiment

Keep the current run immutable and finish its final test. For any subsequent
bounded recovery comparison, preserve separate milestone checkpoints, replay
identical full batches and RNG, and compare the original learning rate with a
lower rate from the same healthy checkpoint. Log per-module gradients and the
actual Adam update norm. Only `latest.pt` was retained in the original run, so
the epoch-450 checkpoint and exact pre-collapse states cannot be recovered
from its checkpoint directory. A fresh or otherwise preserved healthy source
would be required; the epoch-735 snapshot is not a substitute.

No new full calibration or GPU experiment was launched. No production fix is
claimed. Raw receipts: `history.json`, `results.jsonl`, `tests.log`, and
`../cmm-collapse-debug-20261028-r2/{probes,lesions}.json`.
