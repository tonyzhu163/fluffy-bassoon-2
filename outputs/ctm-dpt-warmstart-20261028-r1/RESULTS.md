# First DPT-inspired CTM warm-start candidates

All seven compact workers saved portable `warmstart.pt` files and full
`latest.pt` recovery checkpoints on Baige. These are candidates, not accepted
ICRL initializations. The native worker failed before collecting a history;
its recovery has a separate run identity. No PPO comparison was run from these
seven candidates, so this pilot does not establish a benefit over direct PPO.

## Provenance and verification

- Baige job `job-qqijcsjw5rhu`, submitted 2026-10-28 18:01:14 Asia/Shanghai.
- Source `1569aee35495e2e16e651793537b03eb380aacd1`.
- One RPBZZZ6 GPU, 16 CPU cores, 64 GiB RAM, 60-minute scheduler cap.
  Compact training used three two-thread CPU workers within this allocation.
- All three compact worker groups exited zero. Native exited one, making the
  parent exit one; this does not invalidate the completed compact artifacts.
- Output root: `/mowenjie/fluffy-bassoon-2/outputs/ctm-dpt-warmstart-20261028-r1`.
- [CHECKPOINTS.json](CHECKPOINTS.json) records exact artifact paths, SHA256,
  compatibility, counters and final metrics. All seven export/resume digest
  pairs match the saved summaries and manifests. Hashes were computed from
  the remote files; weights remain on shared storage.
- [audit.py](audit.py) independently checks 84 saved raw behavioral means and
  attempt curves (initial/final, six conditions, seven benchmarks), without
  loading models. Its [receipt](evidence/compact-audit.json) passed. Raw arrays,
  metadata and update logs are in [evidence/run-r1](evidence/run-r1).

## Frozen candidate behavior

One training seed, CTM width 64 / four ticks, eight attempts, 256 training and
16 validation histories, batch eight. Behavioral evaluation uses 32 fixed fresh
trial seeds except eight for each grid. Initial and final policies use the same
evaluation plans. The grid metric is dense return per attempt, **not success**;
other rows are mean binary reward/success. All numbers below are raw means.

| Benchmark | Updates | Initial normal | Final normal | Memory wiped | Reward blind |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deterministic bandit | 200 | .62500 | .92188 | .92188 | .53125 |
| Bernoulli bandit | 200 | .57813 | .50781 | .50781 | .50781 |
| Nonstationary bandit | 200 | .57031 | .56641 | .57813 | .51953 |
| Hidden-gain reach | 200 | .00000 | .00000 | .00000 | .00000 |
| Dark Room | 112 | .00000 | .03125 | .76563 | .03125 |
| Key-to-Door | 37 | .01563 | .06250 | .00000 | .06250 |
| Cached rotation | 200 | .34375 | .50391 | .41797 | .35938 |

The 300-second soft training cap stopped Dark Room and Key-to-Door before 200
updates, with final evaluation/checkpointing bringing elapsed times to 308.2 and
321.4 seconds. Reach took 228.5 seconds; the one-step tasks took 26.8–27.4 seconds.
These are achieved budgets, not matched optimization comparisons or canonical
timing benchmarks. All seven validation NLLs fell, including unsuccessful tasks.

The deterministic bandit reaches 100% success from attempt two in both normal
and memory-wiped evaluation. Its immediately preceding action/reward already
identifies the answer; this is useful feedback-conditioned behavior but does
not establish persistent-memory use. Sampled success is .91016 normal versus
.86719 wiped and .52344 blind in this one-seed screen.

Cached rotation is the most promising history signal here: normal exceeds both
lesions under greedy and sampled actions (sampled .52734/.47266/.39063). It uses
only a scalar cached outcome table with shared object/angle support, not live
robot images, and 32 trials do not establish a reliable performance difference.

Bernoulli, nonstationary and reach show no useful greedy memory advantage.
Dark Room's greedy policy performs substantially better after memory wiping;
its sampled .35938/.29688/.39063 also fails the reward-blind comparison.
Key-to-Door has weak returns and sampled normal (.28125) trails wiped (.37500).
Preserve these negative results. Fresh grid seeds do not guarantee disjoint
latent goals. No checkpoint is automatically promoted to the PPO baseline.

## Native failure and recovery

The frozen pi0.5 model emits eight model-space action steps, but the native
decoder exposes five. The first labeler assumed equal horizons and raised an
einsum shape error before any history was saved. The failure receipt is retained
in [evidence/run-r1/native-collection/failure.json](evidence/run-r1/native-collection/failure.json).

Fix `e90b98b`, merged with reviewed main at
`79da19500f4e898d078539b4c562c150e610c16a`, relabels only the decoded prefix and
excludes the unchanged model-action tail from flow matching. The allocation
qualification passed **67 tests, 9 Slurm-only skips**; the new regression uses a
truncated decoder and checks that the tail is excluded from supervision.
[Qualification receipt](evidence/qualification-horizon-fix.log).

Only the native path was restarted as
[ctm-dpt-warmstart-20261028-r2](../ctm-dpt-warmstart-20261028-r2/PLAN.md).
The first run's data, checkpoints and source remain immutable.
