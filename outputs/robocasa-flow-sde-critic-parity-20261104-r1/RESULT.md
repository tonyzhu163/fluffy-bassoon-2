# Critic parity: saved-batch evidence complete; short comparison running

Owner chat `01a1a4a2`. On Nov4, matching-checkpoint replay confirms the native
RLinf critic mismatch on real Drawer→Counter records. Rollout saves the mean
of ten suffix values; native PPO recomputes the value at the selected suffix.
The measured clipping effect is limited in these cohorts. This does not show
that the defect caused the historical teacher-gate failures or require a
blanket five-task rerun.

## Verified unchanged-weight replay

Sixteen exact archive/actor/sampling/environment-joined B16 cohorts cover
policy0/q1 and production policies1,10,120:256 decisions,246 active loss-mask
entries. All reconstructed saved rollout means and native selected predictions
match exactly on GPU. All16 likelihood checks have zero coordinate and chunk
log-ratio error. Strict checkpoint/model, fixture, READY and consumption hashes
pass the [independent Linux CPU audit](FINAL-AUDIT.json).

| Policy step | Decisions / active | Maximum value gap | Above .2 before SGD | Suppressed prediction gradients |
| --- | ---: | ---: | ---: | ---: |
| 0 | 64 / 61 | .211556 | 1 | 1 |
| 1 | 64 / 61 | .089210 | 0 | 0 |
| 10 | 64 / 62 | .138693 | 0 | 0 |
| 120 | 64 / 62 | .196716 | 0 | 0 |

With the same historical return targets, changing the clipping reference to
the selected prediction changes the head gradient by17.99% in one initial
B16 cohort. Later-production head-gradient changes are at most2.8e−10 relative
(numerical roundoff). The CPU audit independently recomputes native losses
and prediction gradients. Head-gradient vectors were not retained; their
norms and deltas are GPU replay reports. Saved-batch returns are fixed: this
comparison does not recompute corrected GAE or measure learning outcomes.
See [per-cohort GPU reports](gpu-r2/replay/RESULT.json),
[policy summary](critic-summary.csv) and [loss masks](LOSS-MASK-READOUT.json).

The first CPU audit required bitwise equality across a CUDA/CPU float32 mean.
The corrected audit reduces the retained float32 inputs in float64 with a
declared2e−7 absolute bound; the observed maximum is1.0133e−7. Selected-step
gather equality remains exact. The original failed audit log is retained.

## Qualified correction

[The patch](../../patches/rlinf-8a1233-selected-suffix-critic.patch) stores the
same selected suffix prediction that native PPO recomputes. Twenty Linux CPU
numerical cases execute the actual original/patched sampler method bodies
and native sampler/value helpers, using synthetic features. All ten selected
indices have zero fixed value gap; checked actions, chains, likelihoods,
indices and generator state are unchanged. Prefix, joint, NFT, no-value and
ignore-last cases preserve their stated contracts. This is method-level
qualification; the live fixed arm also must pass real pre-SGD parity.
[Numerical evidence](QUALIFIED-FIX.json), [core checks](CORE-QUALIFICATION.json).

The current-main core checks pass38 unique cases on Linux CPU. Two test-harness
setup attempts used a missing/incorrect legacy `EMBODIED_PATH`; the seven
affected config cases passed after setting it to RLinf/examples/embodiment.
No model computation or numerical tests ran on macOS.

## Short learning comparison

The initial active-sample gradient suppression admits the conditional
three-update comparison. [Its declared controls and ceilings](COMPARISON.md)
hold the native recipe, full initial checkpoint and RNG/cursor sidecars fixed.
Both arms reuse the verified original policy-0 before panel. The selected-value
vendor is an isolated clean Git commit `8f9bfa76`; shared active worktrees are
unchanged. The older common runtime `0988b5f9` is intentional to hold all other
learner/environment changes fixed.

Baige Pro6K `job-eigq1w5m3ba9`,
`mwj_1104_flow10m_critic_compare_8x1_r1`, started Nov4 14:18 Shanghai.
Eight-GPU probe passed; baseline initialization is advancing. No comparison
result is claimed yet. [Admission](COMPARISON-ADMISSION.json),
[deployments](COMPARISON-DEPLOYED.json), [state](EXECUTION.json).

## Cost and failure evidence

The replay's first attempt failed at a wrong checkpoint-helper import before
model construction. Its consumer/stager were verified terminal and a corrected
fresh identity passed exact-runtime CPU import qualification before R2.
R1 costs108 conservative GPU-seconds; successful R2 costs285, total393
(0.1092GPUh). Both scheduler allocations are released; the R2 stager naturally
exited after four consumed/reclaimed private copies.
[R1 diagnosis](ATTEMPT-R1.json), [R2 closeout](SCHEDULER-CLOSEOUT-R2.json).

The comparison's42-minute/8-GPU scheduler maximum plus those393 seconds is
20553 seconds (5.71GPUh), below this validation's6GPUh ceiling. Automatic
retry is off. Final allocation release, controls and cost are pending.
