# Meta-World noise screen: reach conditions

**Paused by the user at20:17:43 Shanghai on2026-10-29.** The completed
reach results below remain valid; push-hidden is incomplete and the final
motor fingerprint is unavailable. See [pause record](PAUSED.md).

Independent saved-artifact audits: **2026-10-29 19:24 and 19:58 Asia/Shanghai**.
Baige `job-orgyjb0z9h6d` remains active on its original scientific source
`4d4f46f4cdcd8cfd7c7056e95bf85d0a90e0ed59`. Visible and hidden reach each
closed with all **544 episodes**. The tested **16 fixed random noise arms** fail the headroom and
heterogeneity gates, and leave two of four goals unreached. This rejects
using this bank for the proposed cached selector; it does not reject DSRL
or establish that learned noise steering cannot improve this frozen motor.

| Reach condition | Native success | Split-half noise selection | G1 gain (required ≥15pp) | G2 gain (required ≥10pp) | Goal coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| Visible | 10/32 = 31.25% | 37.5% | +6.25pp, fail | 0pp, fail | Fail |
| Hidden | 9/32 = 28.125% | 25% | −3.125pp, fail | −12.5pp, fail | Fail |

Arm selection uses one half of the scenes and is scored on the other half,
then reverses the halves. G2 compares against an arm selected using the other
goals. The hidden-goal reference-selected static arm scores 37.5% on this
procedure. Neither condition admits cached-selector training over this bank.

Interpretation correction, 2026-10-29: this run supplies one of 16 Gaussian
noise tensors to the flow sampler, reusing that same tensor for every chunk
of the attempt. It trains no noise policy. [DSRL](https://diffusion-steering.github.io/)
learns a policy over continuous noise using reward; its observation-conditioned
choices can change from chunk to chunk while the base motor stays frozen.
Our screen establishes sensitivity to noise and limited headroom for this bank,
not the effectiveness of learned steering. Reusing one tensor across time is
an additional restriction. The visible native score is also only 31.25%, and
hiding the goal marker changes the motor's observation distribution. These
are candidate explanations, not a diagnosed cause. The next discriminating
experiment would train per-chunk noise steering with the motor frozen; no
such training is authorized by or added to this allocation.

| Goal | Visible native | Visible best fixed arm | Hidden native | Hidden best fixed arm |
| --- | ---: | ---: | ---: | ---: |
| 0 | 25% | 50% | 37.5% | 75% |
| 1 | 0% | 0% | 0% | 0% |
| 2 | 100% | 100% | 75% | 100% |
| 3 | 0% | 0% | 0% | 0% |

The best fixed-arm values above use all scenes and are descriptive upper
bounds, not the held-out selection estimates. Goals 1 and 3 were never reached
by any fixed arm or native sampling in either condition. Hidden-goal arms can
prefer goals 0 or 2, but provide no coverage of the other two goals.

| Diagnostic | Visible reach | Hidden reach |
| --- | ---: | ---: |
| G3 rank consistency | .85878 | .69946 |
| G4 log-score gain | .09082 | .22451 |
| Matching fixed-arm binary outcomes | 256/256 pairs | 255/256 pairs |
| Matching complete issued-action hashes | 0/256 pairs | 0/256 pairs |
| Mean fixed-repeat first-chunk L2 distance | .02655 | .02789 |
| Mean between-arm first-chunk L2 distance | .42669 | .42622 |

G3 is defined for only two informative goals in each condition; the two
all-failure goals have undefined correlations. Passing G3 and G4 does not
compensate for failed G1, G2 or goal coverage. The complete action traces are
not exactly deterministic, despite almost identical binary outcomes.

[audit_condition.py](audit_condition.py) independently reconstructs each
goal/scene/arm/repeat grid, success rates, split-half arm selection, G1–G4,
threshold decisions, coverage, repeat counts, finite first-chunk arrays and
artifact hashes without importing the production gate implementation. The
[visible audit](checks/20261029-1918-visible/audit.json) and
[hidden audit](checks/20261029-1958-hidden/audit.json) are mirrored alongside
their raw outcomes, original gate output, contract and first-chunk arrays.
All four scientific input hashes match each Linux audit. Undefined
correlations are represented as null in the audits.

At **19:59 Shanghai**, the authenticated scheduler still reports Running,
one GPU, no worker restart, and the original 175-minute cap. The owned-worker
[progress receipt](checks/worker-progress-195927.json) confirms **245/544
visible-goal push episodes** completed, with the next rollout at step140.
At that check the current output remained buffered. Push-visible later
closed544rows and was preserved before the user-requested stop; push-hidden
did not close. No learner, motor fine-tune or new allocation was launched.

This is one frozen motor on four predeclared goals and four fixed scenes per
goal. It establishes a partial interface-screen result; it does not establish
meta-learning, held-out task transfer or an architecture comparison. The final
motor fingerprint and overall screen summary were not produced before the
user-requested stop. Reach failure alone does not establish the result for push-v3.
