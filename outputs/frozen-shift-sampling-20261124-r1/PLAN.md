# Sampling lane: do other policy-noise draws succeed, and can a source-fitted critic pick them?

Registered November 24 before any result in this directory. [Program plan](../frozen-vla-proposal-shift-20261124-r1/PLAN.md) · [Harness](../frozen-vla-proposal-shift-20261124-r1/harness.py) at `5fd6c426`. Frozen π₀.₅ weights and motor parameters never change.

## Questions

1. **Headroom.** On Kinova3 drawer layouts 14, 15, 17, 36, 37, 38, is the timeout a property of the start or of the one registered noise stream? How often does an independent policy-noise stream succeed with everything else fixed?
2. **Critic.** If some draws succeed, does a progress critic fitted on source arms only, reading public observations and issued commands, pick better proposals among K=4 per replan than a random index among the same four?

## Prediction

Under the program's proposal-failure hypothesis some of the 48 independent-noise episodes earn reward, and the per-start frequency is strictly between 0 and 1 on at least one failed start. The critic then beats random-of-K on paired starts. My prior for the critic is weak: it cannot see the scene.

## What counts against it

- Headroom: 0/48 independent draws succeed. Then sample choice cannot rescue these starts and the critic arm is moot. If instead known successes drop under fresh noise as often as failures rise, noise only reshuffles outcomes.
- Critic: on the same starts it equals random-of-K (paired layout interval covers 0), or both equal K=1. Offline: held-out source ranking no better than its chance level (1/2 for ordered pairs, 1/4 for one-of-four windows).
- Decision comparisons are matched baselines (random-of-K, K=1, the harness `raw` and `repeat`) and chance levels. No other numeric gate.

## Prior work checked

- [V-GPS](https://arxiv.org/abs/2410.13816) re-ranks K action samples of a frozen generalist with a language-and-image Q-function trained by offline RL on large robot datasets. Difference here: the critic is a small proprioceptive progress regressor fitted on source-arm episodes of this one task, the test body is absent from its data, and the comparison includes a random index among the same K and a measured base rate of untreated draws. Not a novelty claim for sample-and-select.
- Embodiment-classifier guidance, [arXiv 2610.22576](https://arxiv.org/abs/2610.22576). The repository's note ([selector README](../frozen-vla-selector-20261124-r1/README.md)) describes it as steering frozen π₀ with successful target rollouts. I did not read the paper in this session. Difference as far as that note goes: this lane uses no target-arm success, changes no denoising step, and only selects among complete samples.
- In-repo: [retry headroom](../libero-retry-headroom-20261119-r1/RESULT.md), the two-draw result in FINDINGS (74 both, 20 one, 206 neither of 300), the [factory](../frozen-vla-factory-20261123-r1/RESULT.md) 2/6–4/6 across jobs, [review rows](../../docs/literature/vla-steering-review.md) on sample-and-select.

## Protocol

Evidence classes are labelled in RESULT.md. Only fresh native reward is task success; none can be produced on this VM.

1. **Archived bound (CPU, archived native outcomes).** `mine.py` walks every JSON under `outputs/` for Kinova3 `libero_goal/3` BASE untreated episodes, grouped by layout, job directory, noise digest and opening-proposal digest. Report per layout: untreated episodes, successes, distinct noise streams, distinct opening proposals. State what this bounds about per-start success frequency (exact binomial upper limit when all fail) and what it cannot (same noise stream repeated is not an independent draw).
2. **Headroom panel (code only here).** `headroom.py`: for each failed start, the untreated stream plus eight `noise-k` adapters whose only change is the policy-noise array at every replan (stream `k`, derived from the start's candidate stream so it is fixed per start and independent across `k`). Two `noise-k` on each of 16, 31, 34, 35, 39. 48 + 10 episodes plus one untreated per start. Report per-start frequency with exact binomial intervals, opening-proposal digests and first divergence.
3. **Critic (CPU fit now).** Public features: end-effector displacement from the reset pose and the commanded gripper state. Label: fraction of the episode elapsed at each frame of a source success. Ridge on quadratic features. A proposal is scored by the critic at the state predicted from its first five commands with the 15-control calibration response. Source data: every Panda and UR5e drawer-task trajectory archived in the repo (inventory in RESULT.md); no Kinova3 or IIWA frame is used for fitting. Held-out source ranking, leaving one episode out: (a) ordered-pair accuracy for frames ≥25 controls apart, chance 1/2; (b) top-1 of the executed five-control window against three windows from other times of the same episode, chance 1/4; (c) successes against failures by final critic value, counts only.
4. **Adapters.** `critic-k4` (four draws, critic choice), `random-k4` (same four draws, seeded random index), `single-k1` (first of the same candidate stream, one draw). All three leave observation, prompt and command untouched. The harness `raw` and `repeat` are the untreated references.
5. **CPU qualification.** Tests on the harness fixture: only noise digests, candidate count and chosen index move; `critic-k4` and `random-k4` see identical candidates; `noise-k` streams are distinct across `k` and repeat for a start; the critic artifact reloads to the reported weights.
6. **Launch.** `run.sh <pin> <attempt>` for one Baige RTX 4090: headroom first, then the panel on development starts for the three adapters. Not run here.

## Stop

About 50 minutes on this VM. No GPU, no Baige.
