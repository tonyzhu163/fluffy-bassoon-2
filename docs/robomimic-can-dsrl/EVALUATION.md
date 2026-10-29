# Can DSRL / RL² evaluation

No trained-method comparison is qualified yet. The Can runtime qualification and native motor evaluation below have passed.

The first allocated runtime check used source
`c37ec48143bf2eadb81539f2b8748f64e45f2eab` on Baige
`job-9pi2eysq5lcx`. All 15 focused CPU tests passed in160.85seconds and the
exclusive RTX4090/CUDA attestation passed. Can initialization then failed on
RoboMimic0.3's missing legacy `mujoco_py` import. No Can episode, learner update,
trajectory parity or recovery continuation ran, and native success is unknown.

Raw evidence is preserved in
[`qualification-r2`](../../outputs/robomimic-can-dsrl-20261029-r1/qualification-r2/).
Runtime repair provenance and new submission receipts are in the parent run
directory. Current job state belongs in `outputs/coordination.md`.

The GTrXL qualification on Baige `job-pnesyoazimd0`, source
`c1032117b953acd94592336629a1d28c4f563d62`, passed all69 focused CPU tests
in132.51seconds inside its allocated RTX4090 container. The complete CUDA
allocation probe and real tensor/own-PID worker attestation also passed. These
check architecture equations, replay, resets and recovery contracts; they do
not establish Can simulator parity, learner performance or a memory benefit.
Those real-simulator and learning checks remain pending at the14:37 Shanghai
artifact audit. Evidence: `qualification-r4/` in the parent run directory.

At14:43 Shanghai the same qualification completed successfully: released motor
sampler parity and every primitive observation/action/reward/horizon in the
1,800-step wrapper parity check matched exactly. PPO likelihood replay error
was zero. PPO and DSRL continuation reproduced the next sampled actions,
simulator data and optimizer updates; the frozen motor stayed unchanged.
The full four-lane/K4/four-epoch GTrXL rollout/replay/update took71.907s,
including10.469s for PPO optimization, at1.548GiB peak allocated CUDA memory.
The released DSRL20+10 update schedule at batch256 took0.461s.

Native Gaussian motor evaluation completed32 episodes on8 initial scenes
with4 attempts each:15/32 successes (46.875%, descriptive episode average;
these are not32 independent scenes), in140.799s. This supplies headroom for
steering. The original serial pilot-r2 refused both training targets because
its conservative timing estimate exceeded80minutes. No pilot learner
training ran. This refusal is a budget decision, not a failed runtime test.

The proposed fresh pilot-r3 preserves all four methods, one training seed,
28,800 common primitive steps and every8-scene/K4 evaluation condition,
using two balanced4090 lanes with a50-minute cap. Its conservative total
allocated charge is100GPU-minutes plus6minutes reserved for the earlier
failure and12minutes conservatively reserved for qualification-r4, below the original120GPU-minute cap.
Online DSRL fine-tuning is deferred; this pilot compares frozen DSRL with
memory-based adaptation and its trained reset control, not with test-time
weight updates. Eight-card follow-on remains conditional on a complete pilot.
