# Does frozen SmolVLA reproduce the Kinova3 drawer-and-bowl gap on CPU?

Registered November 24 before any policy or simulator episode in this directory. Lane "Second motor" of the [proposal-shift program](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). Base revision `092d3148`.

## Question

Frozen π₀.₅ times out on Kinova3 `libero_goal/3` (open the top drawer and put the bowl inside) starts that source arms solve. Does a second, independently trained frozen motor show the same target-arm gap on the same released scenes? If it does, the proposal-shift adapters (state, prompt, policy noise, proposal choice) can be developed with native reward on CPU.

## Prediction and what counts against it

Prediction: SmolVLA solves some Panda starts and fewer of the same starts on Kinova3 under matched resets and policy noise.

Against it: (a) Panda success is not reproduced with official preprocessing. Then nothing is interpreted. (b) Kinova3 native success on matched starts is not below Panda by more than the difference between two Panda runs that differ only in policy noise. With a handful of starts this is a small exploratory panel and the result is reported as counts with an exact interval, not a gate.

## Prior work checked

- [Camera roll](../frozen-vla-factory-20261123-r1/view/README.md) and [rendered arm](../frozen-vla-factory-20261123-r1/rendered-appearance/RESULT.md): the same SmolVLA checkpoint, CPU forward passes on **standard LIBERO Panda** frames only. No native SmolVLA reward exists in this repository, on any arm.
- [Physical arm interface](../libero-crossarm-20261120-r1/SIM.md): `ArmEnv` loads a released Panda row, transfers the scene and solves tool-pose IK for the other arm. Reused unchanged.
- π₀.₅ Kinova3 failures: layouts 14, 15, 17, 36, 37, 38. Known successes 16, 31, 34, 35, 39.

## Protocol

- Simulator: LIBERO-Pro2.0 `e5500c690c1498eca1b7050350654964190cdf70`, robosuite 1.4.0, MuJoCo 3.2.3, BASE `libero_goal/3`, 300 controls, Mesa software EGL. Arms through `ArmEnv` with PandaGripper on every arm.
- Motor: `lerobot/smolvla_libero` at `a24f1f380461b665158d52acd140152c79e8b36d`, LeRobot 0.4.4, strict load, weight SHA-256 checked, 450,046,176 parameters counted and all `requires_grad=False`, eval mode, CPU. Official preprocessing: LeRobot `preprocess_observation`, `LiberoProcessorStep` (180° image flip, position + axis-angle + finger state), released pre/post processors, 512-pixel resize, 10 denoising steps, 50-control chunks executed in full (`n_action_steps=50`), original instruction.
- Starts: layouts 14, 15, 17, 36, 37, 38 first, then 16, 31, 34, 35, 39 if CPU time allows. Reset seed 42710300 + layout. Policy noise for replan `i` is `torch.Generator(seed=noise_seed).normal` drawn in sequence, with `noise_seed = reset seed` for the primary branch. Arms share the reset seed and the noise sequence.
- Order: Panda first. If no Panda start succeeds, debug preprocessing and stop interpreting. Then Kinova3 on the same starts, then IIWA or UR5e if time allows.
- Untreated repeats per arm and start: (1) identical inputs and noise, which measures CPU determinism; (2) a fresh noise seed (`reset seed + 1000003`), which measures how much sampling alone changes the outcome. The gap is read against (2).
- If the drawer task shows no gap, up to three other `libero_goal` tasks, chosen before looking at Kinova3 outcomes: 4 (bowl on cabinet), 1 (bowl on stove), 8 (bowl on plate).
- Report native success and control count per arm, start and branch, and the Panda minus Kinova3 difference with an exact paired sign count. Record prompt, state and image digests, noise digests and consumed commands per episode.
- Adapter inputs are public observations and issued commands. Reward and success are audit data read only to end the episode and score it.

## Evidence labels

Fresh native reward: `evidence/*.json` episodes from this VM's CPU. CPU fixture: `test_harness.py` (no simulator, no weights). No offline or archived-replay evidence is planned.

## Budget and stop

CPU only, about 50 minutes of wall time including setup. Stop when the time is used and report exactly which cells ran.
