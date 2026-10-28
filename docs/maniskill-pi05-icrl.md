# ManiSkill pi0.5 in-context RL proof of concept

The extension asks one question: does a CTM have the capacity to do in-context
RL? This protocol tests it on a real manipulation loop. A frozen pi0.5 motor
executes; a small learner (CTM, Transformer-XL or GRU, matched) must recover
from a hidden per-trial perturbation using reward alone, with frozen weights
at test time. It follows VLRLA (arXiv 2610.19517) in using a frozen VLA, a
distillation warm start and multi-attempt trials, and differs in two ways:
the perturbation makes reward necessary, and a trial-PPO stage follows the
warm start. Option B (a hidden target object) is a later, separate task.

## Motor and scene

- Simulator: RLinf `PutOnPlateInScene25Main-v3` (ManiSkill3 Bridge WidowX
  digital twin), RLinf source `e1a82eb`, 80 steps, chunk 5, 4 denoising steps.
  Training objects 0-15 and overlays 0-15; objects 16-24 are held out.
- Primary motor: `RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise`, revision
  `a0cbb89804934639f2b348afc7206de3a731a955`, global step 150, weights SHA256
  `0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1`.
  Its PPO log reports 43% rollout success at step 0 and 89% at step 150.
  A strong motor keeps "the motor fumbled" from dominating the reward signal.
- Secondary motor: `RLinf-Pi05-ManiSkill-25Main-SFT` (weights SHA256
  `9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89`, RLinf
  reports 40.1%), for a noisier-reward replicate.
- Both are frozen. RLinf eval-mode sampling with seeded initial noise; no
  training of any motor parameter.

## Hidden perturbation

Each trial draws a hidden angle theta. After RLinf's `prepare_actions`, the
translation command `world_vector[:2]` is rotated by theta about the vertical
axis before `env.chunk_step`; z, rotation and gripper are unchanged. Theta is
fixed for all K attempts of the trial; each attempt resets to a different
scene (object, overlay, pose) drawn from the split. Theta never enters any
learner input or log the learner reads.

The learner applies a correction phi in the same place, so the executed
rotation is theta + phi. Rotations about one axis commute, so the closed-loop
motor, scene and success depend only on the net angle. This is exact, and it
is what makes outcome caching valid (below).

Why rotation and not gain: audit F6 showed a translation gain is identifiable
from commanded versus observed motion, so a learner can adapt without reward.
Here the learner never sees images, proprioception or end-effector motion, so
reward is its only evidence about theta. The motor does see images and may
partly compensate within an attempt; that is its own closed loop, measured by
the headroom screen, and cannot carry information across attempts.

## Learner contract (stage A1: one correction per attempt)

- Action: phi from a fixed grid, chosen once per attempt.
- Input per attempt: previous phi, previous latched success reward, attempt
  end flag and public clocks (attempt index / K). Nothing else.
- Trial: K attempts (K = 8 initially), return summed over the trial, gamma = 1,
  GAE lambda = 1 (the compact bandit recipe). Learner state is carried across
  attempts and reset only at trial start.
- This is a structured continuous-armed bandit with a real manipulation loop
  in it. The meta-learned strategy must exploit structure a generic bandit
  algorithm ignores: success peaks at net angle 0 and falls with |theta + phi|,
  and theta follows the training prior.

Stage A2, per-chunk residuals on the motor's actions (VLRLA-bridge-like), is
deferred until A1 shows in-context learning. It needs live simulator RL.

## Outcome cache

Because only the net angle matters, collect success for each (scene, net
angle, repeat) once with the frozen motor on one GPU, then run meta-training
on CPU by sampling Bernoulli outcomes from the table. Each cached row records
episode id, object, overlay, pose, net angle, repeat seed, success at any step
and at step 80, and the first success step. It also records graded outcomes
from RLinf's own evaluation: whether the object was grasped (and held for five
steps), the first grasp step, and the object-to-plate distance at step 80 and
its minimum over the episode (`OUTCOME_FIELDS` in `rl/maniskill_pi05.py`).
Graded fields are not learner inputs in this protocol; feeding any of them to
a learner is a separate protocol. Final evaluation runs live in the simulator
with the learner in the loop, which checks the cache was not a shortcut.

## Stages and gates

1. **Motor gate.** Per-object success for both motors at net angle 0
   (`scripts/diagnostics/maniskill_pi05_object_gate.py`). Exclude objects the
   primary motor cannot place reliably from the choice of training scenes, and
   report that exclusion.
2. **Headroom screen.** Success versus net angle on a coarse grid
   (for example 0, ±15, ±30, ±45, ±60, ±90 degrees) over a fixed scene sample.
   Proceed only if success at 0 is high, falls clearly by the largest angles,
   and is roughly unimodal. Freeze the theta range, phi grid and K from this
   screen before any learner training.
3. **Cache collection.** Training scenes x net-angle grid x repeats, and a
   disjoint held-out-scene cache for development checks.
4. **AD warm start.** Teacher: posterior sampling over theta using the cached
   success curve (a strong structured strategy). Train each learner to imitate
   the teacher's phi given the public history, over ordered teacher trials.
5. **Trial PPO.** `meta_rl/ppo.py` / `train.py` meta mode on the cache, from
   the warm start, with a small anchor toward it. CTM uses `persistent_ctm`
   with bounded synchronization and per-observation decay (audit F2/F3) unless
   the compact T8 results argue otherwise.
6. **Live evaluation.** Frozen learner weights, closed loop, held-out objects
   and held-out theta values, success at attempts 1..K.

## Comparisons and controls

- Learners: CTM, Transformer-XL, GRU with identical inputs, matched on
  parameters and retained state; three or more training seeds each.
- Reference strategies: zero correction, oracle phi = -theta, independent-arm
  Thompson sampling over the phi grid, structured posterior sampling (the
  teacher).
- Memory-wiped lesion: learner state reset every attempt. It should stay flat.
- Reward-blind lesion: reward input zeroed. It should stay flat; improvement
  here would mean information about theta leaks through another input.
- CTM ticks (T1/T4/T8) only after the compact T8 results.

In-context RL is shown if success rises across attempts on held-out scenes
and angles, the two lesions do not rise, and the gap to the oracle closes.
The CTM capacity question is answered by the matched learner comparison.

## Environment

Python 3.11, uv. RLinf source: commit `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566`
(thu2 used a tarball, SHA256
`f12bc3c4cfdd17d4269b2d594038c49353528f4e1a8290c10f9ed502715be18a`; Baige used
the codeload archive, SHA256
`d9e3d14aa19a9b7884defcebbe89497e279162a610e61a1d16ffabcfd1eb28f7`, with all
2,228 blobs verified against tree `0bb0da449ff13778084839c22509fee2119e80cb`).
Baige package freeze SHA256
`4bacb95fff3d6958e0679e0b5001b5d93d2fa4e3d2e70f8536d8114b14f21105`; there,
NumPy-2-only transitives are pinned to ml-dtypes 0.5.1, numpydantic 1.6.11,
pandas 2.2.3 and OpenCV 4.11.0.86.

`requirements.txt`:

```
-e <RLinf checkout>[embodied]
nvidia-nvcomp-cu12
jax[cuda12]==0.5.3
orbax-checkpoint==0.11.13
tyro==1.0.13
rlinf-openpi==0.1.1
mani-skill==3.0.0b22
sapien==3.0.1
```

`overrides.txt` (rlinf-openpi pins torch 2.7.1; thu2's driver 570.86 needed
CUDA 12.8 wheels, so `--torch-backend cu128`; choose the torch build for your
driver):

```
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
torchcodec>=0.11,<0.12
setuptools>=69.5.1,<75.9
datasets==3.6.0
sympy>=1.13.1
tensorflow-metadata<=1.17.3
protobuf>=6.33.5,<7
numpy<2
```

```bash
uv venv ENV --python 3.11
uv pip install --python ENV/bin/python --torch-backend cu128 --override overrides.txt -r requirements.txt
```

The thu2 check printed `2.11.0+cu128 12.8 0.5.3 3.0.0b22` (torch, CUDA, jax,
ManiSkill). thu2's environment is `envs/rlinf-pi05-maniskill-e1a82eb-r1`; its
freeze is in `outputs/lycoreco-rl-pi05-maniskill-setup-20261028-r1/env-freeze.txt`.
Installed LeRobot/LIBERO integration tests need the matching LeRobot runtime.
Standalone recurrent cores are separately qualified in the native environment;
`test_libero_observation_integration.py` remains a required, separate suite.

## Assets

| Asset | Source | Check |
| --- | --- | --- |
| RL motor (primary) | HF `RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise` rev `a0cbb89804934639f2b348afc7206de3a731a955`; weights `checkpoints/global_step_150/actor/model.safetensors`; norm stats `checkpoints/global_step_150/actor/assets/global_step_150/meta/norm_stats.json` | weights SHA256 `0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1` (HF LFS oid), 8,537,972,300 bytes |
| SFT motor | HF `RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`; norm stats `physical-intelligence/maniskill/norm_stats.json` | weights SHA256 `9877be1633a9bc89834d00ef890e1a4cd34a91c155d710b97c92fbab8a5caa89` |
| Task objects | HF dataset `RLinf/maniskill_assets`, `carrot/**` | set `MANISKILL_ASSET_DIR` to its root |
| ManiSkill scene and robot | `bridge_v2_real2sim` and the WidowX robot assets | set `MS_ASSET_DIR` (holds `data/tasks`, `data/robots`) |
| PaliGemma tokenizer | HF `RLinf/openpi_tokenizer` (`big_vision/paligemma_tokenizer.model`) | set `OPENPI_DATA_HOME` to its root if GCS is unreachable |
| PhysX | `sapien-sim/physx-precompiled` release `105.1-physx-5.3.1.patch0` `linux-so.zip` into `~/.sapien/physx/105.1-physx-5.3.1.patch0` | SAPIEN fetches it on first use when GitHub is reachable |

Through hf-mirror, set `HF_HUB_DISABLE_XET=1`; Xet transfers stalled on thu2.
RL and SFT norm-stat values are identical; their bytes differ by formatting
(`930b816b...` vs `67a6e551...`). Pass each checkpoint its own file.

## Launch shapes

Run inside a single-GPU `srun --ntasks=1` step or a Baige container.

Common environment: `FLUFFY_GPU_EXECUTION=slurm` (or `baige`) `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false`
plus the asset variables. Each script requires a clean worktree at
`--source-commit` and runs `scripts/[train]gpu_probe.py` itself. Batch sizes
must divide the row count.

```bash
P=ENV/bin/python; RL=.../RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor
COMMON="--rlinf-root RLINF --source-commit COMMIT"
# gate smoke, then full gate (500 episodes)
$P scripts/diagnostics/maniskill_pi05_object_gate.py $COMMON --model-path $RL \
  --norm-stats $RL/assets/global_step_150/meta/norm_stats.json \
  --expected-model-sha256 0adb5b40f064760ff245f53ae0a4e078b86680b65f77d1f5991a173543edd5a1 \
  --output-dir OUT/gate-rl-smoke --episodes-per-object 2 --num-envs 10
$P scripts/diagnostics/maniskill_pi05_object_gate.py ... --output-dir OUT/gate-rl --episodes-per-object 20 --num-envs 50
# headroom screen: 16 training objects x 4 scenes x 11 angles = 704 rows
$P scripts/diagnostics/maniskill_pi05_rotation_cache.py $COMMON --model-path $RL \
  --norm-stats ... --expected-model-sha256 ... --output-dir OUT/headroom-rl \
  --scenes-per-object 4 --num-envs 64
```

On thu2 the gate used `--mem=18G` and 8 CPUs; peak GPU memory is recorded in
each `summary.json`.

## Implementation pieces

| Piece | File |
| --- | --- |
| Shared model/env settings, scene ids, rotation hook and batch rollout | `src/fluffy_bassoon/rl/maniskill_pi05.py` |
| Gate for either checkpoint (explicit `--norm-stats`) | `scripts/diagnostics/maniskill_pi05_object_gate.py` |
| Headroom screen and cache collector | `scripts/diagnostics/maniskill_pi05_rotation_cache.py` |
| Cached rotation trial task for the compact trainer | `src/fluffy_bassoon/meta_rl/maniskill_rotation.py` (new), registered in `environments.py` |
| Posterior-sampling teacher and AD warm start for compact tasks | `src/fluffy_bassoon/meta_rl/` |
| Live closed-loop evaluator with learner in the loop | `scripts/diagnostics/` (new) |

Run histories, job IDs and results belong in `outputs/` and the method's
`EVALUATION.md`, not here.
