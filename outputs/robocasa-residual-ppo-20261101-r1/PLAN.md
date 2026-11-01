# Plan: bounded action-residual PPO on RoboCasa365 (code changes first)

Draft, 2026-11-01. Code changes 1, 2 and 4 written on branch
`residual-ppo` (`src/fluffy_bassoon/rl/action_residual.py`,
`src/fluffy_bassoon/meta_rl/residual_ppo.py`, `tests/test_residual_ppo.py`);
Linux CPU tests pending. Change 3, the RoboCasa365 binding, waits for the
full-weight owner's qualified RoboCasa runner. No job launched. Implements the
steering decision in [PAPER.md](../../docs/PAPER.md) (bounded action
residual) as the matched baseline and history source. The full-weight PPO
comparison stays with its own owner
([preparation](../robocasa-flora-ppo-preparation-20261101-r1/)); this plan
reuses its qualified environment, pi0.5 loading and Flora scorer contract
and does not touch its worktree or jobs.

## Facts from the existing RoboCasa365 setup

From [robocasa365_pi05_eval.yaml](../../configs/native_ppo/robocasa365_pi05_eval.yaml)
and the [loading handoff](../robocasa-flora-ppo-preparation-20261101-r1/PUBLIC-PI05-LOADING-HANDOFF.md):

- Policy: public `pi05_robocasa365_pretrain_human300`, RLinf HuggingFace
  backend, 10 flow steps, chunk H50 executed whole.
- Action: 12-D, in order eef position 3, eef rotation 3, gripper 1, base
  motion 4, base mode 1. Gripper and base mode are binarized at 0.5 to ±1.
- First task OpenDrawer, horizon 750 (15 chunks per episode); fallback
  ScrubCuttingBoard, horizon 1200.

## Design

- **Residual:** one 6-D vector per chunk on the eef position and rotation
  dimensions, added to every step of the frozen pi0.5 chunk after
  unnormalization and before environment clipping. Gripper and base
  dimensions are untouched. The policy output u maps through
  `δ = b ⊙ tanh(u)`.
- **Bound b (declared before any learning):** 0.25 × the per-dimension
  action std from the checkpoint's own norm stats, in environment units. A
  second bound of 0.5 × std runs only if 0.25 learns nothing while
  full-weight PPO improves on the same task. Record both prospectively; no
  other bound search.
- **Policy and critic:** small MLPs on frozen inputs: 16-D proprio state, the
  base chunk's first action and mean, and pooled pi0.5 prefix features. Mean
  head zero-initialized, so step 0 reproduces pi0.5 exactly. Initial std
  well inside the bound. No memory.
- **PPO:** chunk-level decisions, one terminal reward per episode (simulator
  success or Flora's normalized score, as in the full-weight plan), GAE and
  PPO loss from `src/fluffy_bassoon/meta_rl/ppo.py`. Same Flora revision,
  mapping, camera and frame clock as the full-weight arm.
- **History export:** per chunk, features, base chunk, sampler noise seed,
  executed residual, reward, done, episode start. Write through
  `vla_history.save_dataset`, with the residual in the control field, so the
  plain-decoder meta-learner trains on it unchanged (`noise_dim = 6`, the
  adapter applies the bound).

## Code changes

1. `src/fluffy_bassoon/rl/action_residual.py`: bound from norm stats,
   `tanh` map, application to the 12-D chunk, with gripper and base
   untouched. CPU tests: a zero residual is the identity, outputs stay in
   bound, dimension mapping is correct.
2. `src/fluffy_bassoon/meta_rl/residual_ppo.py`: Markov Gaussian
   actor-critic and PPO update over chunk decisions, reusing `ppo.py`. CPU
   tests on a toy environment.
3. RoboCasa365 driver (new script under `scripts/`): frozen pi0.5 rollout
   through RLinf with no gradient, the residual applied, terminal reward from
   the simulator or Flora, and history export. Not
   `vla_noise_ppo.py`, which is trial-wise recurrent noise PPO on GTrXL for
   LIBERO.
4. History controller adapter: `FrozenHistoryController` outputs passed
   through the same bound map for deployment.

## Gates before any GPU learning

1. CPU tests pass on Linux.
2. Zero-residual equivalence: one OpenDrawer episode with fixed seeds gives
   the same executed actions with and without the residual path.
3. The full-weight owner's Flora agreement and headroom gates pass on the
   chosen task (shared, not repeated).
4. Then residual PPO with simulator reward and with Flora reward, at a
   budget matched to the full-weight arms.
