# Cross-arm learner contract

`learner.py` uses NumPy arrays at the runner boundary. All fits and inference run on CPU. The frozen π0.5 runner supplies its clipped native LIBERO control proposals after conversion to the simulator's action convention.

```python
from learner import fit, fit_teacher, load

model, report = fit(calibration, pretrain, joint_support, joint_queries,
                    train_bodies=("Panda", "Sawyer"), reference_body="Panda",
                    seed=0, pretrain_steps=300, joint_steps=400,
                    batch_size=32, support_size=16)
model.save(path)
model = load(path)
action = model.act(proposal, support_commands, support_motion)
cleared = model.act(proposal)
wrong = model.act(proposal, other_commands, other_motion)
teacher = fit_teacher(reference_calibration_episodes, target_calibration_episodes)
oracle = teacher.act(proposal)
```

| Argument | Required fields per episode dictionary |
| --- | --- |
| `calibration`, `pretrain`, `joint_support` | `body`, `episode_id`, `commands[T,7]`, `motion[T,7]` |
| `joint_queries` | `body`, `episode_id`, `proposals[T,7]` |

Each argument is a sequence of complete episodes. IDs are nonempty strings and must identify the original rollout. Keep the four pools and evaluation lineages disjoint, including reset branches and any duplicated π0.5 tapes. Matched runs on distinct bodies may share an ID within a pool. Supply at least two independently drawn pretrain command trajectories per training body and at least one episode per body in each other pool. The learner rejects identical support/query command tapes in pretraining. Use the same frozen reference proposal bank across source bodies for joint queries when possible. Both stages fix a same-body practice episode for each query episode before sampling controls. The model receives only the public numeric fields.

One row is one simulator control interval. `commands[t]` is the normalized, bounded action actually issued at time t in native OSC order (world XYZ, world rotation-vector XYZ, common gripper). `motion[t]` is measured **after** that action from the two boundary observations. Its order is world wrist/TCP displacement XYZ in **metres**, world relative rotation vector XYZ in **radians** from `R_after @ R_before.T`, and change in common jaw aperture in **metres**, positive opening. Use the same observed site, orientation convention, gripper and control interval across bodies. At inference supply only completed transitions. Proposal can be `[7]` or a chunk `[N,7]`. Supplied support is a pair of matching `[K,7]` arrays from one practice episode, shared across the chunk. Supply up to `support_size` completed transitions for matched comparisons. The cleared call uses an empty history.

Stage one predicts the independent query's 7D response, scaled by per-field RMS from pretraining only. Stage two updates the same reader and a zero-initialized six-axis residual head. Its teacher fits an affine Cartesian response per arm to separate simulator trajectories exciting all six axes. It matches reference motion with a damped inverse, clips to the action bounds and preserves the common gripper command. This is supervised inverse calibration. Native rollouts must establish reachability and teacher effectiveness. Held-out-body calibration belongs exclusively to the oracle.

Long support episodes are thinned evenly to `support_size`. For cleared inference the reader returns a zero context, which also occurs with probability 1/8 during joint fitting. Compare retained, cleared and wrong-body actions with the same frozen checkpoint and proposal. The JSON-compatible `report` records calibration coefficients, pair IDs, seed, scales, update counts, parameter changes and retained/cleared errors on a sampled training batch. Benchmark evidence comes from fresh native episodes.

**Verification.** On 2026-11-20 the learner source compiled in memory on macOS Python 3.9.6. Static inspection checked the documented fit, teacher, inference and checkpoint signatures and the order of both fit stages. Learner SHA-256 `a8e5ea273303fa427ea4202f119292c76afec7c9960d2801c9e27a4658fe2dfc`. This handoff covers source checks only. Linux fitting and simulator evaluation remain pending.
