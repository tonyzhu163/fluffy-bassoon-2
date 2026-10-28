# Hidden-task memory benchmarks

Question: can persistent CTM or CMM provide better held-out performance,
generalization, or training efficiency than conventional recurrent memory?
ML10 efficiency is measured separately. ML1 Push already exists; PointMaze
and HalfCheetahVel use the same policy, PPO replay, subprocess workers,
checkpoint recovery, and evaluation implementation as that benchmark.

## Task contracts

`--benchmark pointmaze` uses Gymnasium-Robotics PointMaze v3, an explicit 7x7
S-shaped maze in `metta_env.py`, sparse +1 reward, and a maximum of 500 actions
per physical episode. A successful episode ends immediately and the next
episode starts with the same hidden task. Only the four-dimensional physical
state is exposed; `desired_goal`, task identity, shortest paths, and success
metadata are not policy inputs. Public reward and boundaries remain available.

The pools contain 100 training, 24 fresh training-range, and 24 test
configurations. A task specifies start/goal cells with shortest path at least
10 plus a reset seed fixing their continuous offsets throughout the trial.
Training and fresh training-range configurations use 40 start/goal cell pairs;
test configurations use disjoint held-out pairs. Continuous offsets are also
disjoint. The splits share the layout and can share individual cells; they do
not establish held-out-map or held-out-goal-cell generalization.

Trials have at most K*H real transitions. After all K episodes have finished,
the collector pads the suffix to its declared maximum length. Padding never
occurs between real episodes, never steps the simulator, and cannot influence
a later real action. Trial-end flags prevent bootstrapping through padding;
actor, critic and entropy losses and state-normalizer updates exclude it.
All interaction counters use the real-action mask. Trial padding must not be
misreported as environment interaction. GPU-hours include its actual cost.

`--benchmark halfcheetah-vel` uses Gymnasium HalfCheetah v5 with hidden target
velocity fixed across a trial and 100-step episodes. Reward is
`-|x_velocity - target| - 0.05 * sum(action**2)`. Targets are sampled from
[0,2]: 100 training, 24 fresh training-range, and 24 IID test velocities.
An additional 24 targets in [2,3] test extrapolation, reported separately.
The primary metric is native return, with absolute velocity error as a
companion. The <=0.2 mean-error success diagnostic is not a published metric.

These adapters are METTA-informed independent implementations. The paper
does not specify all simulator/version/map details and its complete source
is not public in the inspected release. Never label their absolute scores
an exact METTA reproduction. The task manifest, map, package versions and
pool hashes are saved in every training contract.

## Comparisons and interpretation

Use persistent CTM T1, full CMM, CMM without long-term tokens, GRU, and
Transformer-XL with the same public inputs, current-state bypass and 64x64
actor/critic heads. Total parameters are matched within the declared 85%-100%
cap. Record ticks, retained state bytes, achieved interactions, optimizer
steps, GPU identity and allocated GPU-hours separately. A shared PPO GRU is
an architecture control; it is not a reproduction of original RL2/TRPO.

`--mode episode` trains an independent memory-reset control. Frozen final
evaluations include normal, attempt-reset, and reward-blind conditions on
paired task and policy seeds. PointMaze records steps, coverage and consecutive
episode Jaccard; interpret exploration before discovery separately from reuse
after discovery. HalfCheetah reports return and error rather than ranking
methods by the diagnostic binary threshold. Longer-trial evaluation retains
the training clock encoding. Report both training-range and OOD results.

Cached replay carries forward history but truncates gradients. Report chunk
length explicitly; compare 100-step, longer chunks and full-trial replay in
a bounded diagnostic before interpreting a delayed-memory failure. CMM's
trace length is not its effective memory limit: recurrence and synchronization
can retain older information even without long-term tokens.

Before a CMM paper-comparison claim, qualify its supervised calibration and
the shared runtime. A successful PPO update proves execution only. Confirm
promising results across training seeds, matched interaction and hardware
budgets, with a positive baseline and trained reset controls. The three
claim axes are alternatives: measure all three and retain negative evidence.
VLA steering follows a verified advantage and informative adaptation evidence.

Sources: [METTA](https://arxiv.org/html/2610.31656v1),
[PointMaze API](https://robotics.farama.org/envs/maze/point_maze/),
[PEARL target-velocity reward](https://github.com/katerakelly/oyster/blob/master/rlkit/envs/half_cheetah_vel.py).
