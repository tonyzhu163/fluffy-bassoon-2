# DSRL on unchanged RoboMimic Can

This is a mechanism pilot using the original paper's low-dimensional Can motor,
not a VLA experiment. Use the released DSRL repository at
`8d21b9cf55459f022c0e769bca1343edd9d30e5a`, DPPO submodule
`86ce51834055c02f9013e60dd4c4275606d82df7`, and SB3 submodule
`10e5d311b5c42fe571ff3eeb72f622ca616a149d`. External sources stay clean.

The motor is the released Can EMA checkpoint `state_5000.pt`, with its own
normalization statistics, 23 observation dimensions, 7 primitive action
dimensions, four-action chunks, and eight deterministic DDIM denoising steps.
Record both input hashes and the loaded motor hash. Freeze all motor parameters.
An episode lasts 300 primitive steps, with unchanged Can dynamics and goals.
The release shifts each raw success reward by -1 and sums over a chunk; preserve
that training reward and separately measure true simulator success at any step.
Also report the release logger's `chunk_reward > -1` success criterion. With
four primitive actions this requires all four rewards in a chunk to equal one;
it can differ from success at any primitive step. Paper-logger comparisons use
that companion metric rather than silently equating the two definitions.

## Trial and claim

An RL²-style recurrent noise controller observes the current normalized state,
previous noise action, previous reward, episode boundary and normalized clocks.
It never observes scene seeds or simulator state. Noise has 28 dimensions and
is decoded by the identical frozen motor used by DSRL-NA.

The first pilot retries a sampled initial scene over K=4 complete episodes.
Only memory changes between attempts at evaluation. Training and evaluation
scene seeds are disjoint. Iid scene resets are a later diagnostic.
No object mass, friction, controller or reward randomization is introduced.

Meta-RL does not require unobserved physics. This experiment asks the narrower
question of learned exploration/retry on held-out initial scenes of one task.
Success here is not evidence of cross-task meta-learning or a new dynamics
distribution. A fully observed task can favor a strong static policy; that is
an empirical outcome, not grounds to introduce hidden conditions after seeing
the results. Longer training or a new task distribution needs a separate plan.

## Comparisons

- Native frozen motor with Gaussian initial noise.
- Authors' DSRL-NA, preserving actor/critic widths, update-to-data ratio,
  noise-critic distillation, and warmup from the released Can configuration.
- A separate DSRL-NA retry-sampling condition, matching the recurrent learner's
  four attempts per initial scene. Keep the original iid-sampling row separate.
- GRU noise controller trained with whole-trial PPO, and a separately trained
  episode-reset controller with the same architecture and interaction budget.
- Frozen normal, memory-reset and reward-blind evaluations on paired scenes.

DSRL-NA in the released code obtains noise bounds from the original action
space [-1,1]; the `action_magnitude=1.5` setting is used by its DSRL-SAC wrapper
but is not passed to the NA algorithm. Preserve the release behavior for the
first reproduction and use [-1,1] for the recurrent controller too. Document
any subsequent correction as a different run. RLinf's current DSRL example is
SAC on LIBERO; it is not the original Can/DSRL-NA implementation. Use the
authors' runtime for this pilot instead of silently replacing that baseline.

Report environment steps (including warmup, adaptation and evaluation), motor
queries, optimizer steps, wall time, GPU time, and actual parameter counts.
The first bounded pilot uses a 100,000-transition replay capacity; its 48,000
primitive-step target cannot fill it, so this changes storage allocation without
changing replay samples. A longer reproduction must revisit capacity before
the buffer wraps. Match deployment interaction budgets when comparing adaptation. Report total
training cost separately; this pilot does not match optimizer or model size.
The first pilot uses stochastic evaluation; a deterministic companion is a
later diagnostic and must be reported separately. Scene-level pairing
and independent training seeds are needed before interpreting advantages.

This is the authors' DSRL-NA learner in a common comparison harness. The harness
plumbs explicit reset seeds (the outer release wrapper ignores `reset(seed)`),
uses terminal observations for warmup replay rather than the release helper's
auto-reset next observation, and continues from the final warmup state rather
than discarding that partial episode with an uncounted reset. These are recorded
driver differences; a full reproduction of the released training driver remains
a separate check. A complete shared-action trajectory is checked against the
released environment wrappers before learning comparisons are admitted.

## Qualification and recovery

Run tests on allocated Linux only. Before training: attest the actual allocated
GPU; execute real Can episodes; verify reset scene identity, sampler parity,
finite actions, frozen weights and old-policy likelihood replay. If Can's
native success is near ceiling, retain the result and do not claim headroom.

Save complete recovery at initial-episode boundaries after all pending optimizer
updates (complete trials for PPO): policy and all optimizers,
SAC entropy and noise-critic optimizer, replay buffer, sampler and RNG states,
and the next reset scene/stream state. Restore by deterministic reset and verify
the next action, simulator transition and optimizer update. A flattened MuJoCo
snapshot alone does not preserve the controller state. The release omits the
noise-critic optimizer from its save list; the local subclass adds it without
changing training. Keep one latest complete recovery plus cited evaluations.

Use `--resume <prior-output-directory>` to continue a completed-boundary recovery
into a fresh output directory at the identical source/learner contract.
`--method dsrl-online --dsrl-reference <trained-dsrl-output>` copies the complete
trained learner, its optimizers and replay independently for each held-out scene.
It adapts using only that scene's K episodes and never carries held-out updates
between scenes. Report its 20 actor/critic/entropy and 10 noise-critic steps per
observed chunk separately from environment samples; this is more optimization
per single-lane sample than the original four-lane training driver.
