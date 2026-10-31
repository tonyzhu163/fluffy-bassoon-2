# Frozen pi0.5 constant-noise headroom pilot

`meta_rl.vla_noise_headroom` evaluates one predeclared training or validation
task from the hash-checked frozen LIBERO task split. Test and excluded tasks
are rejected. The released OpenPI pi0.5 motor stays frozen: native horizon ten,
32 initial-noise coordinates, ten flow steps, execute ten actions per decision.

For each of initial states zero and one, it runs two independent native
full-Gaussian episodes and eight fixed noise arms twice. The fixed bank is zero
plus seven seeded Uniform[-1,1] vectors. An arm remains constant throughout its
episode and is repeated over the native prediction horizon. All repeats use
the same declared scene state/seed. The complete plan is 36 episodes, at most
14,400 executed primitive steps, with reset settling counted separately.

Whole public RGB observations from both cameras, robot state, initial noise,
executed actions and real rewards are retained in compressed episode traces.
The arm vectors, plan and hashes are written before any rollout. Progress is
saved after every completed episode. Same-noise motor qualification and actual
renderer/motor GPU binding checks precede collection. Optional native prefix
features are qualified only initially; repeated feature extraction is omitted.

Analysis selects an arm using repeat zero and scores its separately executed
repeat one. It reports all eight exploration episodes as selection cost,
success/trajectory repeatability, a constant-arm reference and differences
between the two scenes. Predeclared search and explore-then-commit bandit curves
reuse only observed complete episodes and count each actual attempt. Native
comparisons exist only up to the two native attempts actually measured.

These repeats test replay reliability on the same scene and seed. They do not
establish generalization or inadequacy of an observation-conditioned static
policy. A negative finite constant-arm result does not rule out a controller
that changes steering with observations or between action chunks. One task,
two scenes and four native episodes provide limited uncertainty information;
the descriptive episode-level Wilson intervals do not quantify task transfer.
