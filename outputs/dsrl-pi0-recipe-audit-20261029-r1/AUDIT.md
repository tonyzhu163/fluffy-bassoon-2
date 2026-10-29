# Published DSRL π0 recipe versus the current short-history pilot

Read-only audit; no model computation or GPU launch. Official source pinned to
`7f48937d4553e95244cd81c79236a3256df80597`, tracked files clean. Exact file
SHA-256 values and numbered excerpts: [source-evidence.json](source-evidence.json).
Our comparison snapshot is source `42fd5176f484aa6b884e52881a2a3021ff18e258`;
completed outcomes/costs were independently audited in
[the prior evidence audit](../libero-rlinf130-evidence-audit-20261029-r1/AUDIT.md).

## Main implication

- The published LIBERO result concerns a weaker, differently trained motor:
  public `pi0_libero`, fine-tuned on 40 tasks, targeting an unseen LIBERO90 goal.
  Reported improvement: approximately 20%→100% after approximately 10,000
  “online samples,” averaged over 3 seeds. Keep that author unit; do not silently
  call it10,000 primitive actions. [Paper §5.5/C.5](https://arxiv.org/html/2506.15799v2#S5.SS5)
- The paper's other simulated π0 task is AlohaTransferCube. Its public motor
  was fine-tuned on 50 demonstrations of that task. Its π0-DROID Franka results
  are toaster 5/20→18/20 and spoon 15/20→19/20, after 80/65 episodes respectively.
  These are distinct from its non-VLA diffusion-policy experiments.
  [Paper §5.5/C.5](https://arxiv.org/html/2506.15799v2#A3.SS5)
- Our RLinf π0.5 release was trained on all 130 original LIBERO tasks. Clean
  source 24/24 and native Plus/Pro development 8/8 each supply weak improvement
  headroom; they do not reproduce the paper's unseen-goal condition. Our
  one-goal K4 collection cannot establish a generally competent source teacher.
- The observed ceiling and short training schedule are protocol/backbone
  differences. They are not, by themselves, evidence that noise steering is
  broken or that meta-learning cannot work.

## Released simulation recipe, with units resolved from code

All source links below pin the exact audited commit.

| Item | Original LIBERO DSRL | Original Aloha DSRL | Our current clean/Plus source pilot |
| --- | --- | --- | --- |
| Task | `libero_90:57`, cream cheese→tray | `gym_aloha/AlohaTransferCube-v0` | Goal family `libero_90:47`, cream cheese→basket; clean plus two Plus source variants |
| Motor | `pi0_libero` | `pi0_aloha_sim` | Released RLinf π0.5 LIBERO130 fullshot SFT |
| Motor noise horizon / execute before replan | 50 /20 | 50 /50 | 10 /10 |
| Episode cap, primitives | 400 | 400 | 280 |
| Batch | 256 | 256 | 32 |
| Warmup condition | replay length **>500 query chunks** | replay length ** >1,000 query chunks** | 0; random SAC actor acts on attempt1 |
| Gradient updates per new chunk | 20 | 20 | 20 |
| Script stopping counter | 500,000 **optimizer updates** | 3,000,000 **optimizer updates** | Four completed episodes per independent source trial |
| Latent coordinate | 32D actor output repeated across 50 native slots | Same | Same32D coordinate repeated across 10 native slots |
| Learned latent bounds | [-1,1] | [-2,2] | [-1,1] |
| Discount | .999^20 per replay transition | .999^50 | .999^(actual primitives in chunk) |
| Steering network | Fresh pixel SAC, 3×128, four conv layers,10 critics | Same | Same official pixel SAC architecture; separately recorded4104D frozen motor features for students |
| Actor/critic/temperature learning rates | 1e-4 /3e-4 /3e-4 | Same | Same |
| Target update τ | .005 | .005 | .005 |
| Target entropy | auto, -32/2=-16 | Released script0; paper table says -16 | auto=-16 |
| Initial scene population | `env.reset()` for each trajectory | `env.reset()` | Explicit initial-state index repeated within each K4 trial |

Recipe sources:

- [LIBERO launcher lines17–34](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_libero.sh#L17-L34)
  and [Aloha launcher lines17–35](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_aloha.sh#L17-L35).
- [Task/motor selection, fresh actor and replay capacity lines111–163](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_sim.py#L111-L163).
- [Pixel SAC hyperparameters lines29–52](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/launch_train_sim.py#L29-L52),
  [fresh parameter initialization and auto entropy lines143–225](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/pixel_sac_learner.py#L143-L225).
- [Collection/update counter lines97–153](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L97-L153).
  Collection completes an episode; then SAC performs `new_chunks*20`
  optimizer updates. Weights are fixed during that episode. `i` advances only
  on updates, so `max_steps` names a gradient budget, with possible final
  episode-burst overshoot. It does not name a primitive-action budget.
- [One replay sample per queried noise action, lines163–190](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L163-L190).
  The `num_online_samples` log is replay length; `env_steps` is logged separately.
  Do not substitute one for the other or assume all paper axes share a unit.

## Noise, reward and update details that matter

- Warmup: while no optimizer update has occurred (`i==0`), draw a Gaussian 32D
  coordinate and repeat it over all 50 slots. There is no supplied steering
  checkpoint. After warmup, a newly initialized SAC actor learns from replay.
  [Training lines193–244](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L193-L244)
- Initial **native evaluation** instead draws independent `[1,50,32]` Gaussian
  noise. Tied Gaussian exploration and native sampling are distinct controls.
  Post-update evaluation calls stochastic `sample_actions`, not actor mean.
  [Evaluation lines310–350](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L310-L350)
- Sparse source labels are -1 per **query chunk**, with final 0 on success;
  successful final transition gets mask 0. Timeout/failure keeps mask 1 and
  bootstraps. This is not a dense primitive-reward sum.
  [Reward relabel lines270–287](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L270-L287)
- The critic target consumes the stored discount; it does not multiply an
  additional gamma. Our actual-length discount agrees on full nonterminal
  chunks and intentionally differs on a partial final timeout chunk.
  [Critic target lines14–28](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/critic_updater.py#L14-L28)
- Official LIBERO seeds its environment once, then calls `env.reset()` each
  episode without applying a fixed state. Different policy/noise seeds on an
  identical restored simulator state do not establish distinct placements.
  Count cases using actual initial object/robot states. Exact case-generation
  semantics of our pinned simulator must be audited separately.
  [Environment seed lines38–45](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_sim.py#L38-L45),
  [reset lines200–203](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L200-L203).

## What to change before a larger source corpus

1. Qualify learning on a development population with actual diverse cases and
   observed native failures. Generate cases before outcomes; record placements,
   seed, perturbation identity and separate validation cases. Do not select
   individual final-test failures to train on. More successful copies of the
   same basket goal do not create a learning signal or new-goal generalization.
2. Run a bounded teacher-learning qualification with the published batch 256,
   warmup >500 chunks and UTD20 when making a DSRL reproduction comparison.
   Keep K1/2/4/8 as a separate adaptation evaluation axis. With400/20≤20
   chunks per episode, even maximal-length trajectories require at least 26
   episodes to pass warmup. K4 cannot start learning under that recipe.
3. If using warmup 0/batch 32/K4 for fast source histories, name that departure
   explicitly. The source actor is a final candidate, not an independently
   verified expert. Evaluate improvement and label quality before scaling DPT
   supervision or claiming an algorithm-distillation learning curriculum.
4. Preserve the fixed chosen π0.5 motor for the main question. If diverse
   Plus/Pro development cases still have ceiling native success, move to a
   harder measured development population. The paper supports original
   π0+LIBERO90:57 and π0+AlohaTransferCube as reproduction candidates; it
   does not establish that the present π0.5 motor has headroom there. Changing
   motor/benchmark is a declared experiment, never a silent fallback.
5. Checkpoints/evaluation must count all source interaction, warmup, gradient
   updates and evaluation separately. 500k optimizer updates is not 500k
   environment actions; four source episodes are not the published budget.

No implementation defect was established by this read-only audit. The
simulation launcher is syntactically continued correctly; `--hidden_dims 128`
is intentionally expanded to three 128-wide layers. MLP and SAC are the
authors' chosen instantiation; DSRL does not require TRPO, recurrence or an MLP.
The paper cautions that motor action diversity can limit steerability;
increasing noise does not guarantee useful exploration.
[Method overview](https://diffusion-steering.github.io/),
[paper limitations](https://arxiv.org/html/2506.15799v2).

## Limits of this audit

- Paper v2 and released code can differ: Aloha entropy is one concrete example.
  Reproducing a plot requires its exact logged configuration; script defaults
  alone do not prove which settings produced it.
- Approximately 10k “online samples” is retained literally; no curve
  digitization or retrospective primitive conversion was performed. Aloha
  native/improved percentages are not invented from an unread curve.
- No claim that a learned controller can recover every failed motor rollout,
  or that source pretraining is automatically sample-efficient overall.
