# Learning to Steer: In-Context Reinforcement Learning for Frozen VLAs

Working paper; compact drafting notes. **Design is specified; the central
robot-performance claim is unproven.** Result placeholders are intentional.
Durable choices: [DECISIONS.md](DECISIONS.md). Method contracts remain the
implementation references; this outline does not replace them.

Current data arms: **original-LIBERO clean source versus the same source plus
a declared Plus training partition**, with identical clean validation and
held-out Plus/Pro evaluation cases. Plus is an unseen benchmark shift for the
clean-only arm; for the augmented arm it is within-Plus case generalization.
Pro remains absent from attachment pretraining in both arms.
The [v2 protocol](../../outputs/libero-pi05-corpus-20261029-r2/EVALUATION-PROTOCOL.md)
records evaluation roles; the explicit r3 source extension adds the approved
Plus training partition without rewriting the frozen v1 data split.

## Abstract scaffold

- Question: can a reusable steering controller turn a few deployment episodes
  into high robot success while a pretrained VLA stays frozen?
- Approach: causal history → bounded latent-noise coordinate → frozen π₀.₅;
  pretrain using actual source-policy learning histories; compare data construction,
  supervision and direct meta-RL.
- Evaluation: LIBERO-Plus and LIBERO-Pro; success against actual deployment
  interactions; ordinary DSRL; controls for representation, architecture,
  memory, feedback and training cost. Original task holdouts: auxiliary.
- **Results: TBD.** No percentage gains, efficiency factors or novelty claim yet.

## 1. Introduction / question

- Main outcome: robotic performance, deployment sample efficiency and final
  success. ICRL/meta-RL literature supplies candidate training processes.
- Preserve an existing motor's control competence; learn how to steer it
  from actual prior actions and outcomes.
- Distinguish strong first-attempt transfer from improvement caused by
  deployment feedback. Either can help robot success; only the latter
  supports the proposed adaptation mechanism.
- Target: fewer deployment interactions than ordinary online steering while
  retaining competitive final success. Required improvement and final-success
  margin: **prespecify before final evaluation; TBD**.
- Candidate contributions, conditional on evidence: an effective reusable
  steering process; a controlled study of learning-history construction and
  targets; an audited robot-performance/cost benchmark. No contribution
  wording should imply these outcomes have already occurred.

## 2. Related work / positioning

| Prior work | Established ingredient | Required distinction here |
| --- | --- | --- |
| [DSRL](https://arxiv.org/abs/2506.15799), [official π₀ implementation](https://github.com/nakamotoo/dsrl_pi0) | RL over the noise input of a frozen diffusion/flow policy | A reusable history-conditioned steering process; held-out-task deployment cost and final success |
| [Vintix](https://arxiv.org/abs/2501.19400) | Cross-domain action modeling through ICRL; noise-distillation motivation | Apply the data-generation idea in VLA steering coordinates; verify robot outcomes rather than infer transfer |
| [DPT](https://arxiv.org/abs/2306.14892) | Expert/optimal query-action supervision conditioned on interaction data | Paired expert labels on exactly the same actual noisy steering histories |
| [Algorithm Distillation](https://arxiv.org/abs/2210.14215) | Causal prediction over a source RL algorithm's learning histories | Real improving steering-policy checkpoints; separate from fixed-teacher noise schedules |
| [RL²](https://arxiv.org/abs/1611.02779) | An outer-trained recurrent policy whose state carries within-task learning | Direct meta-RL over the same frozen motor interface; optimizer choice recorded separately |
| [VLRLA](https://arxiv.org/html/2610.19517v1) | VLA ICRL with fixed deployment parameters; bridge variant retains a frozen action expert and changes its KV conditioning | Direct precedent. Compare latent-noise steering, history/label construction, robotic performance and cost; further related-work audit pending |

- Broad claims such as “first ICRL for VLAs” or “first frozen-VLA adaptation”
  are unsupported. Benchmark choice alone does not establish novelty.
- Potential scientific distinction: **which process actually yields efficient,
  competitive robotic adaptation through a frozen motor?** Must survive prior
  art comparison and controlled experiments.

## 3. Problem and frozen steering interface

- Task \(m\); public instruction \(\ell\); observation \(o_t\): two RGB views
  and eight robot-proprioceptive coordinates. No object-state oracle.
- Meta-task: goal/instruction, environment/perturbation condition and nominal
  scene. A trial keeps this condition fixed across attempts; the learned
  controller uses its own prior outcomes to choose subsequent steering.
  Goal revelation does not imply the best noise coordinate is known; nor does
  defining a trial establish that history is useful.
  [Explicit meta-task contract](../frozen-vla-libero/META-TASK.md).
- One trial: \(K\) complete episodes on the same declared task/initial state;
  state and reset seed held fixed across attempts. New trial → empty history.
- Completed interaction history:
  \(H_t=\{(o_i,z_i,r_i,d_i)\}_{i<t}\). Current reward and target unavailable
  at query \(t\).
- Frozen shared feature encoder \(x_t=\phi_{\bar\theta}(o_t,\ell)\).
  Current implementation: visual mean 2048 + instruction mean 2048 + public
  state 8 = 4104 dimensions; mask-aware float32 pooling of materialized native
  prefix hidden states. One feature contract across all source tasks.
- Token \(u_t=[x_t,z_{t-1},r_{t-1},d_{t-1},b_t]\); \(b_t\): episode start.
  Previous feedback zero only at a new trial; retained across episode resets.
- Controller state and proposal:
  \((h_t,\tilde z_t)=f_\psi(u_t,h_{t-1})\);
  \(z_t=g(\tilde z_t)\in[-1,1]^{32}\), with the adapter \(g\) fixed and
  recorded. Supervised Gaussian/clipped and PPO squashed-Gaussian variants
  have distinct likelihood contracts.
- Motor:
  \(a_{t:t+H}=\pi_{\bar\theta}(o_t,\ell;\operatorname{repeat}_H(z_t))\).
  Target: frozen `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`, revision
  `6222623f635769bfc73c9472e29fab9b7fd8e027`; native horizon ten; execute up to ten actions,
  stopping at simulator success/time limit. Record actual primitive steps.
- Sparse chunk feedback: zero on achieved goal, otherwise −1; physical
  episode ends on success or time limit. Source critic timeout bootstrapping
  and student episode-boundary feedback are distinct.
- Meta-training updates \(\psi\); deployment freezes \(\bar\theta\),
  \(\phi\), normalization and \(\psi\). Only history/state changes.
- See [history contract](../frozen-vla-history.md),
  [LIBERO interface](../frozen-vla-libero.md), and
  [common feature/task contract](../frozen-vla-multitask.md).

## 4. Learning the steering process

### Headroom and shortcut checks before a large corpus

- Finish the bounded π₀.₅ technical/teacher pilot, then predeclare training/dev
  task noise-headroom and repeatability checks. **Large teacher collection is
  deferred until this evidence supports a useful steering/adaptation problem.**
- Measure whether noise choices reproducibly change outcomes, how much native
  success leaves to improve, and whether a small search procedure already
  captures that gain. Ceiling results cannot establish sample-efficiency gains.
- A public instruction may identify the goal while leaving the best steering
  choice uncertain. Artificial hidden physics is unnecessary for this question
  and is not part of the agreed experiment.
- Next learning comparison: small corpus against trained static/reset policies
  and a simple declared noise-bandit/search baseline at the same interaction
  budget. This precedes expanding the full task bank.
- Scheduled CND remains a candidate. A learned annealing clock is a possible
  shortcut; reward-dependent corruption is an unadopted ablation candidate,
  not an established remedy. Square-root mixing does not guarantee Gaussian
  noise for bounded or selected teacher coordinates.

| Route | Behavior used to collect context | Query target / outer objective | Interpretation |
| --- | --- | --- | --- |
| CND | Fixed competent teacher mixed with annealed bounded uniform noise | Actually executed mixture | Vintix-inspired noise distillation in steering space; not an improving teacher |
| Paired DPT | **Same actual CND histories**, no extra rollouts | Already queried deterministic teacher coordinate | Change supervision while preserving all executed context/outcomes bitwise |
| AD | Successive real DSRL policy checkpoints, updates between complete episodes | Original executed coordinate | Distill a source learning trajectory; a collection clock is not a teacher-update identity |
| Direct RL² | Student's own complete multi-episode trials | Trial return, PPO in the present implementation | Meta-learn steering directly; compare additional interaction cost |

- CND mixture:
  \(z_t=(1-\epsilon_t)z^{\mathrm{teacher}}_t+
  \epsilon_t\eta_t\), \(\eta_t\sim U([-1,1]^{32})\).
- Anneal from random toward teacher over decisions spanning episode resets.
  Duration/power randomized per complete trial from declared options.
  Schedule, teacher identity and reset seed are audit metadata, never inputs.
- Fixed-teacher competence gate: measured robot success on the exact
  checkpoint/task/interface; current protocol defaults ≥70% over ≥8 episodes.
  Gate choice is a collection criterion, not proof of student quality.
- DPT labels cannot rewrite historical executed noise or its actual reward.
  Same source data and collection cost shared across CND/DPT.
- Current pilot exports paired AD and **DPT-style final-candidate labels**
  from the same fresh-DSRL learning histories. Those final actors have not
  passed an independent expertise test; DPT's optimal-label assumptions are
  not established. Scheduled CND remains a separate candidate data route.
- DPT is not observation-only BC by definition. In this particular dataset a
  student may ignore context; test that possibility against trained static
  policies rather than assume the answer from the objective name.
- GTrXL is the current causal controller. Width/depth/context/data scale and
  pretraining duration: validation-selected, **final settings TBD**.
- Normalize using global training observations only; whole-trial partition
  precedes windows. Loss diagnostics do not replace own-history robot tests.
- Optional supervised warm start → PPO is a separate condition, not silently
  combined with CND/DPT or direct RL².
- See [CND contract](../frozen-vla-cnd.md),
  [paired corpus](../frozen-vla-cnd-corpus.md), and
  [direct RL²](../frozen-vla-rl2.md).

## 5. Dataset and two external evaluation protocols

- Approved data comparison: clean original LIBERO versus clean plus
  predeclared Plus source cases. Hold validation, architecture, labels,
  optimization budget and final evaluation cases fixed within the comparison.
  Plus training adds real collection cost; it does not create a free data arm.
- Scope of distribution claims: clean-only → Plus/Pro is unseen-benchmark
  shift transfer for the attachment. Clean+Plus → disjoint Plus cases is
  within-benchmark generalization; category/parameter/base-family holdouts
  determine any stronger OOD claim. Pro has no attachment-pretraining data.
- Backbone release is trained on the original 130 LIBERO tasks. Externally
  held-out perturbation cases are the main claim; original base tasks are
  not held out from VLA pretraining. Pin weights, normalization and backend
  together; older OpenPI teachers/features are a different contract.
- Frozen [split specification](../../outputs/libero-pi05-corpus-20261029-r1/PLAN.md)
  and [machine-readable manifest](../../outputs/libero-pi05-corpus-20261029-r1/split.json):
  **78 training / 10 validation / 10 official LIBERO-10 test tasks**;
  two additional LIBERO-90 tasks excluded into an auxiliary test bank.
- Excluded ID 38: right moka pot → stove; identical full goal/object bindings
  to official test ID 2 despite different wording/initial stove state.
  Excluded ID 77: book → back caddy compartment; same full goal and instruction
  as official test ID 5 in another scene.
- Semantic family = transitive equality of normalized full goal + referenced
  object bindings, or full instruction. All siblings share a partition.
  Shared physical scenes and atomic subgoals remain permitted.
- All 100 canonical tasks: 50 verified initial states. Training tasks:
  states 0–39 source collection, 40–44 auxiliary familiar-task validation,
  45–49 auxiliary familiar-task test. Main validation: held-out validation
  tasks; auxiliary task test: official ten tasks, all 50 states unless a smaller
  pilot is declared before running it.
- No official test or excluded-task data in **any source DSRL teacher
  collection/updates**, meta-gradients, normalization, label construction,
  curriculum or checkpoint selection. Validation-only teachers remain
  isolated from training teachers/students.
- Distinct claims: familiar-task new-state transfer; held-out full-task/goal
  transfer; later perturbation robustness. None implies absence from VLA
  pretraining.
- Original LIBERO-10 remains an auxiliary attachment-only task-transfer test;
  the v1 source exclusions and split remain unchanged.
- Plus/Pro: the two primary external tests; record canonical base task/family and
  perturbation lineage; descendants inherit base partition. Training-base
  perturbations assess robustness; held-out-base variants additionally assess
  task transfer. Final feedback cannot select models or methods. A separately
  declared development subset is allowed for simulator/method selection.
- Initial Plus/Pro development base: original object task 1, cream cheese →
  basket, typed-goal family matching training task 90:47. Reserve this family
  from blind final tests. Official Plus has no perturbed task 90:57 counterpart.
- First-attempt official-style scores, with suite settings/coverage disclosed,
  are separate from K-attempt adaptation. Disclose any development overlap in
  a full official aggregate and also report its blind excluded subset.
- Exact development cases, source/asset pins, final-case identity fields and
  seed exclusions: [v2 manifest](../../outputs/libero-pi05-corpus-20261029-r2/protocol.json).
  New final IDs await pinned inventories; no outcomes used to invent cases.
- The approved Plus source partition is disjoint from the existing Plus
  development cases667/290: lighting case2387 and sensor-noise case1438,
  three source scenes each. See [source-arm declaration](../../outputs/libero-pi05-corpus-20261029-r3/PLUS-SOURCE-PLAN.md).
- Current Plus source lighting/sensor cases and camera/robot development
  cases occupy different categories. This narrow factor split is explicit;
  it is not a claim that the augmented model has never seen Plus.
- **Current corpus scale:** one goal family47, four clean training scenes,
  two clean local-validation scenes and six Plus source trials. The planned
  78-task bank has not been collected by this smoke test. No new-goal or broad
  meta-task generalization follows from these data.

## 6. Baselines and mechanism controls

- Native π₀.₅ independent Gaussian noise; separate tied Gaussian baseline
  to isolate the dimensionality/tying change.
- Simple noise-bandit/search baseline with predeclared arms/proposals and
  charged interactions. Matching it would weaken the need for a complex
  controller, while remaining compatible with a learned meta-RL process.
- Ordinary official DSRL: SAC steering policy with its published MLP/visual
  encoder; frozen/offline prior and online deployment adaptation reported
  separately. Every held-out online trial gets an independent initialization
  and a charged interaction budget; its feedback never returns to meta-data.
- **Open specification:** fresh DSRL initialization versus a compatible
  train-only shared prior on unseen tasks; report explicitly, avoid hidden
  held-out-task source pretraining.
- Feature-matched SAC control: same shared VLA features as history student;
  needed to distinguish input-representation gains from learning-process gains.
- Source-matched official PixelSAC instead uses its RGB/state encoder;
  history/static students use RLinf4104 features. Matching raw training data
  does not match representation or architecture. The separately trained
  static student provides the current controlled test of history benefit.
- Expert-only BC; independently trained static Transformer with matched
  data/initialization/update budget; independently trained episode-reset
  controller. Resetting the full-trained policy only at evaluation is an
  additional intervention, not an equivalent trained baseline.
- Static: reset every decision, remove previous feedback. Episode reset:
  within-episode memory remains. Neither is interchangeable with full history.
- Reward-blind/reward-permuted and history-shuffled/reset interventions;
  clock/no-history schedule control. Distinguish feedback use from annealing,
  repeated stochastic attempts, architecture and additional training.
- Reward shuffling tests dependence on numeric rewards; visual outcomes can
  carry feedback too. Neither this intervention nor a bandit comparison is
  a universal definition/proof of ICRL; interpret alongside trained controls.
- MLP is an implementation choice in the reference, not a requirement of
  DSRL/SAC. A Transformer-versus-MLP architecture claim requires matched
  objective, critics, optimizer, representation, parameters and budgets.
- A static Transformer can be a better pretrained policy; its architecture
  alone does not demonstrate implicit deployment-time ICRL.

## 7. Measurements, figures and result slots

- Predeclare \(K\in\{1,2,4,8\}\). Per-attempt success; cumulative success at
  least once; first-attempt transfer; final-attempt success; full success curve
  against actual primitive interactions. Extra attempts alone improve
  success-once even without adaptation.
- Pair task/initial-state/reset plans; preserve successes and failures;
  independently seed each method. Treat correlated attempts as one trial.
- A fixed initial-state index with a new RNG seed does not automatically
  generate a new object placement. Verify actual initial observations/states;
  count distinct declared initial states separately from repeated resets.
- Uncertainty: paired task/trial clustering; multiple training seeds;
  task-level aggregates alongside per-task results. Seed count and confidence
  procedure finalized before the full test, not after observed gains.
- Full source cost: teacher learning + competence checks + history generation
  + any direct meta-RL interaction. Shared CND/DPT source cost counted once.
  Separate training/validation source cost, deployment cost, settling actions,
  GPU time, optimizer work and inference latency.
- Deployment efficiency and amortized total cost both reported; source
  pretraining is not free. Break-even versus number of deployments: TBD.

| Planned item | Research question | Required comparison / current result |
| --- | --- | --- |
| Fig. 1: method and chronology | What changes during deployment? | Frozen motor/controller; history updates; actual action→outcome timing |
| Fig. 2: success vs interactions | Does steering adapt faster while keeping high final success? | Ordinary DSRL, native, CND/DPT/AD/RL²; **TBD** |
| Table 1: Plus and Pro first attempts | Does steering preserve external-suite competence? | Separate official-style settings, coverage and blind dev-excluded scores; **TBD** |
| Table 2: same-history targets | Does supervision explain performance? | Current paired AD vs DPT-style labels; CND route kept separate; **TBD** |
| Data-arm comparison | Does Plus-source exposure help, and where does it transfer? | Clean vs clean+Plus; identical clean validation and held-out Plus/Pro cases; **TBD** |
| Fig. 3: memory/feedback controls | Is improvement caused by useful history? | Trained static/reset, reward/history/clock interventions; **TBD** |
| Fig. 4: data and amortization curves | How much meta-pretraining is needed, and when does it pay off? | Source-data/checkpoint scale; deployment vs total cost; **TBD** |
| Table 3: Plus/Pro adaptation | Does own-history steering improve external-suite success efficiently? | K1/2/4/8, base/perturbation strata and source/deployment cost; **TBD** |
| Appendix: original task holdouts | Does the attachment transfer to new full goals? | Immutable v1 ten-task bank; auxiliary; **TBD** |

## 8. Evidence and limits

- Quantitative verified pilot evidence:
  [LIBERO evaluation ledger](../frozen-vla-libero/EVALUATION.md).
- Verified technical interfaces: simulator execution, native-noise replay,
  and canonical feature extraction have qualification checks. These support
  running the experiment; they do not establish meta-learning performance.
- The older OpenPI π₀.₅ motor reached the ceiling on a small task/scene probe. Treat this
  as headroom/runtime evidence; no adaptation benefit follows. Quantitative
  pilot counts belong in the consolidated LIBERO `EVALUATION.md` before use
  in a paper table. Those checks do not qualify the newly selected RLinf130
  motor or establish its Plus/Pro results.
- **Pending:** independent expert qualification if fixed-teacher CND is pursued;
  frozen-student robot-performance comparisons;
  held-out-task own-history success; reliable memory/feedback benefit;
  competitive final success and sample efficiency; Plus/Pro.
- Audited RLinf source data and development baseline rollouts now exist;
  see the evaluation ledger for counts, costs and hashes. Clean source
  episodes have a success ceiling; the Plus source sequence is not monotonic;
  native success also ceilings the selected Plus/Pro development cases.
  None of these runs evaluated a frozen meta-controller. Candidate training
  and held-out own-history comparisons remain separate result slots.
- Earlier non-VLA and older-backbone probes are engineering evidence only.
  They cannot fill any VLA result slot or establish the paper's claim.
- Publication strength and novelty remain empirical/literature questions;
  no venue or acceptance outcome is assumed.
