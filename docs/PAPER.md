# Learning to Steer: In-Context Reinforcement Learning for Frozen VLAs

Working paper plan and durable decisions. **The design is specified; the
central claim is unproven** (see [FINDINGS.md](FINDINGS.md)). Current experiments
use native simulator rewards to test whether one attempt improves a separate
query. See the [VLRLA comparison and feedback plan](../outputs/vlrla-feedback-plan-20261109-r1/PLAN.md).
Flora-Rewarder is planned for physical-robot testing. Shared
motors, interface and protocol: [METHODS.md](METHODS.md).

Merged on 2026-11-01 from `paper/DRAFT.md` (outline sections; its run notes
moved to FINDINGS) and `paper/DECISIONS.md`. The originals are in
[the archive](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/paper/). Verified scores belong in
FINDINGS, run plans in `outputs/<run-id>/`, live state in
[coordination](../outputs/coordination.md).

Current source/student route: **action-expert Flow-SDE PPO → actual learning
histories → AD/DPT action supervision → learned KV bridge through a frozen
original VLA**. Source learning prioritizes a stronger PPO policy; it need not
use the student's steering coordinates. RoboTwin is also an active benchmark arm:
use VLRLA's six source tasks, RLinf online RL and VLRLA-bridge as the baseline
comparison, with the six remaining tasks held out from history training
([protocol](robotwin/rl-baselines.md)). The LIBERO40/90/Plus/Pro protocol below remains
the specification for that benchmark arm, not a replacement for the current
RoboCasa qualification.

The ICRL motor remains the original **frozen pi0.5**. Offline learning updates
the history model and bridge while retaining the motor's weights. Deployment
freezes those modules too: only interaction history and its derived
conditioning/KV activations change. Full-expert PPO/GRPO teacher histories
and online-RL comparisons are separate from the ICRL adaptation mechanism;
their performance gains do not establish fixed-weight in-context learning.
Autoregressive action likelihoods alone are not a reason to replace this motor.

The current RoboCasa learning stage is **atomic-first**. Keep atomic source
training and held-out atomic evaluation separate. Composite tasks are a later
transfer evaluation; they are not part of the current ICRL training stage.

## Abstract scaffold

- Question: can a reusable steering controller turn a few deployment episodes
  into high robot success while a pretrained VLA stays frozen?
- Approach: improve the source action expert with Flow-SDE PPO; train a causal
  history model and learned KV adapter through a frozen original π₀.₅ using
  action-space supervision. Compare actual AD targets with paired final-teacher
  DPT-style targets. Velocity steering remains an optional interface comparison.
- Evaluation: full-suite transfer to all 90 LIBERO90 tasks plus LIBERO-Plus
  and LIBERO-Pro; RoboTwin uses the six-source/six-held-out VLRLA comparison
  with RLinf online RL and VLRLA-bridge baselines. One actual support episode → a separate query episode,
  fixed motor/controller weights; compare query success and full cost against
  ordinary DSRL and controls for representation, architecture and feedback.
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
| [DPT](https://arxiv.org/abs/2306.14892) | Expert/optimal query-action supervision conditioned on interaction data | Paired final-teacher action labels on the same actual full-expert learning histories |
| [Algorithm Distillation](https://arxiv.org/abs/2210.14215) | Causal prediction over a source RL algorithm's learning histories | Actual full-expert learning histories distilled into a frozen-expert bridge |
| [RL²](https://arxiv.org/abs/1611.02779) | An outer-trained recurrent policy whose state carries within-task learning | Direct meta-RL over the same frozen motor interface; optimizer choice recorded separately |
| [VLRLA](https://arxiv.org/html/2610.19517v1) | VLA ICRL with fixed deployment parameters; bridge variant retains a frozen action expert and changes its KV conditioning | Direct precedent. Compare latent-noise steering, history/label construction, robotic performance and cost; further related-work audit pending |

- Broad claims such as “first ICRL for VLAs” or “first frozen-VLA adaptation”
  are unsupported. Benchmark choice alone does not establish novelty.
- The RoboTwin comparison uses a common pi0.5 initializer and task protocol
  across native SFT, RLinf online RL and VLRLA-bridge. PPO collects source
  learning histories; GRPO is the online-RL baseline in VLRLA's main table.
  A locally reconstructed VLRLA baseline must be named as such until
  official code, weights and evaluation details are verified.
- Potential scientific distinction: **which process actually yields efficient,
  competitive robotic adaptation through a frozen motor?** Must survive prior
  art comparison and controlled experiments.

## 3. Problem and frozen steering interface

The feature/motor dimensions below specify the LIBERO arm. RoboCasa source
and bridge runs bind their native checkpoint, transforms and dimensions in
their own contracts; the source/student construction is shared.

- Task \(m\); public instruction \(\ell\); observation \(o_t\): two RGB views
  and eight robot-proprioceptive coordinates. No object-state oracle.
- Meta-task: goal/instruction, environment/perturbation condition and nominal
  scene. A trial keeps this condition fixed across attempts; the learned
  controller uses its own prior outcomes to choose subsequent steering.
  Goal revelation does not imply the best noise coordinate is known; nor does
  defining a trial establish that history is useful.
  [Explicit meta-task contract](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-libero/META-TASK.md).
- One trial fixes a goal and actual scene/perturbation case. Different trials
  vary cases; within a trial preserve the case across support and query resets.
  Record stochastic observation variation rather than assume identical pixels.
- Primary one-shot evaluation: one complete support episode with actual
  feedback, followed by a **separate query episode**. Zero-shot has no support;
  also measure 2/4/8 support episodes. New trial → empty history. A query is
  never counted as its own support; query success is reported separately.
- Completed interaction history:
  \(H_t=\{(o_i,a_i,r_i,d_i)\}_{i<t}\), with actual executed action chunks.
  Current reward and target are unavailable at query \(t\). Full-expert source
  histories contain physical actions, not invented historical steering codes.
- Frozen shared feature encoder \(x_t=\phi_{\bar\theta}(o_t,\ell)\).
  Intended shared interface: visual mean 2048 + instruction mean 2048 + public
  state 8 = 4104 dimensions; mask-aware float32 pooling of materialized native
  prefix hidden states. Requalify the OpenPI40 backend and bind its exact
  feature/normalization receipt across all tasks. RLinf130 coordinates and
  student checkpoints cannot be reused as though the motor were unchanged.
- Token \(u_t=[x_t,\operatorname{embed}(a_{t-1}),r_{t-1},d_{t-1},b_t]\);
  \(b_t\): episode start.
  Previous feedback zero only at a new trial; retained across episode resets.
- Controller state and adapter:
  \((h_t,c_t)=f_\psi(u_t,h_{t-1})\);
  \(\mathrm{KV}'_t=\mathrm{KV}_{\bar\theta}(o_t,\ell)
  +g_\psi(\mathrm{KV}_{\bar\theta},c_t)\). The learned low-rank adapter starts
  with zero output, masks padding and declares its layer/K/V coverage. It is
  not the fixed random projection used by the earlier steering-PPO arms.
- Motor:
  \(a_{t:t+H}\sim\pi_{\bar\theta}(o_t,\ell;\mathrm{KV}'_t)\).
  Primary motor: official OpenPI `gs://openpi-assets/checkpoints/pi05_libero`,
  config `pi05_libero`, native horizon ten. Already staged checkpoint receipt
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`;
  qualification on the actual allocation remains required. Execute up to ten
  actions per chunk; count actual primitives. No new VLA SFT is required for
  this design.
- **Simulation feedback.** Source PPO and student histories use the native
  simulator reward. At deployment the student receives only completed feedback.
  Record reward timing, success termination, truncation and action duration for
  each benchmark so a reward ablation can account for other observable outcome
  cues. Flora enters at the [physical-robot stage](#oracle-free-reward).
- Earlier runs used simulator chunk feedback (zero on achieved goal, otherwise
  −1, with success termination). Their histories and controllers are
  historical evidence only, not this contract.
- Meta-training updates \(\psi\); deployment freezes \(\bar\theta\),
  \(\phi\), normalization and \(\psi\). Only history/state changes.
- See [history contract](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-history.md),
  [LIBERO interface](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-libero.md), and
  [common feature/task contract](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-multitask.md).

## 4. Learning the steering process

### Headroom and shortcut checks before a large corpus

- Qualify the official LIBERO40 motor/runtime and measure source/dev headroom
  across many actual cases per task. Its 40 training tasks may already be easy;
  a different checkpoint does not guarantee useful failure/recovery histories.
  The initial source collection and first fitting/transfer waves are complete. Additional scale must use a concrete recipe and remaining resource budget.
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
| Paired DPT | Same actual full-action-expert PPO histories as AD | Qualified final teacher's action chunk at the recorded query observation | Change supervision while preserving executed context/outcomes; final teacher is not automatically optimal |
| AD | Successive real full-action-expert PPO policy versions, updates between complete episodes | Actual source action chunk | Distill chronological learning histories using a frozen-expert KV bridge and flow-matching supervision |
| Direct RL² | Student's own complete multi-episode trials | Trial return, PPO in the present implementation | Meta-learn steering directly; compare additional interaction cost |

- CND mixture:
  \(z_t=(1-\epsilon_t)z^{\mathrm{teacher}}_t+
  \epsilon_t\eta_t\), \(\eta_t\sim U([-1,1]^{32})\).
- Anneal from random toward teacher over decisions spanning episode resets.
  Duration/power randomized per complete trial from declared options.
  Schedule, teacher identity and reset seed are audit metadata, never inputs.
- Fixed-teacher competence requires independent robot evaluation on the exact
  task/checkpoint/interface. Source qualification criteria and sample size
  must be declared for this arm; older pilot defaults are not an adopted
  collection budget or proof of expert quality.
- DPT labels cannot rewrite historical executed actions or their actual rewards.
  AD/DPT share source data, source collection cost and whole-stream splits.
- Planned first source route: improving action-expert Flow-SDE PPO histories,
  followed by separate supervised AD and DPT learned-KV-bridge fits. Flow
  matching uses action targets; gradients pass through the frozen expert into
  the bridge without updating the expert. No inverse steering-code labels or
  additional steering-PPO source campaign are required. Exact action encoding,
  active loss coordinates and source expertise must be qualified.
  Source teacher learning may require many more episodes than the deployment
  shot budget. Published π₀ DSRL uses batch 256, replay warmup >500 query chunks
  and UTD20; our earlier warmup 0/batch 32/K4 smoke is not that recipe.
  Final actors remain candidates until independently qualified. Scheduled CND
  and direct RL² remain separate routes, not silently combined objectives.
- DPT is not observation-only BC by definition. In this particular dataset a
  student may ignore context; test that possibility against trained static
  policies rather than assume the answer from the objective name.
- A plain causal decoder is the current history controller. Width/depth/context/data scale and
  pretraining duration: validation-selected, **final settings TBD**.
- Normalize using global training observations only; whole-trial partition
  precedes windows. Loss diagnostics do not replace own-history robot tests.
- Optional supervised warm start → PPO is a separate condition, not silently
  combined with CND/DPT or direct RL².
- See [CND contract](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-cnd.md),
  [paired corpus](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-cnd-corpus.md), and
  [direct RL²](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-rl2.md).

## 5. LIBERO40 source and three target evaluations

- Source goals: **all 40 tasks in LIBERO-Spatial, Object, Goal and LIBERO10**,
  ten tasks each. Official checkpoint README → `pi05_libero` config → dataset
  `physical-intelligence/libero`: 40 tasks / 1,693 demonstrations. This audits
  release fine-tuning, not every datum of foundation-model pretraining.
  [Provenance snapshot](../outputs/pi05-libero40-provenance-20261029-r1/PROVENANCE.json).
- The already staged official JAX release is preferred. Public ungated HF
  `lerobot/pi05_libero_base`, revision
  `5fcf06fc42b6717e22ab885fd381a096574b12c7`, is a candidate alternative;
  no binary equivalence to OpenPI weights is claimed.
- Many genuinely different initial scenes per source task; record actual
  object/robot state and perturbation identity. Numeric seeds alone are not
  cases. Thirty-two cases per task is a candidate scale, **not an approved
  budget or known sufficient meta-training amount**. Test data-scale curves.
- Disjoint source training and local-validation cases; whole cases/trials stay
  together before window extraction. Declare actual state/case lists. Older
  0–39/40–44/45–49 rules are not automatically this arm's split.
- No LIBERO90, Plus or Pro histories enter source teachers, replay, student
  gradients, normalization or labels. This includes older 90:57 and 90:47
  teacher data and the previous Plus lighting/sensor source cases. Their
  goals may alias source40 tasks; their historical datasets still stay out.
- Source teacher length and total collection scale remain to be declared.
  First show informative native failures, recoverable steering alternatives
  and source improvement on independent source-development cases. Many easy
  successes or irrecoverable failures alone do not justify a large corpus.

| Target | Primary question | Boundary |
| --- | --- | --- |
| LIBERO90 | Does source40 meta-training support one-shot transfer across the full 90-task suite? | Include all 90 tasks; no semantic filtering or prior-development exclusions |
| LIBERO-Plus | Can feedback steer across the full shifted suite? | No Plus source data; include previously inspected variants; report actual coverage |
| LIBERO-Pro | Can feedback steer across another full benchmark? | No Pro source data; include previously inspected variants; report actual coverage |

- Full-suite transfer is the current claim. Some 90 goals may overlap source40;
  Plus/Pro may use source-familiar goals. Do not label every target a new goal
  or infer distribution separation solely from a suite name.
- Previously inspected 90:1/2/47/57, Plus 667/290/2387/1438 and Pro development
  variants remain in their full-suite evaluations. Disclose earlier inspection;
  no admission audit or new exclusion list is required. Existing semantic
  audits are archival evidence only and do not gate this experiment.
- The main target one-shot endpoint is **query success after one separate
  support episode**. Zero support and 2/4/8 supports are separate endpoints;
  all support/query interactions count. Official first-attempt suite scores
  remain a distinct reporting protocol with coverage/settings disclosed.
- Target feedback does not feed source training or checkpoint selection.
  Use full-suite aggregates with actual coverage/settings and prior inspection
  disclosed; no blind dev-excluded score is required by this design.
- Earlier [v1 split](../outputs/archive/libero-pi05-corpus-20261029-r1/split.json)
  (78/10/10 + two exclusions) and
  [clean+Plus arm](../outputs/archive/libero-pi05-corpus-20261029-r3/PLUS-SOURCE-PLAN.md)
  remain immutable historical protocols. LIBERO10 is now source data; those
  earlier holdout labels do not govern the new arm. The old RLinf130 motor,
  source replay, student weights and feature coordinates do not migrate.

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
- Source-matched official PixelSAC uses its RGB/state encoder; history/static
  students use the qualified OpenPI40 prefix features. Matching raw source
  data does not match representation or architecture. A separately trained
  static student with identical features supplies the context-use control.
  Replay-only SAC source pretraining is not the online teacher-generation
  recipe; report it as an additional shared-source baseline.
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

- Predeclare support counts \(S\in\{0,1,2,4,8\}\). Primary one-shot:
  query success after exactly one completed support episode. Report support
  success, query success, first-attempt transfer and interactions separately.
  At-least-once success across attempts is secondary; repeated stochastic
  attempts can improve it without learning. Earlier K-attempt plots use a
  different convention and must not be relabeled as support-shot results.
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
| Table 1: zero-shot targets | Does steering preserve motor competence? | Full LIBERO90/Plus/Pro; coverage and prior inspection disclosed; **TBD** |
| Table 2: same-history targets | Does supervision explain performance? | Paired AD vs DPT-style labels; CND route kept separate; **TBD** |
| Source scale study | How many diverse cases and source learning steps are useful? | LIBERO40 only; identical held-out query cases and cost accounting; **TBD** |
| Fig. 3: memory/feedback controls | Is improvement caused by useful history? | Trained static/reset, reward/history/clock interventions; **TBD** |
| Fig. 4: data and amortization curves | How much meta-pretraining is needed, and when does it pay off? | Source-data/checkpoint scale; deployment vs total cost; **TBD** |
| Table 3: one-shot target adaptation | Does support feedback improve a separate query efficiently? | LIBERO90/Plus/Pro; 0/1/2/4/8 supports, query success and source/deployment cost; **TBD** |
| Appendix: source-familiar cases | Does adaptation transfer to new cases of source40 goals? | Disjoint source-local validation/test cases; separate from new-goal claims; **TBD** |

## 8. Evidence and limits

- Quantitative verified pilot evidence:
  [LIBERO evaluation ledger](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-libero/EVALUATION.md).
- Verified technical interfaces: simulator execution, native-noise replay,
  and canonical feature extraction have qualification checks. These support
  running the experiment; they do not establish meta-learning performance.
- Earlier OpenPI40 qualification reached 8/8 native and 8/8 tied-noise success
  on a tiny 90:57 development probe. This is runtime/headroom evidence, not
  source40 meta-training or an adaptation result. It does not establish
  headroom across the 40 source goals or qualify a new allocation automatically.
- Historical RLinf130 smoke: clean source 24/24; Plus source 21/24, with all
  six first attempts successful and non-monotonic later outcomes. Native
  Plus and Pro development each 8/8. See the ledger for exact costs, settings
  and hashes. These source datasets and prior clean+Plus comparison remain
  historical; they are excluded from the new source40 training arm.
- **Runtime qualified under the new contract:** one source40 task, independent
  train/validation streams, 128 episodes/127 successes, 1,223 query chunks and
  4,620 actual source SAC updates; complete recoveries and paired AD/DPT
  exports. This qualifies execution, not an improving teacher or adaptation.
- **Source40 development evidence now exists:** all40 SAC source collection,
  shared PPO/reset training and four AD/DPT fits completed. Initial all90
  native31/90 versus AD full/static26/26 and DPT26/27 showed no gain; separate
  PPO comparison native30/90 versus full24/reset26 also showed no gain. These
  contemporaneous controls remain separate. Full0/1/2/4/8 and complete Plus/Pro
  coverage, reliable adaptation and net sample efficiency remain unfinished.
- Independently qualified expert labels, context/feedback benefit, competitive
  final success and net sample efficiency remain open result slots.
- Earlier non-VLA and older-backbone probes are engineering evidence only.
  They cannot fill any VLA result slot or establish the paper's claim.
- Publication strength and novelty remain empirical/literature questions;
  no venue or acceptance outcome is assumed.


## Decisions and claim boundaries

### Agreed design

- Title: **Learning to Steer: In-Context Reinforcement Learning for Frozen
  VLAs**. A working title, not evidence that ICRL already works.
- Main question: high robotic success with few deployment interactions and
  competitive final performance. Meta-RL/ICRL are means to that outcome.
- Primary motor: official OpenPI `gs://openpi-assets/checkpoints/pi05_libero`,
  config `pi05_libero`, native horizon ten. Documented release fine-tuning:
  LIBERO-Spatial/Object/Goal/10, 40 tasks / 1,693 demonstrations. This is not
  an exhaustive foundation-pretraining audit; no new VLA SFT is required.
- Prefer the already staged official release, receipt
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
  Public HF `lerobot/pi05_libero_base`, revision
  `5fcf06fc42b6717e22ab885fd381a096574b12c7`, is an alternative candidate;
  no binary equivalence to the official JAX release is asserted.
  [Provenance](../outputs/pi05-libero40-provenance-20261029-r1/PROVENANCE.json).
- **Meta-training source only LIBERO40; targets LIBERO90, Plus and Pro.**
  The [source/target protocol](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/libero40-meta90-20261029-r1/PROTOCOL.md)
  governs this arm. The RLinf130 motor, its source data and earlier data-arm
  comparisons do not carry into it.
- **Source campaign (human, 2026-11-03): one action-supervised teacher
  corpus serves every meta-learner and steering interface.** This supersedes
  the 2026-11-02 choice of separate velocity-residual and KV-conditioning
  Stage 1 PPO arms and their label rule.
  - Teacher: full action-expert Flow-SDE PPO with the VLM frozen, one
    independent run per source task, launched in parallel under one shared
    contract (commit, PPO config, reward, update count). RoboCasa simulation
    teachers use simulator first-success reward, about 10M controls and 8
    GPUs per task, KL penalty and KL stopping off with KL still logged
    (human, 2026-11-03); the full corpus of every task is collected and
    audited before any meta-learner. Contract: [per-task 10M
    plan](../outputs/robocasa-flow-sde-10m-per-task-20261103-r1/PLAN.md).
    Each task's teacher must pass its own improvement gate; a failing task
    is dropped or replaced, not pooled. Prefer tasks with native success
    roughly 20–70%. Every task passes a one-iteration GPU qualification and
    a static audit before its full run.
  - Students read actual (observation, executed action chunk, reward,
    boundary) histories in update order and are supervised on actions with
    flow matching through the frozen expert, as in VLRLA-bridge. No
    teacher-provided velocity or KV codes are needed. The velocity residual
    and KV conditioning stay the two steering interfaces, now compared as
    supervised student ablations on the same corpus.
  - Meta-learners on this corpus: AD (behavior-action targets), DPT (final
    teacher relabels queried offline on saved query observations; historical
    actions and feedback unchanged), DIT and value-based offline ICRL
    (behavior actions with advantage weights from in-context value
    estimates). None needs behavior log-probabilities or steering codes.
  - Retain for every run: policy version per episode, every update-boundary
    checkpoint (at least the final teacher for DPT), explicit case/episode
    IDs, and enough raw RGB to rescore episodes with Flora-Rewarder offline.
  - Before fitting any history model, fit a reactive (history-free) adapter
    on each interface to the final teacher's actions. An interface that
    cannot reproduce the teacher is dropped; this measures the gap between
    weight-space teacher and frozen-expert student.
  - Not served by this corpus: direct meta-RL (RL²), CND/Vintix-style
    fixed-teacher noise histories, and the ordinary online steering baseline
    (DSRL-style, on the deployment tasks). Each needs its own runs.
  - The 2026-11-02 RoboCasa Stage 0 screen motivated the two interfaces
    (velocity 84/49/53/20, conditioning 89/44/47/20, native 79/43/44/16 of
    160; no gain p<.05) ([screen](../outputs/robocasa-interface-screen-20261102-r3/RESULT.md)).
    Both start at exact native identity with zero-initialized output.
  - Bounded output-action residuals, initial-noise steering, direct action
    heads and sample-and-select remain comparison interfaces. A direct head
    bypasses the action expert and is not evidence of steering it. The
    existing restricted conditioning/velocity screen is not a qualification
    of every learned KV/velocity adapter or of PPO improvement.
  - Primary bridge implementation: a plain causal history decoder plus a
    learned, padding-masked, token-local low-rank KV adapter with zero
    output initialization. The fixed random projection of the earlier
    KV1152 source is a different interface. Gradients pass through the
    frozen original expert into the bridge. Qualify native action encoding,
    active loss masks and own-history robot performance separately.
    [Bridge contract](../outputs/flow-sde-kv-bridge-20261103-r1/PLAN.md).
  - Full-expert PPO is a strong online adaptation comparison. Online PPO
    on the same learned bridge is additionally needed for a claim that
    holds trainable capacity fixed; it is not a source-data prerequisite.
- One frozen motor/tokenizer/normalization/feature receipt across all tasks.
  Requalify the OpenPI40 backend on the actual allocation. RLinf4104 features
  and learned controllers are incompatible receipts, not reusable coordinates.
- Planned source route: see the 2026-11-03 source-campaign decision above.
  Source improvement and final-actor expertise require independent
  evaluation; update counts alone prove neither.
- <a id="oracle-free-reward"></a>**Feedback on physical robots.** Use native
  rewards throughout the simulation study and disclose that supervision when
  describing transfer. When physical-robot testing begins, qualify Flora on
  real task videos against independently judged outcomes. Report physical
  attempts, actions and feedback cost. The [earlier Flora studies](FLORA-PPO.md)
  document the reward protocol and its known failure modes.
- Fixed-teacher CND, direct RL²/PPO and supervised warm-start plus RL remain
  separate candidate conditions, with their own source cost and contracts.
- Primary one-shot: **one complete support episode → a separate query
  episode**, fixed motor/controller weights. Both episodes' interactions
  count. Report query success separately from support success.
- A trial fixes goal/instruction and actual scene/perturbation case; memory
  persists from support to query. New trial → empty context. Different cases
  make different trials. Case/split metadata is never a controller input.
  See [META-TASK.md](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-libero/META-TASK.md).
- **Meta-learner architecture is fixed (human, 2026-11-01):** a plain
  causal decoder, as in Algorithm Distillation, DPT and Vintix (LLaMA-style
  layers, causal mask, ALiBi or RoPE so a sliding-window cache works). No
  segment recurrence or gating: not Transformer-XL or GTrXL. One backbone
  for every condition, including any direct meta-RL arm. Architecture is not
  a tested axis in the main study and the main claim makes no architecture
  claim. Static and reset controls use the same decoder. After a main result
  exists, a backbone ablation (decoder vs GTrXL vs CTM) may run as a secondary
  study with matched data, controls, parameters and update budget. The
  history controller (`vla_history.py`, `HistoryConfig.backbone`) defaults to
  `CausalDecoderCore`; `gtrxl` loads legacy checkpoints and serves the
  ablation. A strong static Transformer establishes policy transfer,
  not implicit deployment-time ICRL.

### Source and target boundaries

- Use all 40 documented release tasks as source and the full LIBERO90, Plus
  and Pro suites as targets. No semantic-goal filtering or prior-development
  exclusions. Declare disjoint source training/local-validation cases; whole
  cases/trials stay together before window extraction.
- Many cases per task means actual object/robot placements and conditions,
  not just many RNG seeds. Same saved initial state plus a new seed need not
  change placement. Record actual states and stochastic observations.
- Thirty-two cases per task is a candidate data scale, not a known sufficient
  amount or approved collection budget. Measure source-data scale and teacher
  learning length separately from deployment-shot efficiency.
- No LIBERO90/Plus/Pro teacher collection, replay, labels, meta-gradients or
  normalization in this source40 arm. Older 90:57/90:47 and Plus 2387/1438
  histories remain excluded even when their goals alias a source40 goal.
- Claim full-suite transfer, not that every LIBERO90 task is semantically unseen.
  Source-goal overlap and source-familiar Plus/Pro goals may remain. Suite
  names alone do not establish new goals or strict distribution separation.
- Prior 90:1/2/47/57 inspection, Plus 667/290/2387/1438 and Pro development
  variants do not exclude them from full-suite evaluation. Disclose prior
  inspection. Existing semantic audits are archival only; stop further
  admission auditing and do not gate execution on those artifacts.
- No LIBERO90/Plus/Pro histories enter source training, even though their
  full evaluations include previously inspected tasks and variants.
- Separate zero-support and 1/2/4/8-support query endpoints. Earlier K total
  attempt conventions cannot be relabeled as support-shot counts. Official
  first-attempt scores are another reporting protocol; disclose settings,
  coverage and prior inspection. No blind excluded score is required.
- No final-target feedback for curriculum, method/checkpoint selection or
  corpus generation. Previously inspected target cases remain disclosed
  within the full-suite results.
- Prior [78/10/10 split](../outputs/archive/libero-pi05-corpus-20261029-r1/split.json)
  with exclusions 90:38/77 and
  [clean+Plus source arm](../outputs/archive/libero-pi05-corpus-20261029-r3/PLUS-SOURCE-PLAN.md)
  remain immutable historical protocols. LIBERO10 is source in the new arm;
  the old source-state 0–39/validation 40–44/test 45–49 rule is not inherited.
- Historical clean-only→Plus/Pro claims concern unseen benchmark shifts for
  the attachment; historical clean+Plus→Plus claims concern within-Plus
  held-out cases. Pro was absent from both source arms. These distinctions
  explain earlier evidence; they are not new source40 data arms.

### Necessary comparisons and accounting

- Native, tied noise, ordinary official DSRL, simple declared noise search;
  cold and source-pretrained DSRL distinguished and interactions charged.
- Independently train static and episode-reset controllers with identical
  source trials, labels, initialization and optimization budgets. Resetting
  the full-trained model only at inference is a separate intervention.
- Static resets each decision and lacks past feedback. Episode reset removes
  cross-episode history while preserving within-episode memory. Neither
  control alone proves a mechanism; compare full-history query outcomes.
- Official PixelSAC uses RGB/state; history/static use qualified OpenPI40
  prefix features. Same raw source data does not isolate representation or
  architecture. Feature-matched SAC addresses a separate representation
  question; the matched static student isolates context use.
- Source-matched replay-only SAC pretraining is an additional control, not
  the online source-teacher recipe. It adds optimizer cost, not new rollouts.
- Reward/history interventions and clock/no-history controls probe feedback
  use. Reward shuffling removes numeric-reward information, not necessarily
  visual feedback; no single intervention is a universal ICRL proof/gate.
- MLP is not intrinsic to DSRL. Architecture attribution requires matching
  optimizer, critics, objective, features, data and budget.
- Report support success, separate query success, primitive interactions,
  task/case strata and trial/task-clustered uncertainty. At-least-once success
  can increase through repeated stochastic attempts without adaptation.
- Count unique source interactions once across paired objectives; separate
  source training/validation, teacher/controller compute and deployment
  support/query costs. Include competence checks and direct meta-RL rollouts.
  Report deployment efficiency and amortized total cost; pretraining is not free.

### Hypotheses and open choices

- Source40 motor competence may already leave little useful headroom. First
  establish diverse native failures that alternate steering can recover.
  Neither many easy successes nor many irrecoverable failures suffice.
- Source teachers need not finish in 4/8 episodes. Published DSRL uses batch 256,
  replay warmup exceeding 500 query chunks, UTD20; a fresh K4 run cannot reach
  that warmup. Declare and validate any shortened schedule rather than call
  earlier warmup 0/batch 32 smoke histories the published recipe.
- Source teacher competence, source scale, optimization schedule and resource
  budget remain open; no giant corpus launch is authorized by a paper update.
  If genuine case diversity still lacks useful headroom, revise task/benchmark
  conditions before large collection. Artificial hidden physics is not assumed.
- A goal-revealing instruction need not reveal the best steering coordinate.
  Test whether feedback supplies value beyond observation-only control.
- Scheduled CND can learn a clock shortcut; reward-dependent corruption is
  only an unadopted ablation. Square-root mixing does not preserve a Gaussian
  law for arbitrary bounded/selected teacher coordinates.
- DPT is not observation-only BC by definition. Context dependence is
  empirical; final-candidate labels are not called expert without evaluation.
  Matching a simple bandit weakens a complex-controller contribution without
  negating the definition of meta-learning.
- Central hypothesis: reusable history-to-steering training improves separate
  query success with few supports, preserves high final performance and earns
  its full source cost. **Not supported by the first source40/target90 waves:** supervised and direct PPO controllers are below their contemporaneous native controls; full shifted-suite and multi-shot comparisons remain incomplete.
- Freeze training seeds, stopping rules, target cases, success margins and
  uncertainty procedure before final comparison. No outcome-selected final
  failures or synthetic hidden condition introduced to force a memory gain.
- [VLRLA](https://arxiv.org/html/2610.19517v1) already studies fixed-parameter
  VLA ICRL and a frozen-expert bridge; DSRL supplies latent-noise steering.
  Broad first/novelty claims and venue acceptance claims remain unsupported.
- Historical one-family RLinf130 smoke: clean 24/24, Plus source 21/24 and native
  Plus/Pro development 8/8 each. Ceiling/non-monotonic results do not establish
  an improving expert, broad task generalization or useful memory. Earlier
  OpenPI40 native 8/8 + tied 8/8 checks are runtime/headroom evidence only.
  These scores stay in the [evaluation ledger](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/frozen-vla-libero/EVALUATION.md),
  outside the new experiment's result slots.
- Source40 runtime qualification is complete on one task: train59/60 and
  validation68/68 success, 1,223 actual chunks and4,620 source SAC updates.
  Paired AD/DPT exports and full recoveries are verified. This is execution
  evidence; it does not establish an improving teacher or meta-controller.

- User selected an initial1M executed-action allowance per benchmark, shared
  across SAC, fresh PPO, validation, settling and retries. All40 LIBERO source
  training goals and six RoboTwin source goals remain. Source-local LIBERO
  validation uses eight fixed goals on disjoint states. This original shared cap applied to the initial pilot. Later human steering
  accepted roughly1M source actions per learner, without increasing the32-hour
  LIBERO compute pool; existing runs retain their original contracts. The human
  removed the per-learner action cap on 2026-11-03. Neither
  quantity is a proven sufficient meta-learning dataset.

- Execution order: after minimal runtime and saved-data qualification, overlap
  collection, fitting and frozen query evaluation on immutable completed
  batches. Keep ordinary target-online DSRL and fresh PPO baselines moving
  while offline students train. Unexpected outcomes get a written diagnosis
  and a bounded discriminating test before repeating the same recipe at scale.
