# Learning to Steer: In-Context Reinforcement Learning for Frozen VLAs

Working paper; compact drafting notes. **Design is specified; the central
robot-performance claim is unproven.** Result placeholders are intentional.
Durable choices: [DECISIONS.md](DECISIONS.md). Method contracts remain the
implementation references; this outline does not replace them.

Current design: **official OpenPI π₀.₅ LIBERO40 motor; source histories only
from LIBERO40; one-support-episode adaptation on LIBERO90, LIBERO-Plus and
LIBERO-Pro.** Use all 40 source tasks and the full target suites, without
semantic-goal filtering or prior-development exclusions. This is full-suite
transfer, not a claim that every LIBERO90 goal is semantically unseen. Case/seed
plans are declared before the comparison. Existing RLinf130
runs, the 78/10/10 split and clean-versus-clean+Plus arms remain historical.
The [source40/target90 protocol](../../outputs/libero40-meta90-20261029-r1/PROTOCOL.md)
is the current design. The original campaign used a1M combined-action cap per benchmark. Later human
steering accepted approximately1M source actions per learner on both, with
complete episodes/trials and benchmark-appropriate horizons. Existing runs
retain their original contracts; future expansion costs stay explicit. The
32 additional LIBERO GPU-hour pool is approved.
Current source bank:583 K4 training cases/2,332 episodes/33,729 steering
decisions across all40 tasks; fixed validation146 cases/584 episodes/8,202
decisions. Training all-in351,297 actions, validation85,301 and historical
overhead21,253 once: resource total457,851. All four new interim100 fits
completed with25/100 full recoveries. Their complete fresh LIBERO90 readout
scores native28/90; ADfull/static29/90each; DPTfull/static28/90each. ADK1full
and same-weight reset both6/9(the same successes as nativeK0); DPT5/9versus
reset6/9. AD pays4014actions versus native1961for that same6/9query outcome.
All522episodes passed the saved-trace audit. This supplies no reliable memory,
high-performance or sample-efficiency benefit. Historical500versus100 changes
both data and native wave, so it does not isolate scaling or overfitting.
Current human priority is ordinary DSRL first on the entirely frozen official
pi0.5, followed by native action-expert PPO; further offline-student sweeps
are held. The fresh task57 pi0.5-adapted PixelSAC curve is running toward500k optimizer
updates, within10GPUh/1M all-in actions/64GiB. Native query is10/10; learned
panels through380120 remain10/10 with shorter observed episodes. At390080
updates the panel drops to9/10,1518 query primitives versus native1451: paired
state48 times out400 versus native success152; the next two panels return10/10.
These repeated panels do not
establish reliable success preservation, memory or total sample efficiency;
final native stability control remains pending. Native PPO
GPU dispatch is held; its later25 TOTAL iterations plus the same128 starting-
model cases are explicitly authorized after the pilot passes. Its sampling
noise cannot directly label steering of the original frozen motor. AD can
instead predict actual z from fixed-motor SAC learning histories; DPT can
later label saved query observations with a competent final same-motor SAC
teacher. [Compatible-data contract](../../outputs/dsrl-pi05-task57-20261031-r1/DATA-CONTRACT.md).
New targeted Motus/RoboTwin result (2026-10-31): actual zero/native parity
and a frozen-suffix KV gradient pass. The completed fixed4D screen yields no
qualifying rescue. A supervised capacity attempt then stops before fitting:
the third fixed published demonstration fails native-route replay after two
successes. This is data/execution compatibility evidence, without a learned
capacity, RL-corpus or Transformer-adaptation result. No teacher is excluded.
[Evaluation and limits](../robotwin-coefficient-steering/EVALUATION.md).

The earlier527-case/2,108-episode source snapshot fitted all four controllers
through500 updates. Its first all90 target queries scored native31/90; AD
full/static26/90; DPT full26/90/static27/90. On the fixed-nine K1 panel,
AD/DPT full each6/9 versus same-weight reset5/9, while native and AD-static
already reach6/9 atK0. The sole memory discordance follows successful support.
Tiny Plus/Pro panels are all4/4 and do not constitute full benchmark results.

Earlier one-trial direct PPO source policies completed238/237 updates and171,166/171,707
actions. Source validation is near ceiling:63/64 full versus61/64 trained
reset, both16/16 on first attempts. Frozen target wave: native30/90,
full24/90, trained reset26/90; all three K1 memory/reset conditions5/9.
All90 noise controls separately score native29/90, zero25/90, tied Gaussian
28/90 and clipped tied29/90. These native controls belong to separate waves:
matching initial inputs and noise can still produce small cross-process
motor-action differences, so scores are not pooled.

Ordinary target-online PixelSAC remains4/6 across0/1/4/8/13/16 training
episodes. The two failed cases completed240 episodes each as outcome-selected
developmentdiagnostics:task0 has72train successes andonequery80success,
butboth160/240queries fail;task40 hasno successes. Repeatedfrozen-seed
checks are being prepared; these are not broad or stable performance results. The
predeclared source-SAC panel is16/16 for native, actor mean and actor sample;
it establishes no teacher advantage. Source histories contain94 failures
among2,108 episodes and only10 cases with learned-behavior failure followed
by learned-behavior success. More of this corpus is not yet justified as a
solution to the transfer deficit. No reliable robotic memory, high-performance
or sample-efficiency advantage is established. Detailed unexpected-result
analysis and next discriminating tests: [experiment summary](../frozen-vla-steering-summary.md).

RoboTwin uses six source goals and six controller-held-out goals, with a
1024D independent-noise interface versus LIBERO's32D tied noise. Its three
source collectors resumed after native simulator reset failures, preserving
replay/optimizer/RNG state. Both PPO arms now completed100 rounds and roughly
1M source actions each. FULL and trainedRESET score8/16 on the two-goal
development panel; cache-reset7/16,rewardblind9/16. A saved native panel is
verified on the same16 cases/underlying motor contract and scores6/16, with
different execution dates and noise draws. No reliable memory or broad
robot-performance improvement is established. SAC source100 is now complete:
277/2400 successes (11.54%),1,003,790 source actions and638,500 joint SAC
updates over32,425 distinct decisions. Quarter success78/57/76/66 per600
shows no sustained improvement. Weak source learning is not limited to
recurrent PPO; the cause is unisolated. Source attempt1–4 success is also
non-monotonic:FULL103/106/97/110 per600,RESET98/110/106/107,
SAC72/70/64/71. These scene-varying source marginals do not establish
feedback adaptation or a causal memory benefit. Final SAC sourceVAL is2/24;
its frozen target policy scores6/8 bottle +2/8 stapler=8/16 on the same
actual cases as FULL/RESET8/16. Source-initialized online SAC scores5/16,
with3320 additional target SACcycles/5120actions and updates inside episodes;
these indices are not separate support/query budgets. Larger student fits
are not complete. Benchmark costs
are separate from LIBERO. Future
collection should overlap initial fitting/evaluation after a bounded smoke,
rather than wait for the whole corpus. PPO still requires fresh policy rollouts.

The same historical matched native bank scores6/16; frozenSAC has3casewins/
1loss against it, with distinct executors/dates/noise draws. Its target totals
at source10/25/50/100 are3/4/3/8 per16, so the weak source averages do not
exclude late target improvement. This remains a small development comparison,
without reliable robotic-memory or sample-efficiency evidence.

## Abstract scaffold

- Question: can a reusable steering controller turn a few deployment episodes
  into high robot success while a pretrained VLA stays frozen?
- Approach: causal history → bounded latent-noise coordinate → frozen π₀.₅;
  pretrain using actual source-policy learning histories; compare data construction,
  supervision and direct meta-RL.
- Evaluation: full-suite transfer to all 90 LIBERO90 tasks plus LIBERO-Plus
  and LIBERO-Pro. One actual support episode → a separate query episode,
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
- One trial fixes a goal and actual scene/perturbation case. Different trials
  vary cases; within a trial preserve the case across support and query resets.
  Record stochastic observation variation rather than assume identical pixels.
- Primary one-shot evaluation: one complete support episode with actual
  feedback, followed by a **separate query episode**. Zero-shot has no support;
  also measure 2/4/8 support episodes. New trial → empty history. A query is
  never counted as its own support; query success is reported separately.
- Completed interaction history:
  \(H_t=\{(o_i,z_i,r_i,d_i)\}_{i<t}\). Current reward and target unavailable
  at query \(t\).
- Frozen shared feature encoder \(x_t=\phi_{\bar\theta}(o_t,\ell)\).
  Intended shared interface: visual mean 2048 + instruction mean 2048 + public
  state 8 = 4104 dimensions; mask-aware float32 pooling of materialized native
  prefix hidden states. Requalify the OpenPI40 backend and bind its exact
  feature/normalization receipt across all tasks. RLinf130 coordinates and
  student checkpoints cannot be reused as though the motor were unchanged.
- Token \(u_t=[x_t,z_{t-1},r_{t-1},d_{t-1},b_t]\); \(b_t\): episode start.
  Previous feedback zero only at a new trial; retained across episode resets.
- Controller state and proposal:
  \((h_t,\tilde z_t)=f_\psi(u_t,h_{t-1})\);
  \(z_t=g(\tilde z_t)\in[-1,1]^{32}\), with the adapter \(g\) fixed and
  recorded. Supervised Gaussian/clipped and PPO squashed-Gaussian variants
  have distinct likelihood contracts.
- Motor:
  \(a_{t:t+H}=\pi_{\bar\theta}(o_t,\ell;\operatorname{repeat}_H(z_t))\).
  Primary motor: official OpenPI `gs://openpi-assets/checkpoints/pi05_libero`,
  config `pi05_libero`, native horizon ten. Already staged checkpoint receipt
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`;
  qualification on the actual allocation remains required. Execute up to ten
  actions, stopping on simulator success/time limit; count actual primitives.
  No new VLA SFT is required for this design.
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
| Paired DPT | Same actual CND or DSRL histories; identify the source route | Qualified teacher/final actor's query coordinate | Change supervision while preserving all executed context/outcomes bitwise; candidate labels are not assumed expert |
| AD | Successive real DSRL policy checkpoints, updates between complete episodes | Original executed coordinate | Distill a source learning trajectory; a collection clock is not a teacher-update identity |
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
- DPT labels cannot rewrite historical executed noise or its actual reward.
  Same source data and collection cost shared across CND/DPT.
- Planned first source route: actual advancing DSRL learning histories with
  paired AD and DPT-style query labels, preserving executed contexts bitwise.
  Source teacher learning may require many more episodes than the deployment
  shot budget. Published π₀ DSRL uses batch 256, replay warmup >500 query chunks
  and UTD20; our earlier warmup 0/batch 32/K4 smoke is not that recipe.
  Final actors remain candidates until independently qualified. Scheduled CND
  and direct RL² remain separate routes, not silently combined objectives.
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

## 5. LIBERO40 source and three target evaluations

- Source goals: **all 40 tasks in LIBERO-Spatial, Object, Goal and LIBERO10**,
  ten tasks each. Official checkpoint README → `pi05_libero` config → dataset
  `physical-intelligence/libero`: 40 tasks / 1,693 demonstrations. This audits
  release fine-tuning, not every datum of foundation-model pretraining.
  [Provenance snapshot](../../outputs/pi05-libero40-provenance-20261029-r1/PROVENANCE.json).
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
- Earlier [v1 split](../../outputs/libero-pi05-corpus-20261029-r1/split.json)
  (78/10/10 + two exclusions) and
  [clean+Plus arm](../../outputs/libero-pi05-corpus-20261029-r3/PLUS-SOURCE-PLAN.md)
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
  [LIBERO evaluation ledger](../frozen-vla-libero/EVALUATION.md).
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

## Latest development notes (2026-10-31)

- Batch14oneupdate passed exactreplay/fullrecovery/frozenmotor checks after
  56already-successfulsourceepisodes.852GPU-s/8,313actions; no targetclaim.
- Mean/std interventions onfixed9 all6/9, same percasevector. Changes are
  numericallyreal butthispanel omitsmostearlier sensitivecases.
- Explicit warm R3→native batch14 continuation completed14K4/56 episodes,
  894 decisions and two new Adam steps;55/56 source successes preceded that
  update. Exact replay/full recovery/frozen motor checks passed.9,272actions
  and910GPU-s; no target or memory benefit established.
- Current32h pool has3,929s unassigned after verified terminal reconciliation.
  SAC continuation qualification is recovering from two zero-action preflight
  faults under the same cumulative allowance. Optional diagnostics are deferred
  while source collection and fitting are prioritized. Larger source compute
  still needs approval; the central robotic-performance claim remains unproven.


Source update03:43Shanghai: the continuation bank now contains583 complete
K4 training cases/2,332 episodes/33,729 decisions across all40 goals, plus
146 validation cases/584 episodes/8,202 decisions. All224 added qualification/
bridge episodes succeeded. This supplies longer learned-policy histories,
not recovery evidence or a demonstrated performance improvement. Immutable
CPU assembly is complete. All four AD/DPT full/static interim100-update fits
were submitted within the existing32h budget; no new target score yet.
The cumulative1M-source/learner and full500-update/matched target phase remain
unfinished. Optional diagnostic branches must not serialize collection and
initial fitting again.


Interim100 update03:51: all four models completed with25/100 full recoveries;
zero new robot actions. AD full/static have effectively equal fixed-window
validation NLL; their feedback/memory paths are distinct. DPT-full logged training
minibatch NLL fell on different batches, while matched-window validation NLL
worsened25→100. Possible
overfit/distribution/calibration explanations are recorded in the summary;
no robot, memory or scaling benefit follows from these fit metrics. Next
robot readout and25h source expansion have separate explicit budgets.


Archival preparation update (before the fixed100 readout):38 source assignments plus two retained source leaves are
ready, bound to an unfunded25h source proposal. The fixed100 target profile
passed48 CPU tests including actual model-file admission, but its4.5h GPU
budget was subsequently authorized and the readout completed above. Historical AD/DPT inference already samples the
Gaussian; a deterministic-mean explanation for the performance gap was
falsified statically and no additional diagnostic was launched. Full Plus
contains10,030 registered variants; full Pro's primary map has200 tasks,
including40 environment tasks requiring official generation. The two tiny
current adapters cannot supply full-suite scores. See the
[shifted-suite inventory](../../outputs/libero-shifted-suite-inventory-20261031-r1/README.md).


Validation-label note: DPT VAL targets use independently trained validation
teachers (different seed/checkpoint/stage for all8 matching source goals).
Rising fixed-window NLL therefore includes cross-teacher generalization and
cannot alone establish worse robot performance or pure state overfit. Actual
actor-function disagreement had not yet been measured at that provenance
check. Legacy labels/checkpoint selection are unchanged. PPO continuation now passed7 allocated CPU tests including
actual checkpoint migration; no new PPO update or robot score follows.
Full Pro public files are staged, but40 environment cases still need official
generation. Source25h plus target4.5h now form the pending29.5h/150GB request;
PPO44h/arm remains separate and unfunded.

An optional CPU-only derivation subsequently measured actual teacher
disagreement on8,202 heldout-state validation observations across all8 goals:
mean absolute noise-label changes0.1403–0.6046 per goal. Eight sampled TRAIN
label replays differ by at most3.28e-7, with matching unchanged actor parameter
digests. This supports treating legacy DPT validation as teacher transfer;
it does not establish the cause of its NLL trend or any robot improvement.
Forty-nine focused tests passed on the actual0GPU allocation. Source
collection/readout remain independent; new validation-label admission is
verified by its complete assembly round trip, with AD/TRAIN data and source
cost unchanged. Exit0,348 conservative CPU-worker seconds, zero GPU time
or new actions. Preserve all legacy metrics and fixed milestones.

RoboTwin's episode-reset PPO endpoint is complete at100 rounds/600K4/
2,400source episodes/963,759source actions/500Adam steps. Target bottle8/8,
stapler0/8; six-goal source validation3/24. Training success is421/2,400,
with25-round block totals112→106→102→101 /600: no useful upward source
learning trend. Nonzero finite updates and very small approximate joint KL
are verified; the cause of weak learning is unresolved. This baseline keeps
within-episode recurrence and clears cross-episode feedback/cache. It cannot
establish memory benefit. Source-stage native probes use a different seed bank;
the final full-memory comparison is reported below; SAC remains pending. This is a two-goal
development panel, not the full benchmark. Target novelty alone cannot
explain weak source training; whether more data or a different optimization
or steering interface helps remains unresolved.
[Diagnosis](../../outputs/robotwin-reset100-diagnosis-20261031-r1/ANALYSIS.md).

RoboTwin fixed100 comparison (bottle/stapler): FULL7/8,1/8;
same-weight cache reset6/8,1/8; rewardblind8/8,1/8;
separately trained episode-reset8/8,0/8. FULL and trainedRESET both total8/16;
neither establishes a memory or reward-feedback advantage. Cache reset keeps
direct previous-noise/reward feedback, whereas trainedRESET clears it.
FULL also trains weakly:416/2400successes, quarter totals116→104→93→103 /600,
sourceVAL2/24;964875source actions/31211valid decisions/500Adam steps.
The source-learning deficit precedes target transfer. Exact optimizer or
interface causes remain unresolved, and sourceSAC is still pending.
[FULL diagnosis](../../outputs/robotwin-ppo100-diagnosis-20261031-r1/ANALYSIS.md).

The earlier qualified native16 panel scores6/8bottle,0/8stapler and is now
verified against all64 learned case records by actual seeds/raw inputs/
features plus the shared frozen-motor and normalization/action contracts.
This is a cross-run matched-case control, not a contemporaneous randomized
replication or shared-noise experiment. It supplies a descriptive native
reference without new rollouts; the modest two/three-case gains do not
establish a robust steering or memory advantage.
[Reconciliation](../../outputs/robotwin-trained-continuation-20261031-r1/NATIVE16-REUSE.md).
