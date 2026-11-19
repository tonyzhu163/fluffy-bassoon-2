# Verified findings

- On a previously studied Goal 2 bottle-placement pose, reducing realized pad/bottle friction from 2.00 to 0.95 left all measured precontact observations identical and made a regrasp fail. The fixed nominal frozen-policy tape still succeeded under both materials, tying the material-aware oracle at 2/2. Six native 300-control replays used zero GPU-hours. [Friction gate and per-control readback](../outputs/libero-xemb-contact-20261119-r1/RESULT.md)

The controlled OpenDrawer reader used one probe to choose a useful correction on measured native branches. The broader RoboCasa studies found no consistent held-out benefit from retaining history. The results below distinguish selection among executable actions, learned motor control and source-policy competence. [Controlled reader](#native-history-under-a-persistent-calibration-error) · [Multi-task comparison](#corrected-three--versus-six-task-transformer-comparison) · [Corpus diagnosis](../outputs/project-diagnosis-20261116-r1/RESULT.md)

## Synthetic learning histories

A five-condition, three-seed hidden-rule selection grid trained four-layer
Transformers for 20,000 updates per fit, then evaluated frozen weights using
each learner's own attempts on fresh scenes. Execution was perfect. On held-out
rules, the choice-and-outcome learner improved from 45.44% at k=0 to 57.05% at
k=6, while exact rule enumeration on those same histories reached 86.50%.
The training-rule gap was similarly large (58.19% versus 87.17%). All seeds
failed the original near-reference screen. Demo-only reached 45.49%, blind
choices 41.14%, full corrections 67.51%, and linked attempts/corrections 69.62%
at held-out k=6. These curves show partial history use in the synthetic task,
not robotic adaptation or learned information-seeking exploration. The
reference uses a full-family prior rather than each split's sampling prior,
and that grid did not include a same-model feedback intervention.
[Results, seed variation and limitations](../outputs/learning-histories-20261111-r1/RESULT.md).

A subsequent frozen-checkpoint intervention reproduced all six original curves
exactly and evaluated fresh paired histories. On held-out rules with independent
random attempts, real feedback reached 54.65%, masked feedback 43.94%, shuffled
feedback 38.83%, and demo-only context 44.93%. Real feedback helped in all three
seeds; both paired within-fit intervals excluded zero in each seed. Independent
random choices remove the training actor's indirect reward channel. This supports
beneficial feedback dependence in the synthetic task. Training-actor histories
raised held-out accuracy only from 57.38% to 59.15%, still below their 84.49%
enumeration reference; training rules had a similar deficit. History-policy
shift therefore does not explain most of the gap. These are frozen-query input
interventions, not alternate closed-loop rollouts or robotic adaptation.
[Paired diagnostics and limitations](../outputs/learning-histories-feedback-20261111-r1/RESULT.md).

A five-recipe, three-seed screen selected dense multi-query supervision with
Fourier features for correction reading. On fresh frozen-checkpoint evaluation
sessions, held-out k=6 accuracy rose from 67.40% to 75.64%, improving in all
three fits. The same-history enumeration reference reached 94.94%; training
rules also retained a 10.41-point deficit. The revised attempt-and-reward
learner reached 60.50% versus 57.34% originally, with one fit slightly worse
and different learner-generated histories. Every fit still failed the
within-five-points-at-every-prefix screen. This confirms a recipe improvement
on the same rule split, not a solved inference problem or robotic adaptation.
The screen changes attention, supervision and features together and does not
isolate their effects. [Fresh confirmation, all recipes and limits](../outputs/learning-histories-fix-20261111-r1/RESULT.md).

## Sampling variation on standard LIBERO-90

Across 300 matched query poses on 15 fitting tasks, two independently sampled frozen π0.5 rollouts had median wrist separations of 2.21 mm after the first five controls and 1.15 cm after 50. Both succeeded on 74 poses, one succeeded on 20, and neither succeeded on 206. Hindsight selection between complete recorded rollouts raises 28.0% average single-draw success to 31.3% on this snapshot. A per-decision selector and a history-conditioned LoRA policy require live paired evaluation. [Matched trajectories, native outcomes and scope](../outputs/libero-native-noise-20261119-r1/RESULT.md)

In the registered full-action LIBERO-90 study, the outcome and imitation readers finished fitting on 960 practice/query pairs. During development, the single-draw native baseline encountered a different motor input from the compute-matched baseline on task 40, repeat 3. The 30-task test never opened, leaving the effect of cross-episode practice on native retry success unmeasured. [Verified failure and recovery evidence](../outputs/libero90-full-action-20261119-r1/RESULT.md)

## LIBERO-Pro2.0 retry development

In a closed-loop Goal 2 experiment, true X decoding plus a grasp-conditioned regrasp succeeded on 24/24 assignments across six poses, two command conventions and BASE/D17. Correct X decoding with always or never regrasping each achieved 22/24. Regrasp rescued D17 on pose 30 and spoiled BASE on pose 33 under both wire signs. Open-jaw pulses identified X on 24/24 and gave identical wrist/jaw readings in all 12 grasp-condition pairs. Contact evidence was already present by the regrasp decision on all six measured poses. [Native motor and contact evidence](../outputs/libero-command-grasp-20261119-r1/RESULT.md)

For the same bottle instruction, released D17 and D18 produced four of twelve pose/goal pairs with two failed practices and disjoint successful query corrections. Retreat and regrasp reached 10/12 D17 queries and zero D18. The archived D18 recovery reached 4/12 D18 and zero D17. Six public wrist/jaw features chose 6/8 correct strict retries, versus 0/8 after swapping histories and 4/8 for the fixed choice. Five-frame π0.5 features chose 4/8. All paired terminal images differed. All four qualifying pairs concerned the cabinet. [Same-goal recovery evidence](../outputs/libero-temporal-recovery-20261119-r1/RESULT.md)

A reader fitted only on 16 bowl practices transferred to six known bottle development poses. With π0.5 features it selected 9/12 correct native retries after two zeros, falling to 3/12 with the matched histories swapped. General PaliGemma PT reached 6/12, pooled RGB 7/12 and a secondary six-feature wrist/jaw control 10/12 on the same fitting data. The respective all-assignment scores were 63, 59, 60 and 64/96. This supports transfer of execution information and retains a compact motion explanation. [Bowl-to-bottle transfer](../outputs/libero-witness-transfer-20261119-r1/RESULT.md)

On the paired bowl test, the LIBERO-tuned π0.5 tower, general PaliGemma PT-224 tower and time-matched pooled RGB classified 16/16, 15/16 and 14/16 development practices. All selected 4/4 strict corrections. Across 32 balanced assignments they scored 23, 24 and 24 native successes. The one-success difference came from an ambiguous dual drop and an unavailable cabinet query. [Frozen encoder comparison](../outputs/libero-witness-encoders-20261119-r1/RESULT.md)

On two of four prespecified development triples, identical bowl commands and two zero rewards called for opposite executable corrections from the same query start. RGB histories selected 4/4, the same histories swapped selected 0/4, and commands or public motion selected 2/4. Final camera views alone also selected 4/4. Across all 32 assignments of outcomes and hidden goals, the final-view reader with native feedback reached 23/32 versus 13/32 when its decision ignored feedback. Four development reset IDs, two goals and a released D18 drop support this controlled result. [Paired physical evidence](../outputs/libero-visual-witness-20261119-r1/RESULT.md)

On the same two-axis diagnostic, all three support-only MLPs that shared an estimator across X and Y recovered both signs on all 60 double-reversal variants. Three joint MLPs missed all 60, despite getting 175–177/180 familiar variants correct. Training samples, supervision and the algebraic command decoder were matched. Command-only and motion-only inputs each had identical empirical distributions across conventions for every source pair. The 60 variants reuse 15 development reset pairs. [Factorization and controls](../outputs/libero-axis-factor-20261119-r1/RESULT.md)

With physical probe commands on fresh support resets, the preregistered shared reader selected corrections yielding 64/64 native successes during open-loop playback of π0.5 tapes across 16 development starts and four constructed command conventions. Cleared support reached 16/64, opposite support 0/64, and the analytic rule 64/64. The withheld double reversal alone gave 16/16 with the reader versus 0/16 cleared. Four full simulator-state hashes differed from the collection archive. All newly paired branches shared identical starts and nominal playback reproduced all 16 archived reward sequences. [Native physical controls and limits](../outputs/libero-axis-native-20261119-r1/RESULT.md)

Across eight project-development BASE/S18 pairs with matched initial physics, S18 offset the public wrist position by approximately (+7.5, −6.0, +4.5) cm while preserving consecutive displacement within 1.21 × 10⁻⁷ m for 80 identical controls. A history reader that only sees command–displacement pairs therefore lacks a useful signal for this released additive calibration case. [Native observability test](../outputs/libero-s18-probe-20261119-r1/RESULT.md)

In a coordinate-calibration diagnostic on archived BASE trajectories, a command–motion sign rule recovered both coordinates of the first frozen π0.5 command on all 15 eligible development reset pairs with both axes reversed. A fresh reader trained on the three other conventions and balanced probe directions recovered both signs on 28, 22 and 23 of the 60 direction variants in three fits. Clearing history or supplying opposite-convention history gave 0/60 in each fit. Each physical query contributes four direction variants. These scores measure offline command recovery. [Calibration results and controls](../outputs/libero-axis-calibration-20261119-r1/RESULT.md)

A linear calibration readout after the two support tokens recovered both signs on 27, 33 and 33 of the 60 double-reversal variants, versus 25, 26 and 15 using the corresponding untrained encoders. The same readout fit at the trained query token reached 9, 11 and 22. This reused the exploratory development panel and the same three training seeds. [Frozen representation analysis](../outputs/libero-axis-readout-20261119-r1/RESULT.md)

The four-mode timing pilot had 32 fitting queries. Only 19 had any successful
action, 12 distinguished successful from unsuccessful modes, and two of
those 12 were delayed. On 32 development queries the retained reader scored
20 in each fit. Clearing or swapping support scored 20, 20 and 19. Swapping
changed up to 16 choices. The fitting-selected fixed mode scored 18, the
best fixed mode in hindsight scored 19, the lag rule scored 21, and the
measured action union scored 22. Ten of sixteen delayed queries failed every
mode. Three paced outcomes changed across workers on matched reset IDs,
making this a poor panel for resolving a one-success advantage. The evidence
establishes sparse decision supervision and limited headroom above the lag
rule on this action menu. [Timing pilot](../outputs/libero-decision-pretraining-20261119-r1/RESULT.md)

A 2,048-update timing adapter fit used 8,192 windows from 1,299 released
demonstrations across 24 fitting tasks. On eight development tasks, true
timing from reset scored 4/16 BASE and 1/16 S20, versus 16/16 BASE and
1/16 S20 for frozen π0.5 and its zero-correction solver control. Fixed
pacing scored 9/16 BASE and 5/16 S20. True timing from reset had zero wins
and one loss against its fitting-selected fixed flag. The public lag rule
identified all 32 timing settings. Held-out demonstration loss improved,
while the learned correction failed to open useful live timing-selection
headroom. [Native S20 motor comparison](../outputs/libero-s20-adapter-20261119-r1/RESULT.md#native-development-benchmark)

On the released-scene development probes, reward-aware retries scored
157/192 versus 81/192 when the selector ignored reward. Adaptive and fixed
second-probe schedules tied at 154/192 when both used two rewards. In the
separate bottle replay, repeat/switch scored 31/48, matching the ceiling
given the observable tape and reward. A motion classifier scored 28/48,
and the privileged hidden-goal ceiling was 34/48. These comparisons show
where a simple use of feedback captures the available gains and where
stronger continuations or more informative interactions are needed.
[Scene probes](../outputs/libero-scene-probes-20261119-r1/RESULT.md) · [Bottle replay](../outputs/libero-failure-diagnosis-20261119-r1/RESULT.md)

On a custom two-instruction LIBERO-Object development study, a bowl-trained
reader of commands, sensed motion and rewards reached 192/288 native successes.
Commands and rewards alone reached the same score, the ceiling of the measured
query alternatives. Motion changed 38 choices and zero native outcomes.
Both instructions placed soup in the soup scene. In the salad-dressing scene
both instructions worked and two rewards sufficed. In the ketchup scene only
the ketchup instruction placed either object. The 288 paired assignments
reused 108 simulated episodes on six reset poses per layout.
[Layout-level outcomes and intervention checks](../outputs/libero-object-history-20261119-r1/RESULT.md)

On a separate Goal 4 BASE bowl panel, opening the gripper during lift allowed
1/10 new practices to complete. The same override during descent allowed 9/10.
A reader trained on 30 earlier practices that added wrist motion around jaw
opening to public timing features earned 237/320 paired native successes versus
224/320 with the timing features. The aggregate difference came from histories
involving one placement where neither trigger fired. On the 256 assignments
where both triggers fired on both practices, wrist and timing readers earned
181 and 182 successes, versus 160.5 and 160 with same-start motion reassignment.
The 320 assignments reuse five practice poses and four query poses in fixed-command
simulator replay. [Physical intervention and transfer checks](../outputs/libero-release-phase-20261119-r1/RESULT.md)

A randomized single-retry scorer averaged 19.0/32 development successes. Giving the same transformer both measured retry outcomes for every fitting history raised the mean to 19.5 and the episode cost from 64 to 96. At equal 48-episode budgets the single, sparsely paired and fully paired fits averaged 18.25, 18.375 and 18.00. A two-row lag table fitted on the same single or full labels reached 20.25 in both equal-budget arms. The best measured candidate reached 22 and the existing lag rule reached 21. All fits reused the same eight development tasks. [Logged and branched comparison](../outputs/libero-logged-retry-20261119-r1/RESULT.md)

## LIBERO-Gen motor qualification

The released frozen Behavior Prompting Policy succeeded on 20/20 paired
native initializations for each of two bowl-to-plate choices: the center
bowl and the bowl beside the plate. Both candidates shared exactly matching
RGB/wrist/proprioceptive resets; all 967 saved tensors loaded exactly and
parameters stayed frozen. Initializations 0–4 repeated the engineering panel;
new indices 5–19 succeeded 15/15 for each candidate. This qualifies these two
motor choices on the native task family, not a released benchmark score,
custom relational layouts or history-based adaptation. No history reader
was trained. [Protocol, raw evidence and limits](../outputs/libero-gen-history-pilot-20261111-r1/RESULT.md).

The first custom-layout smoke then failed on all four tested layouts for
each prompt (0/8 physical episodes), despite correct paired resets and
unchanged frozen weights. Several trajectories transported a bowl but
never reached either native plate predicate; other trajectories reached
empty space. A targeted two-episode native-state0 replay through the same
recorder then succeeded with both candidates, matching the prior initial
observations and the smoke prompts exactly. This supports custom scene
transfer as the leading explanation, with a single-state control limit.
This is a development failure, not a history-learning result. [Custom-layout evidence and limits](../outputs/libero-gen-custom-layouts-20261111-r1/RESULT.md).

A bounded demonstration repair then obtained both intended native goals on
custom layout 0, including a corrected teacher that aligned the center bowl
above the plate before descending. In eight paired frozen-policy episodes
across that scene and a nearby scene, original prompts scored **0/4** and
repaired prompts **0/4**. Initial states, observations and per-episode seeds
matched; all 967 checkpoint tensors loaded exactly, and copied original
prompts reproduced their historical tensor hashes. Every paired action
trajectory changed with the new prompt, but neither placement nor the two
intended choices became reliable. This establishes an unsuccessful prompt-only
repair on this development pair, including the demonstration scene. It does
not establish that motor fine-tuning or another checkpoint will fail; no
history reader was trained. Demonstrations and policy episodes use the same
first-native-success endpoint, without an added released-placement persistence
criterion. [Repair results, teacher fixes and evidence](../outputs/libero-gen-prompt-repair-20261111-r1/RESULT.md).

## RoboCasa365

The Atomic-Seen baseline uses the native JAX π0.5 Human300 checkpoint at step **75,000**. Load `pi05_pretrain_human300/multitask_learning/75000` from `robocasa/robocasa365_checkpoints` at revision `c484448a`, including its bundled normalization. OpenPI predicts 50 actions and executes five before replanning, with ten denoising steps. [Pinned checkpoint](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/c484448aba1a9b60a04c9b0ca117241518ea69f3/pi05_pretrain_human300/multitask_learning/75000)

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Native Human300 atomic SFT reproduction](../outputs/robocasa-sft-reproduction-20261109-r1/ATOMIC-RESULT.md) | 363 successes in 900 episodes (40.33%) across 18 atomic tasks, versus the published 39.6%. | The original JAX evaluator, checkpoint and horizons give a credible atomic reference. The [composite continuation](../outputs/robocasa-sft-reproduction-20261109-r1/CLOSEOUT.md) closed early and is not a composite benchmark score. |
| [Paired RLinf source endpoints](../outputs/robocasa-endpoint512-acceleration-20261104-r1/RESULT.md) | On 512 paired cases per task, OpenDrawer improved from 331 to 369 successes and CloseToasterOvenDoor from 168 to 212. Both passed the two-task Holm correction. | These fixed teachers improved over SFT on their own tasks. In a [separate three-task test](../outputs/robocasa-eight-hour-babysit-20261104-r1/SOURCE5-RESULT.md), Faucet passed (204 → 242/512); Counter and Coffee did not. This does not establish five effective sources. |
| [Frozen AD steering with real prior attempts](../outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md) | On held-out Fridge after five attempts, full-history KV and velocity students succeeded on 33 and 41 of 50 cases; clearing history gave 34 and 39. | Using the frozen expert improved some transfer comparisons with a direct action head. The full/reset comparisons show no consistent benefit from retaining history; one fit per interface. |
| [Additional 10-update PPO pilots](../outputs/robocasa-six-source-decision-20261112-r1/RESULT.md) | Cabinet scored 38→40/64 (nine paired gains, seven losses); Rack 42→41/64 (four gains, five losses); CounterToStove 40→40/64 (seven gains, seven losses). Each preserves 1,280 training episodes and a verified full checkpoint/archive. | These short single-seed pilots establish collection and optimizer execution, without clear evidence of endpoint improvement. Physical/public initial states matched exactly; maximum initial RGB differences were 1, 1 and 2 intensity levels. Explicit DPT teacher-query labels remain uncollected. |
| [Complete four-method frozen KV comparison](../outputs/robocasa-kv-comparison-20261112-r1/RESULT.md) | Across12,000 audited episodes, DIT had the highest sixth-episode full-history success:55.3% ID and56.0% OOD, versus AD46.7% and44.0%. DPT scored47.3%/49.0%, IC-IQL48.0%/44.0%. | DIT exceeded SFT by+8.7pp ID/+12.0pp OOD and fixed PPO130 by+4.0pp/+6.0pp (descriptive gaps). Full-minus-reset was+4.7pp ID (95% interval−2.0 to11.3) and+6.0pp OOD (−3.0 to15.0). The larger fresh paired confirmation below did not reproduce these history gains or establish superiority over SFT/PPO. One fit/method, unequal training durations and first-query replay discrepancies limit causal attribution. Historical headline only: original collector histories are excluded from the corrected matched comparison; closed payloads are in verified cold storage. |
| [DIT first-attempt evidence audit](../outputs/robocasa-dit-diagnostic-20261113-r1/RESULT.md) | All 250 paired first attempts have identical empty entry signatures; 14 outcomes differ. In eight raw Coffee pairs, six match in every compared chunk field. One pair first differs by one RGB component after 100 controls; the success-discordant pair differs by 60 RGB components at reset, each by one intensity level, before features and commands diverge. | No recorded initial history leak was found. Rendering variation is a plausible contributor in the inspected mismatch; this retrospective subset neither establishes its causal effect nor explains all 14 cases. Exact initial pixels alone do not ensure identical later observations. |
| [AD](../outputs/robocasa-eight-hour-babysit-20261104-r1/AD-ID50-RESULT.md) and [DPT](../outputs/robocasa-eight-hour-babysit-20261104-r1/DPT-OOD50-RESULT.md) direct-action baselines | Stronger performance on the three fitted tasks than on held-out Fridge and Microwave. | These students bypass the native action expert. Their performance cannot be attributed to KV steering or taken as proof of adaptation. |
| [Flora calibration](../outputs/robocasa-joint-h600-preparation-20261102-r4/RESULT.md) and [reward mismatch](../outputs/robocasa-ppo-reward-gap-investigation-20261102-r1/RESULT.md) | Native/task-specific AUROC ranged from 0.848 to 0.982 on five fixed 50-episode, 600-control banks. A separate OpenDrawer training audit found maximum grades on 58 of 183 simulator failures. | The rewarder ranks outcomes usefully on these panels, but a high grade is not native success. The matched 50-case residual-PPO comparisons did not establish superiority over simulator reward. |

A [controlled hidden-goal pilot](../outputs/robocasa-meta-feedback-20261109-r1/RESULT.md) executed three frozen π0.5 bowl prompts on each of eight scenes. Two of 12 support rollouts placed and released a fruit, both into the left bowl on one scene. None of the 12 fresh-query rollouts released into either bowl. These development cases provided zero successful query choices for testing reward memory or first-attempt exploration. A separate drawer manipulation check changed lateral miss at closure but yielded no handle acquisition or native success on the two query tapes examined.

In the [compact-table follow-up](../outputs/robocasa-compact-bowls-20261109-r1/RESULT.md), frozen π0.5 grasped in 8/8 trials but all three released fruits went right, regardless of the requested side. A scripted post-grasp carry using simulator coordinates released to either requested side on 3/4 scenes (6/8 continuations). A fixed reward-conditioned rule on these same development traces obtained 3/4 query goals, versus 2/4 with either constant choice and 1/4 with inverted reward. There were only two scene pairs; the original motor did not qualify for held-out evaluation.

On eight fresh compact-bowl scenes, a shared neutral frozen pickup followed by a privileged scripted carry produced both target-specific releases on seven. Under balanced hidden goals, choosing the correct carry reached 14/16 and either fixed carry reached 7/16. A leave-scene-pair-out diagnostic decoded the hidden goal from public support motion and terminal reward on 15/16 cases, versus 8/16 with reward masked. Using the decoded choice to select between the two physically executed query branches scored 13/16 with real reward, 7/16 with reward masked and 1/16 with counterfactual reward. The two ambiguous cases came from one timed-out support. This qualifies the action labels and the information in support for a learned-motor test. [Exact-fork qualification and decoder](../outputs/robocasa-bowl-dpt-labels-20261110-r1/QUALIFICATION.md).

In the subsequent learned-carry test, 24 training scenes supplied 956 masked action chunks, and eight development scenes selected 1,024 updates for both LoRA students. On 16 held-out query scenes crossed with both rewarded sides, DPT reached 3/32 matching-history goals, 5/32 with an opposing-goal donor, 4/32 with counterfactual reward and 5/32 with reward masked. Memoryless LoRA scored 5/32 and frozen π0.5 4/32. Matching history gained zero paired outcomes and lost two against its donor. Both scripted releases were feasible on all 16 scenes. A decoder fitted only on training scenes read 32/32 goals from public support motion and reward, and a linear probe of the selected DPT representation read 17/32. This one-fit binary pilot found no matching-history benefit in physical control. [Locked confirmation and protocol](../outputs/robocasa-bowl-dpt-labels-20261110-r1/RESULT.md).

The property-given LoRA control selected 128 updates on the same development scenes and reached 7/32 confirmation goals. It reproduced 2/8 fixed training goals, and its existing 1,024-update checkpoint reproduced 3/8. A teacher-assisted handoff after the shared lift reached 2/8. Continuing the same instructed fit with restored optimizer and sampler to 4,096 updates reached 8/8 training goals, gaining five and losing zero against 1,024. This establishes reproduction on those four training scenes. Held-out evaluation of the extended checkpoint is the next step. [Motor fitting diagnostics and resumption evidence](../outputs/robocasa-bowl-dpt-labels-20261110-r1/RESULT.md).

In an eight-seed OpenDrawer screen, two scenes opened under the frozen native policy. Both exact nominal replays matched. Both scenes still opened with either sign of a five-control, 4 cm lateral offset and zero correction. Nineteen of 20 branches opened across all five residual choices. The only timeout came after adding a correction that reinforced the offset. All four uncorrected branches succeeded, leaving no failures on which to measure rescue. [Matched drawer results](../outputs/robocasa-drawer-authority-20261110-r1/RESULT.md).

A held lateral correction opened two of six natural OpenDrawer failures. Both rescued at −2 cm, the same sign as their nominal lateral misses. The best fixed offset matched the best per-scene choice. All eight archived command tapes replayed exactly, all eight native outcomes reproduced under fresh inference, and all six matched zero-offset continuations failed. [Paired evidence and videos](../outputs/icrl-idea-checks-20261110-r1/RESCUE.md).

On two OpenDrawer scenes crossed with two noise seeds, normal effort and doubled post-grasp translation each opened 3/4 drawers at normal damping, 3/4 at 8× and 0/4 at 64×. Three pairs still grasped at 64×. The added translation changed no paired native success outcome. [Stiffness measurements](../outputs/icrl-idea-checks-20261110-r1/STIFF.md).

On eight development scenes per task, four native Human300 π0.5 retries from exactly matched starts gave scene ICCs of 1.00 for CloseDrawer, 0.885 for OpenDrawer, 0.222 for CloseFridge and −0.067 for TurnOnMicrowave. All eight CloseDrawer scenes had unanimous outcomes, including three that failed on every attempt. Seven of eight OpenDrawer scenes had unanimous outcomes. This measures repeatability at a fixed starting scene. The effect of support history on a fresh query remains open. [128 native episodes and per-scene counts](../outputs/icrl-idea-checks-20261110-r1/PERSISTENCE.md).

Across eight CloseDrawer development scenes and two noise seeds, the native policy, a stationary wait, and either lateral base movement each closed 10/16 drawers. A forward move closed 7/16, losing three native successes. No movement or wait rescued a native failure. A per-scene oracle also closed 10/16. Of the three recurrent failures, one never made gripper contact, one barely moved the drawer after contact, and one reached about 60% open before stalling. [Matched corrections and camera evidence](../outputs/robocasa-closedrawer-corrections-20261110-r1/RESULT.md).

A memoryless LoRA student trained on the final Drawer130, Toaster130 and Faucet40 PPO actions reduced held-out native flow-matching error from 0.08113 to 0.07887. DPT + LoRA scored 0.08207 with the matching prior SFT attempt and 0.08262 with a different same-task attempt. The paired 95% interval for the own-history improvement included zero across 48 held-out reset scenes. The pilot measured offline imitation with one fit per arm. [Three-teacher RoboCasa pilot](../outputs/robocasa-three-ppo-dpt-lora-20261110-r1/RESULT.md).

On 96 fresh RoboCasa365 scenes with one shared frozen-policy first attempt, two-epoch memoryless LoRA succeeded on 60 retries, frozen π0.5 on 40, and DPT with own versus same-task donor history on 37 versus 54. On the 52 first-attempt failures these four arms recovered 27, 19, 12 and 21, respectively. Selecting training duration on 24 development scenes chose one epoch for DPT and two for memoryless. On the other 72 scenes, matching DPT reached 38, donor DPT 41, and memoryless 42. Own history gained six and lost nine paired cases against donor history. Native checkpoint selection improved DPT success without establishing a memory benefit. [Paired native retries and three-seed learning curves](../outputs/robocasa-three-ppo-retry-20261110-r1/RESULT.md).

For historical Flow-SDE teachers, the [critic parity audit](../outputs/robocasa-flow-sde-critic-parity-20261104-r1/RESULT.md) found that rollout and replay could use different denoising-step values even when likelihoods matched. Record the RLinf revision and critic definition when comparing these runs.

### Frozen DIT paired confirmation

The same frozen DIT checkpoint was evaluated on 256 fresh paired cases per task,
with the sample count fixed before outcomes. All 1,280 cases and 11,520 physical
episodes passed the archive audit. The endpoint is episode-six success; DIT
retains five prior attempts in the full-history arm and starts with empty replay
in the reset control. ID comprises Drawer, Toaster and Coffee; OOD comprises
Fridge and Microwave relative to the meta-training corpus.

| Panel | DIT full | DIT reset | SFT | PPO130 |
| --- | ---: | ---: | ---: | ---: |
| ID | 43.23% | 44.92% | 41.15% | 49.74% |
| OOD | 46.09% | 44.14% | 43.36% | 49.41% |

| Full DIT minus control | ID difference (pp), paired 95% interval | OOD difference (pp), paired 95% interval |
| --- | ---: | ---: |
| Reset: history effect | −1.69 [−4.43,1.17] | +1.95 [−1.17,5.08] |
| SFT | +2.08 [−1.17,5.34] | +2.73 [−1.17,6.64] |
| PPO130 | −6.51 [−10.29,−2.60] | −3.32 [−7.23,0.59] |

The larger panel does not confirm a history benefit or an advantage over SFT.
Full DIT trails PPO130 on ID, with its interval below zero; the OOD comparison
remains uncertain. Every per-task full/reset interval also includes zero.
The earlier small-panel history gain is therefore not replicated, rather than
evidence that the true effect is exactly zero. These are marginal, unadjusted
task-stratified paired bootstrap intervals conditional on one DIT fit and these
five tasks. They do not establish training-seed robustness, explain the original
method ranking, or resolve the corrected matched four-method comparison.
[Audited result](../outputs/robocasa-dit-paired-confirmation-20261114-r1/RESULT.md) ·
[Coverage](../outputs/robocasa-dit-paired-confirmation-20261114-r1/COVERAGE.json) ·
[Figure (PDF)](../outputs/robocasa-dit-paired-confirmation-20261114-r1/figures/frozen-dit-paired-confirmation.pdf).

### Corrected six-task PPO30 source baselines

All six fixed-step30 teachers now have verified 64-case paired comparisons
against SFT, using native success and 600-control horizons. Five panels were
recovered from saved source evaluations; OpenDrawer was evaluated on a fresh
paired bank after its orderly training exit skipped the original final panel.
The endpoint was fixed before outcomes.

| Task | SFT successes /64 | PPO30 successes /64 | Paired gain (pp), descriptive 95% interval |
| --- | ---: | ---: | ---: |
| OpenDrawer (fresh panel) | 37 | 48 | +17.19 [6.25,29.69] |
| CloseToasterOvenDoor | 19 | 15 | −6.25 [−17.19,4.69] |
| CoffeeSetupMug | 10 | 21 | +17.19 [3.13,31.25] |
| PickPlaceCounterToStove | 40 | 38 | −3.13 [−12.50,6.25] |
| TurnOffStove | 7 | 4 | −4.69 [−12.50,3.13] |
| TurnOnSinkFaucet | 24 | 28 | +6.25 [−3.13,15.63] |

OpenDrawer gains14 cases and loses3 relative to SFT on the same fresh64 cases;
its old SFT33/64 panel is separate and is not paired or pooled with this result.
The repair completed128 physical episodes, with every archive read back and all
paired case identities verified. Faucet uses its saved intermediate step30
panel, not its step40 final panel. [Drawer paired evidence](../outputs/robocasa-six-task-scale-20261113-r1/evidence/OPEN-DRAWER-BASELINE-RESULT.json) · [Saved-panel audit](../outputs/robocasa-six-task-scale-20261113-r1/evidence/SAVED-BASELINES-AUDIT.json).

Drawer and Coffee show positive gains with intervals above zero; the other four
intervals include zero. These are one training seed per task and marginal,
unadjusted paired bootstrap intervals, not a six-task superiority claim.
Collection completion alone does not establish uniformly stronger teachers.
These source comparisons do not measure held-out meta-learning or history gains;
the completed fixed-final transformer readout is reported below.
[Six-task baseline summary](../outputs/robocasa-six-task-scale-20261113-r1/SIX-TASK-BASELINES.json).
[Paired baseline figure (PDF)](../outputs/robocasa-six-task-scale-20261113-r1/figures/six-task-baselines.pdf).

### Corrected three- versus six-task transformer comparison

All eight fixed-final AD/DPT/DIT/IC-IQL fits and the complete13,200-episode
comparison finished. The audit verified25 cold archives, all40 paired cases per
task, and no missing slots or outcome-conditioned exclusions. The effective
panels are120 ID cases across Drawer/Toaster/Coffee and80 OOD cases across
Fridge/Microwave; repeated physical episodes are not independent cases.

| Method | Training tasks | Full-history sixth success, ID | OOD |
| --- | ---: | ---: | ---: |
| AD | 3 | 45.00% | 36.25% |
| DPT | 3 | 50.83% | 37.50% |
| DIT | 3 | 41.67% | 36.25% |
| IC-IQL | 3 | 49.17% | 43.75% |
| AD | 6 | 45.00% | 40.00% |
| DPT | 6 | 45.00% | 42.50% |
| DIT | 6 | 46.67% | 40.00% |
| IC-IQL | 6 | 47.50% | 41.25% |

Same-case SFT scored49.17% ID/41.25% OOD and PPO30 scored52.50%/40.00%.
No aggregate model-versus-baseline contrast establishes a positive advantage.
Three-task DIT was below PPO30 on ID by−10.83pp [−19.17,−2.50]. PPO30 is
task-matched on ID; both OOD tasks use the fixed OpenDrawer30 teacher, so this
is a single-source transfer control. These are separate cases from the64-case
source panels above, and separate checkpoints from historical PPO130.

The sole positive aggregate history interval is three-task IC-IQL on ID:
**+7.50pp [0.83,15.00]**, with14 wins,5 losses and101 ties. Its OOD effect is
+3.75pp [−3.75,11.25]; six-task IC-IQL gives−1.67pp [−9.17,5.83] ID and
+2.50pp [−5.00,10.00] OOD. No model has an OOD history interval excluding zero.
All eight three-to-six-task full-success intervals and all eight corresponding
history-change intervals include zero. DIT is not consistently the strongest
method in this corrected comparison.

These are marginal95% task-stratified paired bootstrap intervals, with one fit
per cell and no multiplicity correction or training-seed uncertainty. Budgets
match10,000 actor updates; DIT/IC-IQL additionally receive1,000 Q/V updates.
Adding tasks changes diversity, data volume and per-task exposure together.
OOD is relative to meta-training tasks, not necessarily base-VLA pretraining.
First full/reset outcomes disagree on2–7/120 ID and2–12/80 OOD cases depending
on the fit, despite passing the prescribed state/RGB-tolerance and reset-action
replay checks. The causes were not isolated; these discrepancies limit causal
attribution to history. All cases remain included.

The recommendation is **stop scaling this recipe in its current form**:
the isolated IC-IQL interval does not justify collecting more teachers without
independent confirmation and an advantage over stage controls. This is a
bounded resource decision, not proof against meta-learning generally. All GPU
jobs are closed at219.5/230 conservative GPU-hours; valid models, data and
evidence remain preserved.
[Decision and full comparisons](../outputs/robocasa-six-task-scale-20261113-r1/RESULT.md) ·
[Audited statistics](../outputs/robocasa-six-task-scale-20261113-r1/matched-finalize01/analysis/RESULT.json) ·
[Coverage](../outputs/robocasa-six-task-scale-20261113-r1/matched-finalize01/analysis/COVERAGE.json) ·
[History figure](../outputs/robocasa-six-task-scale-20261113-r1/matched-figures02/matched-performance.pdf) ·
[Scaling figure](../outputs/robocasa-six-task-scale-20261113-r1/matched-figures02/matched-scaling.pdf) ·
[Baseline gaps](../outputs/robocasa-six-task-scale-20261113-r1/matched-figures02/matched-baseline-gaps.pdf).

### Frozen DIT advantage and history diagnostic

The completed CPU diagnostic found uneven PPO quality and weak signals in the
current learner. Each source task contains537–2,179 successful training episodes
out of3,360, so successful behavior is present. First-to-last-ten-update success
changes are modest and vary by task; these collection rates are separate from
paired SFT/PPO30 evaluations.

On all16 validation streams/task at four fixed collection updates, DIT weights
are nearly uniform: effective sample size is99.50–99.86% of the panel size.
They change the all-query successful-example share by−0.453 to+0.188 percentage
points per task. No first-query DIT advantage AUC interval is wholly above0.5;
IC-IQL also has no first-query interval above chance. Some Q estimates predict
eventual success, but Q−V has weaker outcome ranking. This is observational
evidence, not a counterfactual test of individual action quality.

The all-query success label includes the suffix after the one success reward
is spent. Those queries have zero remaining return; their outcome ranking is
not an action-quality test. The later pre-success analysis below qualifies
that interpretation. First-query statistics are unaffected.

Archived native flow checks show DIT full/reset history differences of0.234%
and0.214% of native output RMS for the three-/six-task fits. The corresponding
relative imitation-loss improvements from history are+0.0056% and−0.0147%.
These are flow predictions on128 held-out queries per actor, not new simulator
success rates. Together with the native readout above, this favors qualifying
advantage weighting and useful history influence before more teacher collection;
it does not establish that stronger PPO would fix the result.

All four frozen Q/V models,13,824 model-query records and144 independent
statistic checks passed. Batched inference matched production stepping within
3.58e-7. The panel contains4,608 unique source queries in384 episode records;
uncertainty resamples whole streams, with one fit/model and unadjusted intervals.
No GPU-hours were added. [Diagnostic and limitations](../outputs/robocasa-dit-diagnostic-20261115-r1/RESULT.md) ·
[Complete statistics](../outputs/robocasa-dit-diagnostic-20261115-r1/evidence/analysis01/RESULT.json).

### Project data, training exposure and frozen-value diagnosis

A further CPU audit found 30 distinct reset seeds in every source stream,
while the native evaluation repeats one canonical scene across attempts.
Training and testing therefore differ in the scene structure of their
histories. Current initial-query pooled features identify source task with
99.375% accuracy on 2,880 held-out queries using fixed training centroids.
This measures visible task identity, not absence of useful scene-specific
unknowns or policy competence.

Reconstructing the exact actor sampler, with final RNG agreement against all
eight fits, gives 80,000 sampled queries per fit. Unique-query coverage is
47.54% for three tasks and 27.89% for six. Respectively 21.42% and 19.82% of
samples follow first success; 34.10% have no preceding complete episode.
This establishes unequal data exposure at the fixed update budget, not that
more training or success masking would necessarily improve native outcomes.

Removing post-success queries from the outcome-ranking proxy reveals modest
positive six-task DIT AUC intervals on Toaster and Faucet, and a negative one
on CounterToStove. DIT weights remain nearly uniform (ESS 99.43–99.83%).
In a separate frozen-value intervention, substituting another stream's action
increases observed-return prediction error in all eighteen task/model panels,
with nine positive marginal intervals. Substituting prior history has mixed
effects: one positive and two negative Q-error intervals among eighteen.
Thus some action information is learned, without consistent evidence that the
particular prior history improves prediction. Donor actions lack measured
counterfactual returns, so this does not validate action-value ranking.

All inputs were hash-verified; repeated Q/V predictions agree exactly with the
earlier diagnostic. Independent raw-record recomputation checked 588 statistics
and count identities. Both stages exited zero, used no GPU-hours, and their
1.44 MB of raw evidence was cold-archived with verified readback. The measured
limitations support stopping scale-up of the current recipe and first testing
whether matched history supplies useful information for attainable actions.
[Diagnosis, method differences and proposed next experiment](../outputs/project-diagnosis-20261116-r1/RESULT.md) ·
[Corpus evidence](../outputs/project-diagnosis-20261116-r1/evidence/corpus01/RESULT.json) ·
[Value interventions](../outputs/project-diagnosis-20261116-r1/evidence/value01/RESULT.json).

### Native history under a persistent calibration error

A controlled OpenDrawer pilot establishes useful history for a small causal
correction selector. The frozen Human300 pi0.5 motor receives a hidden signed
4 cm approach offset; the reader chooses left, no correction or right. Eight
nominally successful scenes train the reader and eight are held out. At the
fixed 1,024-update endpoint, averaged over three fit seeds:

| Reader | Prior probe action and reward | Uncorrected-attempt motion history |
| --- | ---: | ---: |
| DPT teacher labels | 95.83% | 100% |
| DIT, tau 1 | 50% | 100% |
| DIT, tau 0.1 | 58.33% | 97.92% |
| Measured-outcome weighting | 92.71% | 100% |
| Memoryless teacher-label control | 50% | 50% |

These are held-out second-attempt success rates scored against measured native
candidate outcomes. Clearing the main DPT reader's history reduces 95.83% to
50%; flipping its support reward reduces it to 4.17%. Main-version critic AUC
is 0.748–0.800, versus 0.998–1.000 with motion history. The latter variant also
changes the support policy, so the difference is not a single-feature ablation.
The results support usable history and identify estimated DIT weighting under
sparse feedback as an unresolved limitation; they do not justify new broad
PPO collection or establish a unique critic failure mechanism.

This is one task family, a hand-designed offset, a privileged geometry trigger,
and a three-action selector, not the earlier large VLA adapter or a multi-task
ID/OOD comparison. The 32 primary constructed trials share only eight held-out
scenes and two dependency blocks. Training data used 4090 and held-out data
Pro6K. All 16 tape replays match exactly; fresh closed-loop repeats preserve
success in all 16 and terminal physics in 14. The audit passed 1,333 checks;
the completed phase charged 6.578 of 8 GPU-hours. The earlier multi-task null
results stand. [Full results, controls and limitations](../outputs/robocasa-feedback-advantage-20261117-r1/RESULT.md).

On a separate OpenDrawer development panel, the full history adapter given the true condition and privileged timing succeeded on **3/8** fresh episodes, the uncorrected SFT motor on **2/8**, and the privileged teacher on **8/8**. This measures the learned motor interface with the answer supplied. The panel and action interface differ from the three-choice reader above. [Execution record](../outputs/robocasa-full-history-adapter-20261117-r1/EXECUTION.md) · [Command diagnostics](../outputs/robocasa-full-history-adapter-20261117-r1/AUTHORITY.json)

## Procedural LIBERO

Eight training layouts fixed before simulation passed construction, initial contact,
settling and object-centre visibility checks; all 48 dropped-bowl controls satisfied
only the intended receptacle predicate. Those controls establish predicate semantics,
not policy execution or robot reachability. Recomputing the 2,912 hidden rules on
each settled layout changed 3,040 of 23,296 nominal labels across five layouts.
Rendered inspection also found three partly cropped bowls. History targets must be
bound to realized reset geometry, and centre visibility alone does not establish
full-object visibility. [Geometry evidence and limits](../outputs/libero-procedural-qual-20261111-r1/GEOMETRY-RESULT.json).

## RoboTwin

### Sidney phone-stand baseline

All three policies start from `SidneyXie/pi05_robotwin` at revision `e49e2ab6`. The frozen evaluation uses the native per-target controller, a 1 cm near clipping plane on both wrist cameras, ten denoising steps, 50-action chunks and a 400-target episode limit. Within each domain, all three policies use the same 100 scenes, prompts and sampling seeds. [Pinned checkpoint](https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416) · [Matched evaluation](../outputs/robotwin-sidney-native-endpoints-20261109-r1/RESULT.md)

| Policy | Clean successes / 100 | Randomized successes / 100 |
| --- | ---: | ---: |
| Original SFT | 75 | 87 |
| PPO after 40 Adam updates | 77 | 88 |
| GRPO after 40 Adam updates | 78 | 90 |

**Training volume per algorithm** was two collections × 1,024 trajectories × 400 action slots = **819,200 nominal action steps**, or 16,384 chunks of 50. Each policy received 40 synchronized Adam updates. The training journal counted 842,240 slots including 23,040 setup/reset allowances. Actual delivered motor calls during training were unmeasured. [Training accounting](../outputs/robotwin-sidney-matched-baselines-20261106-r1/RESULT.md)

The subsequent frozen evaluation delivered **34,533 targets for PPO** and **33,768 for GRPO** across 200 episodes each, with zero further updates. The 1–3 point gains over SFT had two-sided exact paired p-values of at least 0.45. [PPO readback](../outputs/robotwin-sidney-native-endpoints-20261109-r1/PPO40-READBACK.json) · [GRPO readback](../outputs/robotwin-sidney-native-endpoints-20261109-r1/GRPO40-READBACK.json)

At the original 10 cm wrist clipping distance, SFT scored 23/100 clean and 31/100 randomized. Restoring the fingertips visible in training recovered 55 clean failures and lost three successes. On randomized scenes it recovered 58 and lost two. The reference loader matched all 813 released tensors. The uploader reports 66/100 clean on a separately evaluated cohort. [Camera experiment](../outputs/robotwin-sidney-gap-20261109-r1/RESULT.md)

On clean phone-stand seeds 100000 and 100001, two independent native SAPIEN replays per seed reproduced the saved Sidney success on action 128. All three camera views, public joints, articulation positions and velocities, and phone and stand poses were bitwise identical at all 129 observed boundaries, including the action-50 fork. [Replay evidence](../outputs/robotwin-e1-replay-20261110-r1/RESULT.md).

### Motus and other RoboTwin results

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Released Motus native phone baseline](../outputs/robotwin-motus-release-baseline-20261109-r1/RESULT.md) | 27/100 clean and 25/100 randomized successes on `place_phone_stand` under the documented reconstruction. | Below the reported 49%/53% for that task. This is one motor and controller protocol, not the 50-task aggregate. |
| [Corrected Motus source confirmation](../outputs/robotwin-restoration-20261106-r1/MOTUS-SOURCE-GRPO-CONFIRMATION01-READBACK.json) | SFT 7/64 vs. selected GRPO 10/64 on disjoint paired scenes; eight gains, five losses, exact two-sided p = .581. | The predeclared improvement gate failed. Working PPO/GRPO execution is not evidence of an improving source or transferable adaptation. |
| [Teacher-forced action replay](../outputs/robotwin-motus-replay-20261109-r1/RESULT.md) | On 20 clean demonstrations, mean shift-one arm error was 0.0154 rad versus 0.0789 for holding state. Wrist-camera swaps increased error. | The frozen motor predicts useful actions on expert observations. This offline measurement does not explain the closed-loop success gap. |

The [decoded-learner audit](../outputs/robotwin-restoration-20261106-r1/CPUDECODEDFIT01.json) and [four-method steering qualification](../outputs/steering-four-method-pipeline-20261103-r1/RESULT.md) test objectives, causal context and recovery. They are software qualifications, not held-out performance scores.

The [full pre-consolidation evidence ledger](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/24fcc5b016f6d6f15cd7bad09cec9bd723873338/docs/FINDINGS.md) preserves detailed benchmark results, negative findings and citations for retired experiments. Consult a run’s own source and evidence before comparing results across protocols.

## Agent-collected RoboCasa context

A local Codex CLI selecting `gpt-6-astra` controlled native RoboCasa365 through
SSH/MCP and completed OpenDrawer, pretrain, seed 7 with hidden native feedback.
Native success first occurred at control 210 and remained true at the final
control 220. The recorder exported 15 pre-action context/action decisions,
221 observations and 663 camera images with verified alignment. This is one
successful integration example; no downstream VLA training or cross-task
success estimate was measured. [Run, replay and dataset evidence](../outputs/astra-robocasa-demo-20261112-r1/RESULT.md).

The direct Responses API client also passed synthetic and native RoboCasa MCP
qualification with `gpt-6-astra`: five calls per smoke, image tool outputs,
two recorded controls and an aligned causal export. The native smoke checked
all nine camera-image hashes and used 0.016101 conservative GPU-hours. This
qualifies the API transport; it did not attempt manipulation or learner training.
[API qualification evidence](../outputs/agent-sim-api-debug-20261112-r1/RESULT.md).

A paired teacher-update pilot used frozen Human300 pi0.5 on four fresh OpenDrawer
scenes (3/4 native successes under a 500-control budget). Astra abstained in all 72
judgments across current-only, actual-history and donor-history inputs, leaving
zero estimable teacher-change targets. Six native candidate pairs qualified;
one had different success outcomes. Thirteen of sixteen planned branches were
verified, including seven exact baseline continuations. The remainder stayed
unqualified after an initial-reset rejection and action drift on motor restart.
The experiment also reproduced and locally repaired camera corruption caused
by rendering-context cleanup. No TTT learner was trained. This is a negative
result for raw five-control candidate ranking under the tested prompt, not a
test of learned capability transfer. [Pilot results and limitations](../outputs/teacher-update-pilot-20261112-r1/RESULT.md).
