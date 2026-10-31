# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-10-29
Asia/Shanghai** (DSRL smoke audited at 12:59; benchmark decision: LIBERO-Plus/PRO and Meta-World). Keep current decisions and jobs here, method
contracts in `docs/`, verified results in `EVALUATION.md`, and dated evidence in
`outputs/<run-id>/`. Follow [AGENTS.md](../AGENTS.md) and the
[retention policy](../README.md#artifact-retention).

The previous [cutover placeholder](archive/coordination/coordination-pre-cutover-20261028.md)
is archived. Older `fluffy-bassoon` runs retain their immutable source-pinned
worktrees, artifacts and [source-repository records](https://github.com/tonyzhu163/fluffy-bassoon/blob/lycoreco_rl/outputs/coordination.md).
New-repository job status lives here; no active run is migrated.

## Question and decision

Can a persistent CTM learn from interaction history with fixed deployment
weights, and does its recurrence offer an advantage over conventional memory?
Separate architecture performance from motor competence, steering headroom,
informative history and optimization.

- If XL adapts and CTM does not, retain the qualified pipeline and use the
  stronger learner while revising the CTM claim.
- If neither adapts, diagnose the interface, histories and training signal
  before attributing failure to memory.
- Judge adaptation with paired fresh probes, trained static/reset controls
  and valid history interventions. Dynamics identification can help without
  scalar reward; reward interventions establish the contribution of feedback.
- The fallback directions below are research candidates. Existing run
  contracts and resource caps remain in force.

Contracts: [support/probe](../docs/ctm-support-probe.md),
[feature bridge](../docs/pi05-feature-bridge.md),
[PPO replay](../docs/pi05-bridge-policy.md).
Results: [meta-RL evaluation](../src/fluffy_bassoon/meta_rl/EVALUATION.md).

## Three fallback ideas

### 1. In-context strategy selection over a frozen VLA

**Best practical pivot.** Let the motor supply competent behavioral
alternatives. A small history-conditioned learner selects a skill, approach
variant or low-dimensional steering latent using actual attempts, outcomes
and rewards. Freeze both motor and selector during evaluation; keep latent
choices coherent over a skill or attempt so outcomes can be attributed.

First qualify headroom: an oracle selecting among available behaviors must
substantially beat the default motor on paired scenes. Then compare with a
trained static selector, random choice and a simple bandit/search baseline
at equal interaction budgets. One choice per attempt is a useful bandit
proof; multi-stage decisions support a stronger sequential-learning claim.

[DSRL](https://diffusion-steering.github.io/) already learns noise-space
steering and [VLRLA](https://arxiv.org/abs/2610.19517) already demonstrates
reward-bearing recurrent adaptation. The proposed contribution must concern
efficient exploration, avoiding repeated failures and transfer of evidence,
rather than simply combining memory with latent steering.

Concrete plan (2026-10-29): [noise-mode selection over frozen
pi0.5](noise-mode-icrl-20261029-r1/PLAN.md). Fixed initial-noise arms under a
hidden per-trial condition, with headroom, heterogeneity, consistency and
transfer gates before any learner training. Hidden object physics (C2) is the
primary condition, because only a condition the observation does not reveal
separates in-context RL from observation-conditioned
[DSRL](https://diffusion-steering.github.io/). DSRL static and DSRL
per-condition fine-tuning at K = 8 are required baselines.

**Benchmark decision (2026-10-29, user):** run idea 1 on **LIBERO-Plus/PRO**
with **Meta-World** as the hidden-goal companion; ManiSkill Stage 0 becomes
optional. Plan: [noise steering on LIBERO-Plus/PRO and
Meta-World](libero-noise-icrl-20261029-r1/PLAN.md). It keeps the arms and
gates above, replaces the pilot of 78 per-task DSRL teachers on clean
LIBERO-90 with a pi0.5-base motor trained only on the 78 training tasks and a
headroom screen per perturbation type, and fixes the
meta-learning tests: within-trial gain, memory-reset, reward-shuffle and
wrong-task controls, a bandit and static multitask DSRL at equal budget, and
per-task DSRL as the gradient reference. Better held-out success alone counts
as generalization, not meta-learning.

**Run order (2026-10-29, user: Meta-World and LIBERO stages may run
concurrently; released motors first):**

1. Meta-World with the released `RLinf/RLinf-Pi05-MetaWorld-SFT`, no
   training: noise-arm screen over four discrete goals with the marker visible
   (headroom) and hidden (mode spread). This is the kill test for the noise
   interface.
2. In parallel, LIBERO L0 with the released RLinf LIBERO motor on LIBERO-Plus.
   The released motors trained on all 130 LIBERO tasks, so the motor's
   out-of-distribution test is LIBERO-Plus/PRO; held-out tasks are new to the
   controller only.
3. Meta-World cached arm tests (RL²/DPT selectors, T1–T5): the first
   meta-learning result.
4. LIBERO L2 cached tests, then live per-chunk steering only if warranted.
5. Later (user, 2026-10-29): our own motors, Stage M0 on the 78 training tasks
   and, if hidden-goal mode spread fails, a goal-hidden Meta-World fine-tune.

Each job's GPU budget is still fixed with the user at launch.

### 2. Task-directed physical experimentation

**Strongest scientific target.** Learn when an interaction is worth its cost
because it improves subsequent task decisions. Start with a small set of
competent probes and ordinary task behaviors; add WAM imagination after
showing that better evidence collection improves return.

Hold an object's unknown mass distribution or friction fixed while changing
the goal, probe cost or remaining attempts. A lift may be valuable for precise
placement and unnecessary for a coarse push. The learner should change
whether and how it probes as the value of information changes. Count every
probe and failed attempt in total trial return.

Separate evidence collection from evidence use with the
[collector-by-reader comparison](../docs/ctm-support-probe.md#separate-evidence-collection-from-evidence-use).
Compare no/fixed/random probes and generic uncertainty reduction with the
task-dependent choice. [ICWM](https://arxiv.org/abs/2606.26025) already uses
interaction context for system identification;
[ASTEROID](https://arxiv.org/abs/2610.30105) offers on-policy teacher
supervision for exploration. The proposed target is the downstream value
and cost of information, beyond exploration alone.

### 3. Reward-sensitive WAM decision calibration

**Conditional on a competent action-conditioned WAM.** Store executed
actions, predicted consequences, observed consequences and rewards. Use
context to correct candidate-plan rankings across attempts, with all
deployment weights frozen. Start with decision correction rather than
requiring accurate future-video reconstruction.

Hold out conditions where the base model systematically misranks plans.
Test whether a few real outcomes improve rankings and control at the same
candidate/computation budget; compare simple numerical residual calibration.
Imagined outcomes remain predictions, not additional observed evidence.
Verify specified-action conditioning before treating a joint video/action
generator as a counterfactual simulator.

[Imagine-RL](https://arxiv.org/abs/2609.24033) already uses past prediction
residuals to weight imagined futures in a critic;
[PhysiCalWorld](https://arxiv.org/abs/2610.30125) calibrates world models from
interaction context. The sharper hypothesis is persistent, reward-sensitive
decision correction across attempts with frozen deployment weights.

**Preferred sequence:** qualify strategy-selection headroom, establish
adaptation with a strong simple learner, then study task-dependent probing.
Execution correction is a useful control experiment; selective memory
revision is a later extension.

## Ongoing work and pending checks

All times are **2026-10-28 Asia/Shanghai**. The native PPO and balanced
support/probe rows were audited from saved artifacts at **21:48**; other
rows retain their individually stated checks, without a fresh scheduler query.
[Status receipt](coordination-refresh-20261028-r1/jobs-1923.json) contains
exact source commits, paths and observations. A missing terminal receipt
does not alone establish that a job remains running.

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| RoboTwin matched noise steering | CPU export/admission COMPLETE; AD GPU work deferred for ordinary-baseline priority | **Closed2026-10-31:** Baige job-vth2qv6r4lxu/podSuccess/exit0,13:17:19→13:38:49Shanghai=1290CPU-worker-s/0GPU; parent1272.080/pod1278separate. Exactaux7332/sourcee567/profile4613. ADtrain32425decisions/2400E/600K4/1003790historicalactions; BCVAL328/24/6/10180. Actual146softwarePASS adopted,newtests0; allallocatedphases/finalcleanupclean/sessions[]survivors[]. Root2index+2NPZSHA/42CRCmembersPASS; independent606caseIDs/source-VAL actualseed/initialobsdisjointness andancestryprovenancePASS. No newGPU/robot/VLA/SGD, teacher_usedfalse, no student/performance/ICRLclaim. CPUgrant2588actual/7200,4612unusedSEALED; conservative96727780397Bwithin128GiB. Largepayloads/indices,paidR1projection/allfailedrecoveries remainprotected. AlladmittedCPUworkterminal/costsettled; CPUmonitorDELETED/appconfirmed. NEW ADGPU10800s/60000targetactions remainUNSPENT/DEFERRED for newerdirecthuman priority DSRL/SACfirst,nativePPOafterward; source root/existingowners solebaselinelaunchers,noobserverfromthisclosedCPUtask. Original3learner100roundcampaign/all evals complete/source monitordeleted; neverresume. [CPU final report](robotwin-final100-ad-export-20261031-r4/RESULTS.md); [closure](robotwin-final100-ad-export-20261031-r4/CLOSURE.json); [current handoff](robotwin-final100-ad-export-20261031-r1/CURRENT.md); [source campaign](robotwin-trained-continuation-20261031-r1/FINAL-CAMPAIGN-RESULTS.md). |
| Baige environment maintenance | `environment-cleanup-baige-20261029-r1` | **2026-10-29 18:17 Shanghai: complete.** Current project reduced from5 UV virtual environments +2 overlays to1 environment +2 overlays. Four closed Can/ML10-baseline/RoboTwin envs preserved in an819MiB verified recovery archive before removing expanded copies. Seven legacy shared envs and one overlay, managed Python, uv cache, inputs and checkpoints retained. [Inventory and restore](environment-cleanup-baige-20261029-r1/README.md). |
| Meta-World released pi0.5 noise screen | Baige **`job-orgyjb0z9h6d`**, source `4d4f46f`; run `metaworld-pi05-noise-baige-20261029-r2` | **2026-10-29 20:17:43 Shanghai: stopped at user request; monitor PAUSED.** Three closed544-row conditions preserved (1,632episodes); push-hidden incomplete/unflushed. Two reach conditions independently audited: visible G1+6.25pp/G2=0pp; hidden G1−3.125pp/G2−12.5pp; both lack coverage of goals1/3. This rejects the tested16-arm bank for cached selection; it does not test learned continuous DSRL. No final summary or end-of-run motor hash; no training/new allocation. [Pause and preservation](metaworld-pi05-noise-baige-20261029-r2/PAUSED.md). Next: remain paused; user shifted focus to LIBERO/RoboTwin learned steering. |
| LIBERO / RoboTwin learned steering focus | Existing owners: `Plan VLA in-context learning` and `Clarify DSLR online RL post-training`; both20-minute continuation monitors ACTIVE | **2026-10-29 23:20 Shanghai:** confirmed1M executed-action cap PER benchmark, shared across SAC/freshPPO, train/validation/settling/retries; older per-learner ceilings superseded for next launches. LIBERO first capped stream Baige `job-p5we8pk45d8i`, source `8ada0c0`,9,500-action slot:40episodes/10cases,4,068actions includingsettling,383chunks;0updates before500-chunk warmup. Prior source train/validation had2,220/2,400updates; history integrationr3 exited0. RoboTwinr4 exited0 with32episodes/11,475sourceactions,0optimizerupdates, fullrecovery. Capture gate `job-lspkxidy3haj` exited0; independent saved-trace audit passes two tested seeds (timeout+earlysuccess),2.18× paired speedup,1,551qualificationactions charged/no learning credit. Next: verify LIBERO postwarmup updates and continue budgeted source/PPO work; finish RoboTwin ledger/raw-data integration and measured continuation proposal. Existing ownership retained; no duplicate allocation. [Current evidence and next steps](libero-robotwin-focus-20261029-r1/README.md). |
| Native pi0.5 CTM/XL PPO | Baige `job-61p1rncjnr1x`; `c3e9cca68b93b6e777cbe5314e130bcaf85c8e14` | **21:48: both workers exited 0**, capped at **50/100 updates**, 32,000 training simulator steps and 200 optimizer steps each. Final normal/wiped/blind: CTM **75.0/50.0/62.5%**, XL **50.0/62.5/75.0%**, only two K4 trials/condition. Full optimizer/sampler/RNG checkpoints and frozen motor audited. Next: preserve partial-run evidence; no extension or reliable ICRL/architecture claim. [Final audit](pi05-live-bridge-poc-baige-20261028-r1/FINAL-AUDIT-20261028.md). |
| Balanced native support/fresh-probe gate | Baige `job-37q79sp1o23m`; `5ac79dce8ec8bbc962a93d623cf0f56b5b74d73a` | **21:48: complete, exit 0**, all four fresh-probe methods, 228 saved-artifact checks passed, unchanged frozen motor. Privileged learned bridge **3/6**, zero **3/6**, external oracle **4/6**: interface competence remains unresolved. Normal never beats reset memory on success-once. Zero-control outcomes differ across methods despite matching initial prefix/noise; next gate adds repeated zero controls and complete trajectory hashes. Longer paired PPO held. [Audit and raw evidence](pi05-support-probe-baige-20261028-r1/AUDIT-20261028-2148.md). |
| Privileged bounded tracker with repeated controls | Baige `job-o16q3eimo8bq`; `c2c47152303ccfc1ac469fd3e1ad1e1344201e42` | **22:25: complete/audited, exit 0**, 826.63 driver seconds. Tracker **4/6**, zero **3/6**, external oracle **4/6**; tracker **0/2 at -45 degrees**, so the prespecified capacity gate remains unresolved. All three zero repeats match complete trajectories exactly; frozen motor unchanged, final checkpoint/RNG audited. 151 Linux tests and 297 independent artifact checks passed. Actual cost 2,880 simulator steps, 928 prefix/3,712 suffix queries. Crossed diagnostic not admitted; no PPO. Narrow monitor closed after audit. Next: diagnose negative-rotation steering using saved traces before further training. [Result and audit](pi05-tracker-gate-baige-20261028-r2/RESULTS.md). |
| RoboTwin GRPO learning gate | Baige `job-vh9lp7fvgf9n`; `5484bb31c9cfc6f2fbb5aaf926281861a708603d` | **19:23:** submitted 19:19:50; allocation probe and four focused tests passed. Initializing, no saved training metrics or terminal result yet. One GPU/20-minute cap; group size 8, reward filtering, 64 trajectories, horizon 400, one update. Next: require retained mixed groups, signed advantages and a finite nonzero gradient. Earlier all-success smoke had zero gradients. |
| MetaWorld bounded pilot r2 | Baige `job-x2x4kdpfn011`; `adf741d6e955a624e9f785b9d1572dabf9ff4ab1` | **19:23:** probe passed; first ML1 push GRU cell saved an initial evaluation at 19:22, zero success before training. Six-cell pilot, no terminal receipt. Next: verify advancing updates and train/test adaptation curves within the original one-GPU/180-minute contract. |
| Direct native-context pilot r3 | Baige `job-8kkjhb65px67`; `b5d97b0a7c0f6aed29660a96e2ab10cce9610915` | **19:23: failed, exit 1.** Saved error: `Require finite public feedback, binary reward/boundary and normalized clocks`. No learning result. Next: diagnose feedback validation against saved evidence before a separately identified recovery; preserve r3. |
| CMM full few-shot calibration | Secondary school cluster (thu2), **48876**, source `b6919d6`, K20 seed0 | **19:22 upstream check:** dev query accuracy **15.4% at epoch 350**, MSE .091; update 58.5k/157k. Keep running under the existing contract; about 7.5 hours remained at that check. [Full calibration record](#cmm-calibration). |
| CMM decay diagnostics | Secondary school cluster (thu2), **48983 / 48984**, source `bd38dd0` | **19:22 upstream record:** queued (Priority), retained by the user. Two 50-epoch runs compare decay starts 1 and uniform[0,3]. Verify probes/tests after startup and compare matched-epoch metrics; leave 48876 untouched. [Full diagnostic record](#cmm-calibration). |
| LIBERO-Plus/PRO and Meta-World noise steering | Plan `libero-noise-icrl-20261029-r1`; code on `main`; SFT drafts on branch `motor-sft-draft` | **2026-10-29: Meta-World launched; see its dedicated row. LIBERO work has separate ownership in the frozen-VLA campaign. Not on the thu clusters (user).** (1) Meta-World, released `RLinf/RLinf-Pi05-MetaWorld-SFT` rev `60b2d03`: one GPU, at most 3 h, reach-v3 first, both goal visibilities; env = RLinf `e1a82eb` + `metaworld==3.1.1` (pins `mujoco==3.3.0`, `gymnasium>=1.1`, unlike the existing RLinf envs); [command](libero-noise-icrl-20261029-r1/PLAN.md#running-the-meta-world-screen-gpu-approval-needed). (2) LIBERO-Plus/PRO L0, released `RLinf-Pi05-LIBERO-130-fullshot-SFT`: one GPU, at most 6 h, ~1,080 episodes; env = RLinf `e1a82eb` + `rlinf-libero==0.1.3`, `rlinf-liberoplus>=0.1.3`, `rlinf-liberopro` and assets; [command](libero-noise-icrl-20261029-r1/PLAN.md#libero-pluspro-contents-and-the-l0-screen). Run `tests/test_metaworld_pi05.py`, `tests/test_libero_variants.py`, `tests/test_libero_split.py` in each env first (real-env tests must not skip there). Finding: Plus/PRO perturb only held-out LIBERO-10 tasks (not LIBERO-90), so controller training conditions need a separate decision after L0. |
| DSRL tests (noise-mode Stage 0) | Code `352edc4`; CPU suite thu2 **49206**; Baige smoke **`job-gjkse1sb5gyj`** | **2026-10-29 12:59 Shanghai: terminal/audited.** Upstream thu2 suite reports 458 passed/17 skipped/27 subtests in both environments. Baige independently passed 31 focused Linux tests in 8.82 s and completed S0a 68/68 episodes, frozen motor unchanged. Only 4/32 fixed-arm trajectory pairs and 29/32 success outcomes match; 5/32 first chunks match. Noise and scene identity hashes verified. Deliberate exit 3 stopped S0b/S0c; no learner training or active job. Next: instrument initial observations and control reset seeds; budget repeated outcomes if variation persists. [Results and evidence](noise-mode-icrl-20261029-s0-r1/RESULTS.md); [how to run](#dsrl-tests-how-to-run). |

Check unfinished work on the existing 20-minute cadence, or earlier on a
meaningful result/failure; use each owning monitor's current schedule. Preserve full
optimizer/sampler/RNG recovery state and immutable run identities.

## DSRL tests: how to run

The DSRL tests are Stage 0 of the
[noise-mode plan](noise-mode-icrl-20261029-r1/PLAN.md): is there headroom for
reward-driven, fixed-weight selection among frozen pi0.5 noise arms, beyond
what observation-conditioned DSRL could learn? They do not yet run DSRL
itself; the DSRL static and per-condition baselines come after the gates pass.

1. **CPU tests** (no ask needed). On thu2, pin a worktree to the commit and run
   the suite in both existing environments, inside a CPU-only Slurm step:

   ```bash
   R=/cluster/nvme5/zhujiayang/fluffy-bassoon-2; ID=fb2-cpu-tests-<short>-<date>-r<N>
   git -C $R/repo fetch origin && git -C $R/repo worktree add --detach $R/worktrees/$ID <commit>
   sbatch -J $ID -p RTX3090 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:40:00 \
     -o $R/outputs/$ID/slurm-%j.out --wrap "srun --ntasks=1 $R/outputs/$ID/run.sh"
   ```

   `run.sh` (copy from `outputs/fb2-cpu-tests-352edc4-20261029-r1/` on thu2)
   checks the clean commit and runs `pytest -q tests` with
   `envs/lycoreco-de5277f-r1` and `envs/rlinf-pi05-maniskill-e1a82eb-r1`
   under `/cluster/nvme5/zhujiayang/fluffy-bassoon/`, writing
   `pytest-<env>.log` and `exit-codes.txt`. The noise-specific tests are
   `tests/test_maniskill_noise.py`.

2. **GPU screen** (needs approval: one exclusive Baige GPU, at most 2 hours).
   In the allocated container, create `/mowenjie/fluffy-bassoon-2/worktrees/<run-id>`
   at the reviewed commit, then:

   ```bash
   bash outputs/noise-mode-icrl-20261029-r1/launch-s0.sh <40-hex commit> noise-mode-icrl-20261029-s0-r<N>
   ```

   [launch-s0.sh](noise-mode-icrl-20261029-r1/launch-s0.sh) checks the source
   and the RLinf loader hash, then runs the noise and ManiSkill CPU tests in
   the container. It then runs S0a (determinism, 68 rows), S0b (four hidden
   friction:density variants, 1,088 rows) and S0c (object identity, 816
   rows), with time budgets of 900 s, 3,600 s and 2,400 s, and finally the
   gate analysis. Outputs go to `/mowenjie/fluffy-bassoon-2/outputs/<run-id>/`.

3. **Read the result.** Each `s0*/gates.json` holds G1–G4 per condition and in
   summary, plus `determinism` and `mode_effect`; `gate-failures.txt` lists
   any analysis that errored. To rerun the analysis on CPU from saved files:

   ```bash
   python -m fluffy_bassoon.meta_rl.noise_screen --cache-dir <run>/s0b --condition variant \
     --train 0,1,2,3 --evaluate 0,1,2,3 --output <run>/s0b/gates.json
   ```

   Proceed to Stage 1 only if S0a is deterministic, the arms change actions,
   physics variants change success, and S0b (or S0c) passes all four
   [gates](noise-mode-icrl-20261029-r1/PLAN.md#gates-fixed-before-any-learner-training).
   Otherwise record the negative headroom result.

## CMM calibration

Paper target (arXiv 2610.07907, few-shot linear regression, K=20): CMM
30.63 ± 0.24% query accuracy. Accuracy counts |error| < 0.05 on unit-variance
targets, so predicting the mean scores about 4%. Code:
`src/fluffy_bassoon/meta_rl/fewshot_regression.py`. Runs live under
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/{worktrees,outputs}/<run-id>/` on thu2.

| Run | State |
| --- | --- |
| Full calibration, CMM K=20 seed 0: thu2 job 48876, `fb2-cmm-cal-k20-s0-b6919d6-20261028-r1`, source `b6919d6`, node02 RTX 3090, user-approved | **Complete, failed calibration.** Exit 0, 157,000 updates, 12.2 h. Final 10,000-episode test: query accuracy **4.55%** (MSE .758) vs paper 30.63 ± 0.24%; predicting the mean scores ~4%. Dev curve: chance through epoch 200, breakout to a peak of 16.5% (MSE .081) at epoch 450, then unstable: collapses to chance at 625–750 and 875–1000 with a brief recovery to 15.5% at 800–825 (dev accuracy at 500/550/600/650/700/750/800/850/900/950/1000: 15.4/11.5/9.5/4.0/3.8/4.4/15.3/8.4/5.4/4.6/4.6%). Sync decay rates stayed near 0 (max r .29 at update 57k). Even the peak is half the paper's number, and training does not hold it at lr 1e-4 / clip 5. Not a calibrated port; no CMM comparison may be interpreted. Next: diagnose before any new full run (ask the user first) |
| Decay diagnostic, packed: thu2 job 49178, `fb2-cmm-decay-pair-k20-s0-bd38dd0-20261029-r1` (`run-r1/` with `--decay-init 1`, `run-u03/` with `uniform:0,3`), source `bd38dd0`, user-approved, packed on one GPU at the user's request | **Complete, no clear speedup.** Exit 0, 39 min 48 s on thu2 node02; probe passed, 29 CPU tests passed. Neither reached the <0.8 MSE speedup bar by epoch 50. r0=1: flat at MSE ~.956 through epoch 40, then .921 / .914 at epochs 45 / 50 (final 10,000-episode test MSE .908, accuracy 4.07%). That is the same first small step 48876 took at epoch ~150 (.957 → .910), so it may arrive ~3x earlier; one seed, weak evidence. r0~U[0,3]: no movement (MSE .956 at 50, test accuracy 3.88%). 48876 control: .9575 at 25, .9565 at 50. The decay start is not what keeps the CMM at half the paper's accuracy or makes it collapse. No further runs launched |
| CMM structure on RL benchmarks: `cmm-structure-icrl-20261028-r1`, [plan](cmm-structure-icrl-20261028-r1/PLAN.md) | **Planned, not approved for launch.** Gated on 48876 passing. Bandit K64/K128 and Dark Room with an attempt-length sweep; CMM vs `--ltm-tokens 0` vs `--core ctm` vs GRU/XL at matched parameters; AD/DPT supervised and meta-PPO; CMM-specific lesions, linear probes and 4→8/16 attempt extrapolation. ML10 deferred. Code on `main`: `screen --variant-set cmm-structure` (grid tasks cherry-picked from `claude/icrl-grid-screen`), `evaluate --lesions/--attempts/--export-states`, `meta_rl.probe`, `warmstart_train --core cmm|ctm --objective ad_noise`. CPU tests at `953a980` passed in thu2 job 49036 (node01, clean worktree `fb2-cpu-tests-953a980-20261028-r1`): 372 passed, 17 CUDA-only skips, in both the lycoreco and rlinf-pi05 environments. Next: a small allocated qualification screen (`--variant-set cmm-structure --task darkroom`) once 48876 passes; ask before the full matrix |

The calibration rows above preserve the concurrent `main` update at
`1a21c86cd2275244ffc91edde5c36efc8aa39b9f` (last check 19:22 Shanghai).
Our 19:23 SSH attempt could not resolve the required alias; the fresh school
status comes from that upstream coordination record, not a new direct check.

## Recently completed work

- **CMM cached screen:** Baige `job-qp353vjtueol`, source
  `2bb88a9899919031cd3b5ba8a1b8b7f812502b12`, **500/500, exit 0**.
  Final greedy normal/wiped/blind **54.30/42.38/37.74%**; sampled
  **54.35/41.80/37.45%**; 424.2-second worker wall time. One cached seed;
  paper calibration remains unresolved. Next: independently audit final
  reward arrays and checkpoint hashes before promoting results.
- **Cached CTM/XL/GTrXL comparison:** Baige `job-r0t43xdk6bg2`, source
  `07fb91dce14b56334e4ea639ef22064191187afd`, complete and audited at 18:55.
  Each reached 500 updates; greedy normal/wiped/blind: CTM **57.37/45.17/37.35%**,
  XL **58.15/45.46/37.26%**, GTrXL **54.79/44.68/36.82%**.
  CTM used about 5.21x XL worker wall time. One cached seed establishes
  neither a CTM advantage nor live robot ICRL.
- **DPT warm starts:** seven compact and one native candidate saved;
  none promoted. Native Baige `job-l3coajmm9noh`, source
  `79da19500f4e898d078539b4c562c150e610c16a`, exited 0 at 18:46:56.
  [Compact audit](ctm-dpt-warmstart-20261028-r1/RESULTS.md);
  [native audit](ctm-dpt-warmstart-20261028-r2/RESULTS.md).
  Coverage and tiny probe counts prevent reward-driven ICRL or downstream-PPO claims.
- **Legacy cutover prerequisites:** native motor gates/headroom, the
  one-epoch CMM cost smoke and compact persistent-CTM T8 diagnostics are
  recorded complete in the source-repository coordination record.
  Keep those results with their original sources and artifacts.
