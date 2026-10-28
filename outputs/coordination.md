# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-10-28 21:59
Asia/Shanghai**. Keep current decisions and jobs here, method contracts in
`docs/`, verified results in `EVALUATION.md`, and dated evidence in
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
| Native pi0.5 CTM/XL PPO | Baige `job-61p1rncjnr1x`; `c3e9cca68b93b6e777cbe5314e130bcaf85c8e14` | **21:48: both workers exited 0**, capped at **50/100 updates**, 32,000 training simulator steps and 200 optimizer steps each. Final normal/wiped/blind: CTM **75.0/50.0/62.5%**, XL **50.0/62.5/75.0%**, only two K4 trials/condition. Full optimizer/sampler/RNG checkpoints and frozen motor audited. Next: preserve partial-run evidence; no extension or reliable ICRL/architecture claim. [Final audit](pi05-live-bridge-poc-baige-20261028-r1/FINAL-AUDIT-20261028.md). |
| Balanced native support/fresh-probe gate | Baige `job-37q79sp1o23m`; `5ac79dce8ec8bbc962a93d623cf0f56b5b74d73a` | **21:48: complete, exit 0**, all four fresh-probe methods, 228 saved-artifact checks passed, unchanged frozen motor. Privileged learned bridge **3/6**, zero **3/6**, external oracle **4/6**: interface competence remains unresolved. Normal never beats reset memory on success-once. Zero-control outcomes differ across methods despite matching initial prefix/noise; next gate adds repeated zero controls and complete trajectory hashes. Longer paired PPO held. [Audit and raw evidence](pi05-support-probe-baige-20261028-r1/AUDIT-20261028-2148.md). |
| Privileged bounded tracker with repeated controls | Baige `job-o16q3eimo8bq`; `c2c47152303ccfc1ac469fd3e1ad1e1344201e42` | **21:59: running**, submitted 21:54:34; **151 focused Linux tests passed** in 20.02 s. Allocated GPU/renderers verified on PCI a2:00.0; first two-object calibration episode complete, next trajectory advancing, residuals within bound. One GPU/60-minute cap, tracker 1,500 s plus conditional crossed diagnostic 1,800 s; no PPO. Next: inspect repeated zero trajectory stability and held-out capacity before admitting crossed histories. [Exact launch](pi05-tracker-gate-baige-20261028-r2/LAUNCH.json), [plan](pi05-tracker-gate-baige-20261028-r2/PLAN.md). |
| RoboTwin GRPO learning gate | Baige `job-vh9lp7fvgf9n`; `5484bb31c9cfc6f2fbb5aaf926281861a708603d` | **19:23:** submitted 19:19:50; allocation probe and four focused tests passed. Initializing, no saved training metrics or terminal result yet. One GPU/20-minute cap; group size 8, reward filtering, 64 trajectories, horizon 400, one update. Next: require retained mixed groups, signed advantages and a finite nonzero gradient. Earlier all-success smoke had zero gradients. |
| MetaWorld bounded pilot r2 | Baige `job-x2x4kdpfn011`; `adf741d6e955a624e9f785b9d1572dabf9ff4ab1` | **19:23:** probe passed; first ML1 push GRU cell saved an initial evaluation at 19:22, zero success before training. Six-cell pilot, no terminal receipt. Next: verify advancing updates and train/test adaptation curves within the original one-GPU/180-minute contract. |
| Direct native-context pilot r3 | Baige `job-8kkjhb65px67`; `b5d97b0a7c0f6aed29660a96e2ab10cce9610915` | **19:23: failed, exit 1.** Saved error: `Require finite public feedback, binary reward/boundary and normalized clocks`. No learning result. Next: diagnose feedback validation against saved evidence before a separately identified recovery; preserve r3. |
| CMM full few-shot calibration | Secondary school cluster (thu2), **48876**, source `b6919d6`, K20 seed0 | **19:22 upstream check:** dev query accuracy **15.4% at epoch 350**, MSE .091; update 58.5k/157k. Keep running under the existing contract; about 7.5 hours remained at that check. [Full calibration record](#cmm-calibration). |
| CMM decay diagnostics | Secondary school cluster (thu2), **48983 / 48984**, source `bd38dd0` | **19:22 upstream record:** queued (Priority), retained by the user. Two 50-epoch runs compare decay starts 1 and uniform[0,3]. Verify probes/tests after startup and compare matched-epoch metrics; leave 48876 untouched. [Full diagnostic record](#cmm-calibration). |

Check unfinished work on the existing 20-minute cadence, or earlier on a
meaningful result/failure; use each owning monitor's current schedule. Preserve full
optimizer/sampler/RNG recovery state and immutable run identities.

## CMM calibration

Paper target (arXiv 2610.07907, few-shot linear regression, K=20): CMM
30.63 ± 0.24% query accuracy. Accuracy counts |error| < 0.05 on unit-variance
targets, so predicting the mean scores about 4%. Code:
`src/fluffy_bassoon/meta_rl/fewshot_regression.py`. Runs live under
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/{worktrees,outputs}/<run-id>/` on thu2.

| Run | State |
| --- | --- |
| Full calibration, CMM K=20 seed 0: thu2 job 48876, `fb2-cmm-cal-k20-s0-b6919d6-20261028-r1`, source `b6919d6`, node02 RTX 3090, user-approved | **Running, improving.** At chance through epoch 200 (MSE .907, accuracy 4.2%), then broke out: dev query MSE .714 / .444 / .143 / .096 / .091 and accuracy 4.6 / 6.1 / 11.7 / 14.8 / 15.4% at epochs 250 / 275 / 300 / 325 / 350. Last check 2026-10-28 11:22 UTC, update 58.5k of 157k; about 7.5 h left. Sync decay rates are still near 0 (max r .29 at update 57k), so the whole-episode average did not block learning. The user's rule was to cancel it if it had not improved; it improved, so it runs to completion. **Pass:** final 10,000-episode test within the paper's spread. Next: monitor; on completion read `run/summary.json` and report |
| Decay diagnostic: thu2 jobs 48983 (`fb2-cmm-decay-r1-k20-s0-bd38dd0-20261028-r2`, `--decay-init 1`) and 48984 (`…-u03-…-r2`, `--decay-init uniform:0,3`), source `bd38dd0` | **Queued (Priority), kept by the user.** User approval, verbatim: "Approved: implement --decay-init in fluffy-bassoon-2, run the CPU tests, and launch the two 50-epoch decay diagnostics (r0=1 and r0~U[0,3]) on thu2 under new run IDs. Leave 48876 running." CMM K=20 seed 0, 50 epochs, dev every 5 epochs, 3 h cap, RTX3090/4090D/6000D, `--exclude=node01,node02`. Jobs 48981/48982 were refused by the GPU probe because thu2 node02 GPU `fef80646` runs a foreign process outside Slurm; no training ran. Each step runs the probe, then the fewshot/CMM CPU tests (`tests.log`), then training. Question: does a higher decay start shorten the plateau? Compare dev MSE with 48876 (.9575 at epoch 25, .9565 at 50). Below 0.8 by epoch 50 means a speedup; report it, and ask before any full-length run with that init. Near .95 means no speedup within 50 epochs, not a failure |
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
