# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-10-31 Asia/Shanghai**.
Keep current decisions and jobs here, method contracts in `docs/`, verified
results in `EVALUATION.md`, and dated plans and evidence in
`outputs/<run-id>/`. Follow [AGENTS.md](../AGENTS.md) and the
[retention policy](../README.md#artifact-retention).

Earlier state, including the fallback ideas, run order, DSRL how-to, CMM
calibration table and completed 2026-10-28/29 jobs, is in the
[2026-10-31 snapshot](archive/coordination/coordination-20261031.md). The
[cutover placeholder](archive/coordination/coordination-pre-cutover-20261028.md)
and [source-repository records](https://github.com/tonyzhu163/fluffy-bassoon/blob/lycoreco_rl/outputs/coordination.md)
are older.

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
  and valid history interventions.

Contracts: [support/probe](../docs/ctm-support-probe.md),
[feature bridge](../docs/pi05-feature-bridge.md),
[PPO replay](../docs/pi05-bridge-policy.md).
Results: [meta-RL evaluation](../src/fluffy_bassoon/meta_rl/EVALUATION.md).

### Diagnosis and proposed redesign (2026-10-31, awaiting user decision)

No robot comparison so far shows memory or adaptation, and the setups could
not have shown it. The motor fixes almost every outcome (oracle about one case
above native), source histories contain almost no failure-then-success, the
learners respond to history observations but not reward, and panels are no
larger than the run-to-run noise. Proposed order: H0 detectable-effect audit
from saved data (CPU, can start now); H1 oracle headroom screen over
semantic options, per-chunk noise and residual; H2 histories that contain
learning; H3 GRU RL²/DPT with reward-shuffle and reset controls; H4 CTM vs
XL/GRU. Proposed pauses, not yet decided: LIBERO AD/DPT scaling on the current
corpus and the RoboTwin final100 export.
[Diagnosis and plan](icrl-headroom-diagnosis-20261031-r1/PLAN.md).

## Coordination rules in force

From the [2026-10-31 review](archive/coordination/coordination-20261031.md#coordination-review-2026-10-31):

- Push coordination edits to `main`, rebased, editing only your own row.
  Branch-only edits leave this file stale.
- One or two readable sentences per row: job ID, last check, progress, next
  action and a link. Put hashes, byte counts and budget ledgers in the
  linked run record.
- Budget: roughly **1M source actions per learner** (human, 2026-10-30),
  superseding "1M per benchmark, shared". The RoboTwin owner should confirm
  which ledger the R4 export and deferred AD work are charged against.
- Native VLRLA PPO is the shared priority; `frozen-vla-poc`
  (`finish_ppo`/`finish_assembly`) is its sole launcher.

## Ongoing work

Rows are as last recorded by their owners; none was re-queried for this
update.

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Native VLRLA-recipe PPO pilot | Run `vlrla-native-ppo-pilot-20261031-r1`; branch `codex/frozen-vla-poc` | **2026-10-31 12:58 Shanghai:** specified at 6 GPU-hours / 1M actions; 8 GPUs preferred, or 4 GPUs with 16-step accumulation preserving the batch. CPU qualification pending; no PPO job yet. |
| LIBERO frozen-VLA steering | Branch `codex/libero-fixed100-closure-20261031` | **2026-10-31 12:58 Shanghai:** fixed100 phase closed. Native 28/90, AD full/static 29, DPT 28; no memory or sample-efficiency gain. AD100/500 proposal held. [Fixed100 result](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/libero-fixed100-closure-20261031/outputs/libero40-fixed100-target-plan-20261031-r1/RESULT.md). |
| RoboTwin final100 AD export (R4) | Baige `job-vth2qv6r4lxu`, CPU only; branch `codex/robotwin-native-wholetrials` | **2026-10-31 13:30 Shanghai:** running, training export in progress, no indices yet. New AD GPU work deferred for native PPO. Original 100-round campaign complete with no memory advantage. [Current handoff](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/robotwin-native-wholetrials/outputs/robotwin-final100-ad-export-20261031-r1/CURRENT.md); [campaign result](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/codex/robotwin-native-wholetrials/outputs/robotwin-trained-continuation-20261031-r1/FINAL-CAMPAIGN-RESULTS.md). |
| Meta-World released pi0.5 noise screen | Baige `job-orgyjb0z9h6d`; run `metaworld-pi05-noise-baige-20261029-r2` | **2026-10-29 20:17 Shanghai: paused at user request.** Reach screens fail headroom; goals 1 and 3 never reached. Remain paused. [Pause record](metaworld-pi05-noise-baige-20261029-r2/PAUSED.md). |

### Needs owner refresh

Last recorded 2026-10-28 19:23 Shanghai; state since then unknown. Owners:
confirm terminal state and move each to completed work.

- RoboTwin GRPO learning gate, Baige `job-vh9lp7fvgf9n`: initializing, one
  GPU / 20-minute cap.
- MetaWorld bounded pilot r2, Baige `job-x2x4kdpfn011`: first evaluation
  saved, one GPU / 180-minute cap.
- CMM decay diagnostics thu2 48983/48984: last recorded queued. The packed
  diagnostic 49178 below covers the same comparison; confirm these were
  cancelled.

## Recently completed work

Details for each are in the [snapshot](archive/coordination/coordination-20261031.md).

- **CMM calibration**, thu2 48876: complete, failed (4.55% vs paper 30.63%);
  optimization instability. Packed decay diagnostic thu2 49178: no clear
  speedup. CMM structure tests stay gated on calibration.
- **DSRL Stage 0 smoke**, Baige `job-gjkse1sb5gyj`: repeat determinism failed
  (4/32 identical trajectories); S0b/S0c not run.
- **Native-context pilot r3**, Baige `job-8kkjhb65px67`: failed feedback
  validation, exit 1; preserved.
- **Native pi0.5 CTM/XL PPO, support/probe gate, privileged tracker**:
  complete and audited 2026-10-28; no ICRL or architecture claim.
- **Cached CTM/XL/GTrXL, CMM cached screen, DPT warm starts**: complete; one
  cached seed, no CTM advantage.
- **Baige environment cleanup**: complete 2026-10-29.
